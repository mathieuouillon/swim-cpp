// CLAS12 v_z / PID diagnostic analysis (C++ port of the Rust analysis).
//
// Reads CLAS12 HIPO files with hipo4, matches reconstructed tracks to MC truth
// (MC::RecMatch) for charged pions/muons, swims them back through the CLAS12
// field, and writes v_z / PID diagnostic histograms to a ROOT file.
//
// Parallelism is by PROCESS, not thread: with --jobs N the program shards the
// input files across N single-threaded copies of itself (self re-exec), each
// writing a partial ROOT file, then merges them (see process_pool.hpp). This is
// the farm-friendly model — independent processes scale across allocated cores
// with no ROOT thread-safety locking or shared global state.
//
// Usage:
//   swim-analysis <input>... [--output FILE] [--jobs N] [--quiet]
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <glob.h>

#include <fmt/format.h>

#include "TH1.h"
#include "analysis.hpp"
#include "argparse/argparse.hpp"
#include "constants.hpp"
#include "field.hpp"
#include "hipo4/hipo.hpp"
#include "hipo_chain.hpp"
#include "process_pool.hpp"

namespace {

struct cli_args {
    std::vector<std::string> inputs;
    std::string output = "output/cpp/vz.root";
    int jobs = 0;  // 0 = one worker process per hardware thread
    bool quiet = false;
    std::string torus = "Full_torus_r501_phi361_z501_31Mar2021.dat";
    std::string solenoid = "Symm_solenoid_r601_phi1_z1201_21May2019.dat";
    double torus_scale = -1.0;     // inbending (sign to verify)
    double solenoid_scale = -1.0;
    double solenoid_z_shift = -3.0;  // move the solenoid map by -3 cm in z
    // Hidden: set by the supervisor on each worker re-exec. Non-empty => this
    // process is a worker — stay silent and write its counts sidecar here.
    std::string emit_counts;
    long long record_begin = -1;  // hidden: worker's global HIPO record range [begin, end)
    long long record_end = -1;
    std::string progress_file;  // hidden: shared progress-counter path (supervisor -> worker)
};

// Build the command-line parser (external/argparse). Hidden flags (--emit-counts)
// are accepted but kept out of --help; the supervisor sets them on each re-exec.
auto build_parser() -> argparse::parser {
    argparse::parser p("swim-analysis",
                       "CLAS12 v_z / PID diagnostic: match reco tracks to MC truth, swim them "
                       "back through the CLAS12 field, and histogram v_z / PID diagnostics.");
    p.add_argument("inputs")
        .nargs(argparse::nargs::one_or_more)
        .metavar("input")
        .help(".hipo files, globs, directories, or @list files");
    p.add_argument("-o", "--output")
        .default_value("output/cpp/vz.root")
        .help("output ROOT file");
    p.add_argument("-j", "--jobs", "-t", "--threads")
        .default_value(0)
        .help("worker processes, 0 = one per hardware thread (--threads is an alias)");
    p.add_argument("-q", "--quiet").flag().help("suppress the progress bar");
    p.add_argument("--torus")
        .default_value("Full_torus_r501_phi361_z501_31Mar2021.dat")
        .help("torus field map");
    p.add_argument("--solenoid")
        .default_value("Symm_solenoid_r601_phi1_z1201_21May2019.dat")
        .help("solenoid field map");
    p.add_argument("--torus-scale").default_value(-1.0).help("torus field scale (inbending = -1)");
    p.add_argument("--solenoid-scale").default_value(-1.0).help("solenoid field scale");
    p.add_argument("--solenoid-z-shift").default_value(-3.0).help("solenoid map z-shift [cm]");
    p.add_argument("--emit-counts").default_value("").hidden();  // worker counts sidecar path
    p.add_argument("--record-begin").default_value(static_cast<long long>(-1)).hidden();
    p.add_argument("--record-end").default_value(static_cast<long long>(-1)).hidden();
    p.add_argument("--progress-file").default_value("").hidden();
    return p;
}

auto looks_like_glob(const std::string& s) -> bool {
    return s.find_first_of("*?[") != std::string::npos;
}

auto expand_glob(const std::string& pattern) -> std::vector<std::string> {
    std::vector<std::string> out;
    glob_t g{};
    int rc = glob(pattern.c_str(), 0, nullptr, &g);
    if (rc == 0) {
        for (std::size_t i = 0; i < g.gl_pathc; ++i) out.emplace_back(g.gl_pathv[i]);
    }
    globfree(&g);
    return out;
}

auto trim(const std::string& s) -> std::string {
    auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Expand the CLI inputs into a flat list of file paths: `@list` files are read
// line by line, globs are expanded, directories contribute their `*.hipo`
// files, and anything else is taken verbatim.
auto collect_inputs(const std::vector<std::string>& inputs) -> std::vector<std::string> {
    std::vector<std::string> paths;
    for (const auto& arg : inputs) {
        if (!arg.empty() && arg[0] == '@') {
            std::string list = arg.substr(1);
            std::ifstream f(list);
            if (!f) {
                std::fprintf(stderr, "warning: cannot open file list '%s'\n", list.c_str());
                continue;
            }
            std::string line;
            while (std::getline(f, line)) {
                std::string t = trim(line);
                if (t.empty() || t[0] == '#') continue;
                paths.push_back(t);
            }
        } else if (looks_like_glob(arg)) {
            auto g = expand_glob(arg);
            paths.insert(paths.end(), g.begin(), g.end());
        } else if (std::filesystem::is_directory(arg)) {
            std::string pattern = arg;
            while (!pattern.empty() && pattern.back() == '/') pattern.pop_back();
            pattern += "/*.hipo";
            auto g = expand_glob(pattern);
            paths.insert(paths.end(), g.begin(), g.end());
        } else {
            paths.push_back(arg);
        }
    }
    return paths;
}

// Bank names the analysis reads. Required: REC::Particle, MC::Particle,
// MC::RecMatch. The rest are optional (absent -> that variable is skipped).
const std::vector<std::string> WANTED_BANKS = {
    "REC::Particle", "MC::Particle",    "MC::RecMatch",  "MC::True",
    "REC::Track",    "REC::Traj",       "REC::CovMat",   "REC::Scintillator",
    "REC::ScintExtras", "REC::Cherenkov", "REC::Calorimeter"};

// Keep only the wanted banks that exist in the first file's dictionary (so the
// banklist construction never throws on a missing schema, e.g. MC banks in real
// data). Absent banks map to a -1 index in bank_index -> nullptr in fill_event.
auto present_banks(const std::string& first_file) -> std::vector<std::string> {
    std::vector<std::string> present;
    auto f = hipo::file::open(first_file);
    if (!f) {
        std::fprintf(stderr, "warning: cannot open '%s' to read dictionary: %s\n",
                     first_file.c_str(), f.error().message.c_str());
        return present;
    }
    for (const auto& n : WANTED_BANKS) {
        if (f->find_schema(n) != nullptr) present.push_back(n);
    }
    return present;
}

// Per-event callback for chain::process: fold each event into the worker's one
// analysis accumulator. A named functor (not a lambda), per the project's
// preference for chain/event-processing callables. With single-threaded
// workers there is exactly one accumulator per process.
struct event_filler {
    vz::analysis*        acc;
    const vz::bank_index* idx;
    const vz::magnetic_field*     field;
    auto operator()(vz::bank_list& bl, int /*file_idx*/, long /*event_idx*/) const -> void {
        acc->fill_event(bl, *idx, *field);
    }
};

using FillGrid = std::array<std::array<std::uint64_t, vz::N_SPECIES>, vz::N_SPECIES>;

// The run report, mirroring the Rust output. Printed by the user-facing process
// (a lone --jobs 1 run, or the supervisor after merging the workers).
auto print_summary(const std::string& output, std::size_t n_files, std::uint64_t events,
                   const FillGrid& fills) -> void {
    fmt::print("----------------------------------------\n");
    fmt::print("files            : {}\n", n_files);
    fmt::print("events           : {}\n", events);
    fmt::print("fills [truth/reco]:\n");
    for (std::size_t t = 0; t < vz::N_SPECIES; ++t) {
        for (std::size_t r = 0; r < vz::N_SPECIES; ++r) {
            fmt::print("  {} / {} : {}\n", vz::SPECIES_NAME[t], vz::SPECIES_NAME[r], fills[t][r]);
        }
    }

    const std::size_t n_reco = vz::N_SPECIES * vz::N_SPECIES * vz::N_MOM_BINS;
    const std::size_t n_theta = vz::N_SPECIES * vz::N_SPECIES;
    const std::size_t n_vars = vz::N_SPECIES * vz::N_PION_SPECIES * vz::N_PT_BANDS * vz::N_TRACK_VARS;
    const std::size_t n_res = vz::N_RES_VARS * vz::N_SPECIES * vz::N_SPECIES * vz::N_MOM_BINS;
    const std::size_t n_ploss = 5 * vz::N_PION_SPECIES;
    const std::size_t n_gen = vz::N_PION_SPECIES;
    const std::size_t n_swim = (5 + 4) * vz::N_PION_SPECIES + 3 * vz::N_PION_SPECIES * vz::N_MOM_BINS;
    const std::size_t n_pr1 = 2 * vz::N_PION_SPECIES;
    fmt::print("histograms written to {} (Forward Detector only):\n", output);
    fmt::print("  reco v_z   : {}\n", n_reco);
    fmt::print("  truth v_z  : {}\n", n_reco);
    fmt::print("  dv_z       : {}\n", n_reco);
    fmt::print("  v_z-theta  : {}\n", n_theta);
    fmt::print("  track vars : {}  (1-D, truth x reco-pion, low+high p)\n", n_vars);
    fmt::print("  res grids  : {}  (sigtx/sigty/sigtheta, truth x reco, 12 p-bins)\n", n_res);
    fmt::print("  p-loss     : {}  (true p vtx vs DC R1: ptrue/dp/dp-vs-p/dp-frac, reco pion)\n",
               n_ploss);
    fmt::print("  gen p      : {}  (generated momentum, all primary pions, no reco cut)\n", n_gen);
    fmt::print("  swim       : {}  (true-state + reco-state swim vs reco/true, reco pion)\n", n_swim);
    fmt::print("  p@R1       : {}  (reco |p| vs true p at DC R1, reco pion)\n", n_pr1);
    fmt::print("  total      : {}\n",
               3 * n_reco + n_theta + n_vars + n_res + n_ploss + n_gen + n_swim + n_pr1);
}

// One single-threaded pass over `paths`: open the chain, load the field, fold
// every event into one analysis, and write the ROOT file. In worker mode the
// counts go to a sidecar and all chatter is suppressed; otherwise (a lone
// --jobs 1 run) it prints the inputs line and the full summary itself.
auto run_worker(const cli_args& args, const std::vector<std::string>& paths, bool worker_mode) -> int {
    vz::chain ch(/*progress=*/!args.quiet && !worker_mode, /*verbose=*/false);
    for (const auto& p : paths) ch.add(p);
    try {
        ch.open(/*validate_all=*/true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not open inputs: %s\n", e.what());
        return 1;
    }

    if (!worker_mode)
        fmt::print("inputs: {} file(s), {} events\n", ch.size(), ch.total_events());

    std::unique_ptr<vz::composite_field> field_ptr;
    try {
        field_ptr = std::make_unique<vz::composite_field>(vz::composite_field::load(
            args.torus, args.torus_scale, args.solenoid, args.solenoid_scale,
            args.solenoid_z_shift));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not load field maps: %s\n", e.what());
        return 1;
    }
    const vz::magnetic_field& field = *field_ptr;

    vz::bank_list banks;
    try {
        banks = ch.get_banks(present_banks(paths[0]));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not build banklist: %s\n", e.what());
        return 1;
    }
    vz::bank_index idx(banks);

    vz::analysis result;
    event_filler filler{&result, &idx, &field};
    const std::size_t rec_begin =
        args.record_begin >= 0 ? static_cast<std::size_t>(args.record_begin) : 0;
    const std::size_t rec_end = args.record_end >= 0
                                    ? static_cast<std::size_t>(args.record_end)
                                    : std::numeric_limits<std::size_t>::max();
    // Report progress to the supervisor's shared counter (amortized) when re-exec'd.
    vz::shared_counter prog =
        args.progress_file.empty() ? vz::shared_counter{}
                                   : vz::shared_counter::map(args.progress_file, /*create=*/false);
    std::uint64_t prog_local = 0;
    ch.process(
        banks,
        [&](vz::bank_list& bl, int fi, long ei) {
            filler(bl, fi, ei);
            if ((++prog_local & 0x3FFFU) == 0) prog.add(0x4000);
        },
        rec_begin, rec_end);
    prog.add(prog_local & 0x3FFFU);

    try {
        result.write(args.output);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not write '%s': %s\n", args.output.c_str(), e.what());
        return 1;
    }

    if (worker_mode) {
        vz::run_counts rc;
        rc.events = result.events();
        rc.fills = result.fills();
        try {
            rc.write(args.emit_counts);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error: could not write counts: %s\n", e.what());
            return 1;
        }
    } else {
        print_summary(args.output, ch.size(), result.events(), result.fills());
    }
    return 0;
}

// Supervisor: split the work across `jobs` re-exec'd workers, merge their partial
// ROOT files into args.output, and print the summed summary. With at least `jobs`
// input files each worker takes a subset; with fewer (e.g. a single big file) the
// work is sharded by HIPO record range so one file still parallelizes.
auto run_supervisor(const cli_args& args, char** argv, const std::vector<std::string>& paths,
                    int jobs) -> int {
    const bool by_file = paths.size() >= static_cast<std::size_t>(jobs);

    // Open the inputs once: total events (for the aggregate progress bar) and
    // total records (to record-shard one/few files). On a multi-GB train file
    // this index scan takes a moment, so announce it for immediate feedback.
    if (!args.quiet) {
        fmt::print("swim-analysis: scanning {} input file(s)...\n", paths.size());
        std::fflush(stdout);
    }
    vz::chain probe(/*progress=*/false, /*verbose=*/false);
    for (const auto& p : paths) probe.add(p);
    try {
        probe.open(/*validate_all=*/true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not open inputs: %s\n", e.what());
        return 1;
    }
    const auto total_events = static_cast<std::uint64_t>(probe.total_events());
    const std::uint64_t total_records = probe.total_records();

    int n_workers = jobs;
    if (!by_file) {
        n_workers = static_cast<int>(std::min<std::uint64_t>(
            static_cast<std::uint64_t>(jobs), std::max<std::uint64_t>(1, total_records)));
        if (n_workers <= 1) return run_worker(args, paths, /*worker_mode=*/false);
    }

    const std::string exe = vz::self_exe_path(argv[0]);
    std::string dir;
    try {
        dir = vz::make_temp_dir("swim-analysis");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    // Record sharding hands every worker the full input list (one shared @list)
    // plus a disjoint global record range; file sharding gives each its own list.
    std::string shared_list;
    if (!by_file) {
        shared_list = fmt::format("{}/inputs.list", dir);
        std::ofstream lf(shared_list);
        for (const auto& p : paths) lf << p << '\n';
    }
    const std::uint64_t chunk = by_file ? 0 : (total_records + n_workers - 1) / n_workers;

    const std::string progress_path = fmt::format("{}/progress", dir);
    vz::shared_counter progress = vz::shared_counter::map(progress_path, /*create=*/true);

    std::vector<std::string> parts, counts;
    std::vector<std::vector<std::string>> argvs;
    for (int i = 0; i < n_workers; ++i) {
        const std::string part = fmt::format("{}/part-{:03}.root", dir, i);
        const std::string cnt = fmt::format("{}/part-{:03}.counts", dir, i);
        parts.push_back(part);
        counts.push_back(cnt);
        std::vector<std::string> av = {exe, "--jobs", "1", "--quiet", "--output", part,
                                       "--emit-counts", cnt, "--progress-file", progress_path,
                                       "--torus", args.torus, "--solenoid", args.solenoid,
                                       "--torus-scale", fmt::format("{}", args.torus_scale),
                                       "--solenoid-scale", fmt::format("{}", args.solenoid_scale),
                                       "--solenoid-z-shift",
                                       fmt::format("{}", args.solenoid_z_shift)};
        if (by_file) {
            const std::string list = fmt::format("{}/shard-{:03}.list", dir, i);
            std::ofstream lf(list);
            for (std::size_t j = i; j < paths.size(); j += static_cast<std::size_t>(n_workers))
                lf << paths[j] << '\n';
            av.push_back("@" + list);
        } else {
            const std::uint64_t rb = static_cast<std::uint64_t>(i) * chunk;
            const std::uint64_t re = std::min<std::uint64_t>(rb + chunk, total_records);
            av.insert(av.end(), {"--record-begin", fmt::format("{}", rb), "--record-end",
                                 fmt::format("{}", re)});
            av.push_back("@" + shared_list);
        }
        argvs.push_back(std::move(av));
    }

    fmt::print("swim-analysis: {} worker process(es) sharded by {} over {} -> {}\n", n_workers,
               by_file ? "file" : "record",
               by_file ? fmt::format("{} file(s)", paths.size())
                       : fmt::format("{} records in {} file(s)", total_records, paths.size()),
               args.output);

    // Spawn the workers, then render one aggregate progress bar fed by the shared
    // counter while they run (no progress thread exists at fork time).
    const auto pids = vz::spawn_workers(argvs);
    bool ok = !pids.empty();
    if (ok) {
        std::unique_ptr<progress_tracker> bar;
        if (!args.quiet && total_events > 0) {
            progress_tracker::config cfg;
            cfg.label = "Processing";
            cfg.show_eta = true;
            cfg.show_rate = true;
            bar = std::make_unique<progress_tracker>(static_cast<std::size_t>(total_events), cfg);
            bar->start();
        }
        std::uint64_t last = 0;
        ok = vz::reap_workers(pids, [&] {
            if (!bar) return;
            const std::uint64_t c = progress.load();
            if (c > last) {
                bar->add(static_cast<std::size_t>(c - last));
                last = c;
            }
        });
        if (bar) bar->finish();
    }
    if (!ok) {
        std::fprintf(stderr, "error: one or more workers failed; '%s' not written\n",
                     args.output.c_str());
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        return 1;
    }

    try {
        vz::merge_root_files(parts, args.output);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: merge failed: %s\n", e.what());
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        return 1;
    }

    vz::run_counts total;
    try {
        for (const auto& c : counts) total.add_from_file(c);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "warning: could not read worker counts: %s\n", e.what());
    }

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    print_summary(args.output, paths.size(), total.events, total.fills);
    return 0;
}

}  // namespace

auto main(int argc, char** argv) -> int {
    // Stop histograms registering themselves in the (non-thread-safe) global
    // directory; harmless for the supervisor, required before the worker builds
    // its TH1Ds. No ROOT::EnableThreadSafety: workers are single-threaded.
    TH1::AddDirectory(kFALSE);

    argparse::parser p = build_parser();
    try {
        p.parse(argc, argv);
    } catch (const argparse::help_requested&) {
        fmt::print("{}", p.help_text());
        return 0;
    } catch (const argparse::error& e) {
        fmt::print(stderr, "error: {}\n\n{}", e.what(), p.usage());
        return 2;
    }

    cli_args args;
    args.inputs = p.get<std::vector<std::string>>("inputs");
    args.output = p.get<std::string>("--output");
    args.jobs = p.get<int>("--jobs");
    args.quiet = p.get<bool>("--quiet");
    args.torus = p.get<std::string>("--torus");
    args.solenoid = p.get<std::string>("--solenoid");
    args.torus_scale = p.get<double>("--torus-scale");
    args.solenoid_scale = p.get<double>("--solenoid-scale");
    args.solenoid_z_shift = p.get<double>("--solenoid-z-shift");
    args.emit_counts = p.get<std::string>("--emit-counts");
    args.record_begin = p.get<long long>("--record-begin");
    args.record_end = p.get<long long>("--record-end");
    args.progress_file = p.get<std::string>("--progress-file");

    std::vector<std::string> paths = collect_inputs(args.inputs);
    if (paths.empty()) {
        std::fprintf(stderr, "error: no input files found\n");
        return 1;
    }

    const bool worker_mode = !args.emit_counts.empty();
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int want = args.jobs > 0 ? args.jobs : static_cast<int>(hw);

    // Ensure the output directory (default output/cpp/) exists before any write.
    vz::ensure_parent_dir(args.output);

    // A re-exec'd worker processes its assigned files / record range. A lone run
    // processes everything. Otherwise the supervisor splits across `want` workers
    // — by file when there are enough, else by HIPO record range (one big file).
    if (worker_mode) return run_worker(args, paths, /*worker_mode=*/true);
    if (want <= 1) return run_worker(args, paths, /*worker_mode=*/false);
    return run_supervisor(args, argv, paths, want);
}
