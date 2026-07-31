// Reconstructed v_z in (p, theta) bins for Forward-Detector e-/pi+/pi-, read
// directly from recon HIPO files -- no intermediate particles tree and no swim.
//
// Built to overlay the EB-reconstructed v_z of two reconstruction passes (e.g.
// the same RG-D CuSn simulation reconstructed with torus scale 0.998 vs 1.000)
// cell-by-cell. Run it once per pass to get one ROOT file each, then overlay the
// matching (species, p, theta) cells with plot_rec_vz_compare.py.
//
// Selection mirrors hipo2root's default (the population the vz analysis uses):
// Forward-Detector e-/pi+/pi- with |chi2pid| < 3, the REC::Traj DC-edge fiducial
// (R1/R2 > 5, R3 > 10 cm) and, for electrons, the PCAL lv/lw > 9 cm fiducial.
// The (p, theta) grid is the species-dependent one from vz-swim-hist: electrons
// get 1-GeV bins over [2, 10) and 2-deg theta bins over [6, 26); pions get a
// softer [0.3, 6] GeV grid and 3-deg theta bins over [6, 42).
//
// Per (species, p, theta) cell one histogram is written:
//   vz_rec_<sp>_pAA_BB_thCC_DD   EB-reconstructed v_z  [-5, 5] cm (--vz-min/--vz-max)
// with <sp> in {ele, pip, pim}, AA_BB the momentum-bin edges in GeV and CC_DD
// the theta-bin edges in degrees (both zero-padded; fractional GeV edges keep
// one decimal, matching the vz-swim-hist naming the Python plotters parse).
// Plus, per species, one integrated histogram over the whole selection:
//   vz_rec_<sp>_all             EB-reconstructed v_z  [-5, 5] cm (--vz-min/--vz-max)
// (every accepted track of that species, independent of the (p, theta) grid).
//
// Parallelism is by PROCESS (see process_pool.hpp): with --jobs N the inputs are
// sharded across N re-exec'd single-threaded workers -- by file when there are
// at least N, else by HIPO record range so a few big train files still
// parallelize -- each writing a partial ROOT file the supervisor then merges
// (same-named histograms summed bin-by-bin).
//
// Usage:
//   rec-vz-hist <input.hipo>... [--output FILE] [--jobs N] [--max-events N] [--quiet]
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>  // hardware_concurrency + the parallel input scan
#include <vector>

#include <fmt/format.h>

#include "TFile.h"
#include "TH1D.h"

#include "argparse/argparse.hpp"
#include "bank_access.hpp"
#include "constants.hpp"
#include "hipo4/hipo.hpp"
#include "hipo_chain.hpp"
#include "process_pool.hpp"
#include "progresstracker.hpp"

namespace {

// The three Forward-Detector species this tool histograms, by reconstructed pid.
constexpr std::size_t N_SP = 3;
constexpr std::array<int, N_SP> SP_PID = {11, 211, -211};
constexpr std::array<std::string_view, N_SP> SP_TAG = {"ele", "pip", "pim"};
constexpr std::array<std::string_view, N_SP> SP_TEX = {"e^{-}", "#pi^{+}", "#pi^{-}"};

// Species index (0..2) for a reconstructed pid, or -1 if not one of the three.
inline auto species_of(int pid) -> int {
    for (std::size_t s = 0; s < N_SP; ++s)
        if (SP_PID[s] == pid) return static_cast<int>(s);
    return -1;
}

// (p, theta) grid, both axes species-dependent (make_grid) -- identical to
// vz-swim-hist so the histogram names line up. N_P_MAX / N_TH_MAX bound the
// per-cell arrays; the active grid for a species uses n_p / n_th of them.
constexpr std::size_t N_P_MAX = 8;
constexpr std::size_t N_TH_MAX = 12;

struct pt_grid {
    std::size_t n_p = 8;
    std::array<double, N_P_MAX + 1> p_edges{{2, 3, 4, 5, 6, 7, 8, 9, 10}};
    std::size_t n_th = 10;
    double th_min = 6.0;
    double th_width = 2.0;
    [[nodiscard]] auto p_bin(double p) const -> long {
        for (std::size_t i = 0; i < n_p; ++i)
            if (p >= p_edges[i] && p < p_edges[i + 1]) return static_cast<long>(i);
        return -1;
    }
    [[nodiscard]] auto th_bin(double theta_deg) const -> long {
        const double f = (theta_deg - th_min) / th_width;
        if (f < 0.0) return -1;
        const auto i = static_cast<long>(f);
        return i < static_cast<long>(n_th) ? i : -1;
    }
    [[nodiscard]] auto th_lo(std::size_t tb) const -> double { return th_min + tb * th_width; }
    [[nodiscard]] auto th_hi(std::size_t tb) const -> double { return th_min + (tb + 1) * th_width; }
};

// Per-species (p, theta) grid: pi+/- get the 6-bin [0.3 .. 6] GeV momentum grid
// and 12 theta bins of 3 deg over [6, 42); electrons keep the 8-bin [2 .. 10]
// GeV grid and 10 theta bins of 2 deg over [6, 26). Matches vz-swim-hist.
[[nodiscard]] auto make_grid(int pid) -> pt_grid {
    if (pid == 211 || pid == -211)
        return pt_grid{6, {{0.3, 1, 2, 3, 4, 5, 6, 0, 0}}, 12, 6.0, 3.0};
    return pt_grid{8, {{2, 3, 4, 5, 6, 7, 8, 9, 10}}, 10, 6.0, 2.0};
}

// Active per-species grids; set once from the fixed species list in main().
std::array<pt_grid, N_SP> GRIDS;

// v_z axis for every histogram, per-cell and integrated. Defaults to [-5, 5]
// cm; the window is configurable via --vz-min/--vz-max (set once from the CLI
// in main(), workers re-parse the same flags).
constexpr int VZ_HIST_BINS = 200;
double VZ_HIST_MIN = -5.0;
double VZ_HIST_MAX = 5.0;

// Format a momentum edge for a histogram name: integer GeV stay 2-digit (e.g.
// "02") so the electron grid's names match vz-swim-hist; fractional edges get
// one decimal ("0.3"). The Python plotters parse both back with float().
auto p_tag(double e) -> std::string {
    const long r = std::lround(e);
    return std::abs(e - static_cast<double>(r)) < 1e-6 ? fmt::format("{:02d}", static_cast<int>(r))
                                                       : fmt::format("{:.1f}", e);
}

// `<sp>_pAA_BB_thCC_DD` name fragment for cell (pb, tb) of species `s`.
auto cell_tag(std::size_t s, const pt_grid& g, std::size_t pb, std::size_t tb) -> std::string {
    const int tlo = static_cast<int>(std::lround(g.th_lo(tb)));
    const int thi = static_cast<int>(std::lround(g.th_hi(tb)));
    return fmt::format("{}_p{}_{}_th{:02d}_{:02d}", SP_TAG[s], p_tag(g.p_edges[pb]),
                       p_tag(g.p_edges[pb + 1]), tlo, thi);
}

// The (species x p x theta) grid of reconstructed-vz histograms. Each worker
// owns one (filled without locking); the workers' grids are merged into the
// main one by summing same-named histograms.
struct hist_set {
    std::array<std::array<std::array<std::unique_ptr<TH1D>, N_TH_MAX>, N_P_MAX>, N_SP> rec;
    // Per-species integrated v_z, summed over the whole (p, theta) selection on
    // the wider [-20, 20] cm window: vz_rec_<sp>_all.
    std::array<std::unique_ptr<TH1D>, N_SP> all;

    hist_set() {
        for (std::size_t s = 0; s < N_SP; ++s) {
            const pt_grid& g = GRIDS[s];
            for (std::size_t pb = 0; pb < g.n_p; ++pb)
                for (std::size_t tb = 0; tb < g.n_th; ++tb) {
                    const std::string tag = cell_tag(s, g, pb, tb);
                    const std::string range =
                        fmt::format("{}, {}<p<{} GeV, {}<#theta<{}#circ", SP_TEX[s], g.p_edges[pb],
                                    g.p_edges[pb + 1], g.th_lo(tb), g.th_hi(tb));
                    rec[s][pb][tb] = std::make_unique<TH1D>(
                        ("vz_rec_" + tag).c_str(), (range + ";v_{z} (rec) [cm];counts").c_str(),
                        VZ_HIST_BINS, VZ_HIST_MIN, VZ_HIST_MAX);
                }
            all[s] = std::make_unique<TH1D>(
                fmt::format("vz_rec_{}_all", SP_TAG[s]).c_str(),
                fmt::format("{}, integrated;v_{{z}} (rec) [cm];counts", SP_TEX[s]).c_str(),
                VZ_HIST_BINS, VZ_HIST_MIN, VZ_HIST_MAX);
        }
    }

    auto merge(const hist_set& other) -> void {
        for (std::size_t s = 0; s < N_SP; ++s) {
            for (std::size_t pb = 0; pb < GRIDS[s].n_p; ++pb)
                for (std::size_t tb = 0; tb < GRIDS[s].n_th; ++tb)
                    rec[s][pb][tb]->Add(other.rec[s][pb][tb].get());
            all[s]->Add(other.all[s].get());
        }
    }
};

struct cli_args {
    std::vector<std::string> inputs;
    std::string output = "output/cpp/rec_vz.root";
    int jobs = 0;        // 0 = one worker process per hardware thread
    bool quiet = false;
    double vz_min = -5.0;  // v_z histogram window [cm] (per-cell and integrated)
    double vz_max = 5.0;
    long long max_events = 0;     // process at most ~N events (0 = all); see --max-events
    bool worker = false;          // hidden: set by the supervisor on each re-exec
    long long record_begin = -1;  // hidden: worker's global HIPO record range [begin, end)
    long long record_end = -1;
    std::string emit_counts;    // hidden: worker's counts sidecar path
    std::string progress_file;  // hidden: shared progress-counter path (supervisor -> worker)
};

auto build_parser() -> argparse::parser {
    argparse::parser p("rec-vz-hist",
                       "Reconstructed v_z in (p, theta) bins for FD e-/pi+/pi-, read straight "
                       "from recon HIPO files (no particles tree, no swim).");
    p.add_argument("inputs")
        .nargs(argparse::nargs::one_or_more)
        .metavar("input")
        .help(".hipo file(s) from one reconstruction pass");
    p.add_argument("-o", "--output")
        .default_value("output/cpp/rec_vz.root")
        .help("output ROOT file");
    p.add_argument("-j", "--jobs", "-t", "--threads")
        .default_value(0)
        .help("worker processes, 0 = one per hardware thread (--threads is an alias)");
    p.add_argument("--vz-min").default_value(-5.0).help("v_z histogram lower edge [cm]");
    p.add_argument("--vz-max").default_value(5.0).help("v_z histogram upper edge [cm]");
    p.add_argument("--max-events")
        .default_value(static_cast<long long>(0))
        .help("process at most ~N events (0 = all; approximate, rounded to whole HIPO records)");
    p.add_argument("-q", "--quiet").flag().help("suppress the progress bar");
    p.add_argument("--worker").flag().hidden();  // re-exec'd worker marker
    p.add_argument("--record-begin").default_value(static_cast<long long>(-1)).hidden();
    p.add_argument("--record-end").default_value(static_cast<long long>(-1)).hidden();
    p.add_argument("--emit-counts").default_value("").hidden();
    p.add_argument("--progress-file").default_value("").hidden();
    return p;
}

auto read_args(const argparse::parser& p) -> cli_args {
    cli_args a;
    a.inputs = p.get<std::vector<std::string>>("inputs");
    a.output = p.get<std::string>("--output");
    a.jobs = p.get<int>("--jobs");
    a.quiet = p.get<bool>("--quiet");
    a.vz_min = p.get<double>("--vz-min");
    a.vz_max = p.get<double>("--vz-max");
    a.max_events = p.get<long long>("--max-events");
    a.worker = p.get<bool>("--worker");
    a.record_begin = p.get<long long>("--record-begin");
    a.record_end = p.get<long long>("--record-end");
    a.emit_counts = p.get<std::string>("--emit-counts");
    a.progress_file = p.get<std::string>("--progress-file");
    return a;
}

// Banks read for the selection. REC::Particle is required; REC::Traj (DC-edge
// fiducial) and REC::Calorimeter (electron PCAL fiducial) are optional -- absent
// means the corresponding cut rejects the row, as in hipo2root.
const std::vector<std::string> WANTED_BANKS = {"REC::Particle", "REC::Traj", "REC::Calorimeter"};

auto present_banks(const std::string& file) -> std::vector<std::string> {
    std::vector<std::string> present;
    auto f = hipo::file::open(file);
    if (!f) {
        std::fprintf(stderr, "warning: cannot open '%s' to read dictionary: %s\n", file.c_str(),
                     f.error().message.c_str());
        return present;
    }
    for (const auto& n : WANTED_BANKS)
        if (f->find_schema(n) != nullptr) present.push_back(n);
    return present;
}

// Print the torus (and solenoid) scale from the first file's RUN::config -- the
// field scale the reconstruction used, so a run is self-documenting. RUN::config
// can be empty in the first events, so scan up to the first 100. Best-effort: it
// warns and returns on any miss rather than aborting the run.
auto print_torus_scale(const std::string& file) -> void {
    auto fr = hipo::file::open(file);
    if (!fr) {
        std::fprintf(stderr, "warning: cannot open '%s' to read RUN::config: %s\n", file.c_str(),
                     fr.error().message.c_str());
        return;
    }
    hipo::file& f = *fr;
    auto cfg = f.bank("RUN::config");
    if (!cfg) {
        std::fprintf(stderr, "warning: no RUN::config in '%s'\n", file.c_str());
        return;
    }
    const std::uint64_t n = std::min<std::uint64_t>(100, f.event_count());
    for (std::uint64_t i = 0; i < n; ++i) {
        auto ev = f.event_at(i);
        if (!ev) continue;
        hipo::bank_view c = ev->get(*cfg);
        if (c.rows() > 0) {
            fmt::print("RUN::config: run {}, torus scale {}, solenoid scale {}\n",
                       c.get<int>("run", 0), c.get<double>("torus", 0),
                       c.get<double>("solenoid", 0));
            return;
        }
    }
    std::fprintf(stderr, "warning: RUN::config empty in the first 100 events of '%s'\n",
                 file.c_str());
}

// Cut-flow counters the histogram file does not hold; the supervisor sums one
// per worker. Plain text, one value per line in declaration order.
struct counts {
    long long n_events = 0;
    std::array<long long, N_SP> n_sel{};     // accepted tracks per species (integrated hist)
    std::array<long long, N_SP> n_filled{};  // ... of those, falling inside a (p,theta) cell

    auto write(const std::string& path) const -> void {
        std::ofstream out(path);
        if (!out) throw std::runtime_error(fmt::format("cannot write counts '{}'", path));
        out << n_events << '\n';
        for (long long v : n_sel) out << v << '\n';
        for (long long v : n_filled) out << v << '\n';
    }
    auto add_from_file(const std::string& path) -> void {
        std::ifstream in(path);
        if (!in) throw std::runtime_error(fmt::format("cannot read counts '{}'", path));
        long long ne = 0;
        in >> ne;
        n_events += ne;
        for (auto* arr : {&n_sel, &n_filled})
            for (long long& v : *arr) {
                long long x = 0;
                in >> x;
                v += x;
            }
    }
};

// Per-event callback: select FD e-/pi+/pi- passing the quality cuts and fill the
// reconstructed-vz histogram of the matching (species, p, theta) cell. A named
// functor (not a lambda) so the hist_set / counters it accumulates into are
// stable members. Selection mirrors hipo2root's default keep logic.
struct rec_vz_filler {
    long rec_particle = -1;
    long rec_traj = -1;
    long rec_calo = -1;

    hist_set grid;
    counts c;

    auto operator()(vz::bank_list& bl, int /*file_idx*/, long /*event_idx*/) -> void {
        if (rec_particle < 0) return;
        hipo::bank_view rec = bl[rec_particle];
        hipo::bank_view traj = rec_traj >= 0 ? bl[rec_traj] : hipo::bank_view{};
        hipo::bank_view calo = rec_calo >= 0 ? bl[rec_calo] : hipo::bank_view{};
        const int n = static_cast<int>(rec.rows());
        for (int i = 0; i < n; ++i) {
            const int sp = species_of(rec.get<int>("pid", i));
            if (sp < 0) continue;
            if (!vz::is_forward(rec.get<int>("status", i))) continue;

            // Quality selection (same thresholds as hipo2root's default mode).
            const double chi2pid = rec.get<double>("chi2pid", i);
            if (!(chi2pid > -3.0 && chi2pid < 3.0)) continue;
            const auto edge1 = vz::layer_f64(traj, i, vz::DET_DC, vz::DC_LAYERS[0], "edge");
            const auto edge2 = vz::layer_f64(traj, i, vz::DET_DC, vz::DC_LAYERS[1], "edge");
            const auto edge3 = vz::layer_f64(traj, i, vz::DET_DC, vz::DC_LAYERS[2], "edge");
            if (!edge1 || !edge2 || !edge3) continue;
            if (!(*edge1 > 5.0 && *edge2 > 5.0 && *edge3 > 10.0)) continue;
            if (SP_PID[static_cast<std::size_t>(sp)] == 11) {
                const auto lv = vz::layer_f64(calo, i, vz::DET_ECAL, vz::ECAL_PCAL_LAYER, "lv");
                const auto lw = vz::layer_f64(calo, i, vz::DET_ECAL, vz::ECAL_PCAL_LAYER, "lw");
                if (!lv || !lw) continue;
                if (!(*lv > 9.0 && *lw > 9.0)) continue;
            }

            const double px = rec.get<double>("px", i);
            const double py = rec.get<double>("py", i);
            const double pz = rec.get<double>("pz", i);
            const double p = std::sqrt(px * px + py * py + pz * pz);
            if (p <= 0.0) continue;
            const double vz_val = rec.get<double>("vz", i);

            // Integrated v_z: every accepted track of this species, regardless of
            // its (p, theta) cell, on the wide [-20, 20] cm window.
            const auto s = static_cast<std::size_t>(sp);
            grid.all[s]->Fill(vz_val);
            ++c.n_sel[s];

            // Per-cell v_z: only tracks falling inside the species (p, theta) grid.
            const double theta = std::acos(std::clamp(pz / p, -1.0, 1.0)) * vz::RAD2DEG;
            const pt_grid& g = GRIDS[s];
            const long pbin = g.p_bin(p);
            const long tbin = g.th_bin(theta);
            if (pbin < 0 || tbin < 0) continue;

            grid.rec[s][static_cast<std::size_t>(pbin)][static_cast<std::size_t>(tbin)]->Fill(vz_val);
            ++c.n_filled[s];
        }
    }
};

auto write_grid(const hist_set& grid, const std::string& path) -> void {
    TFile fout(path.c_str(), "RECREATE", "", vz::merge_compression());
    if (fout.IsZombie()) throw std::runtime_error(fmt::format("cannot create '{}'", path));
    for (std::size_t s = 0; s < N_SP; ++s) {
        for (std::size_t pb = 0; pb < GRIDS[s].n_p; ++pb)
            for (std::size_t tb = 0; tb < GRIDS[s].n_th; ++tb)
                grid.rec[s][pb][tb]->Write();
        grid.all[s]->Write();
    }
    fout.Close();
}

auto print_summary(const counts& c, const std::string& output) -> void {
    fmt::print("----------------------------------------\n");
    fmt::print("events read         : {}\n", c.n_events);
    long long n_cells = 0;
    for (std::size_t s = 0; s < N_SP; ++s) {
        fmt::print("{:>3}: {} accepted (integrated), {} in (p,theta) cells\n", SP_TAG[s],
                   c.n_sel[s], c.n_filled[s]);
        n_cells += static_cast<long long>(GRIDS[s].n_p * GRIDS[s].n_th);
    }
    fmt::print("histograms written to {}: {} cells (vz_rec_<sp>_<cell>) + {} integrated "
               "(vz_rec_<sp>_all)\n",
               output, n_cells, N_SP);
}

// One single-threaded pass over the global record range [rec_begin, rec_end):
// fill the (species, p, theta) grid and write it to args.output. In worker mode
// the chatter is suppressed and a counts sidecar is written; a lone run prints
// the inputs and the summary itself.
auto run_dump(const cli_args& args, const std::vector<std::string>& paths, bool worker_mode)
    -> int {
    vz::chain ch(/*progress=*/!args.quiet && !worker_mode, /*verbose=*/false);
    for (const auto& p : paths) ch.add(p);
    try {
        ch.open(/*validate_all=*/true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not open inputs: %s\n", e.what());
        return 1;
    }

    std::vector<std::string> present = present_banks(paths[0]);
    if (std::find(present.begin(), present.end(), "REC::Particle") == present.end()) {
        std::fprintf(stderr, "error: REC::Particle not found in '%s'\n", paths[0].c_str());
        return 1;
    }
    if (!worker_mode) fmt::print("inputs: {} file(s), {} events\n", ch.size(), ch.total_events());

    vz::bank_list banks;
    try {
        banks = ch.get_banks(present);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not build banklist: %s\n", e.what());
        return 1;
    }
    auto bank_index = [&](const char* n) -> long {
        try {
            return vz::get_banklist_index(banks, n);
        } catch (const std::exception&) {
            return -1;
        }
    };

    rec_vz_filler filler;
    filler.rec_particle = bank_index("REC::Particle");
    filler.rec_traj = bank_index("REC::Traj");
    filler.rec_calo = bank_index("REC::Calorimeter");

    const std::size_t rec_begin =
        args.record_begin >= 0 ? static_cast<std::size_t>(args.record_begin) : 0;
    std::size_t rec_end = args.record_end >= 0 ? static_cast<std::size_t>(args.record_end)
                                               : std::numeric_limits<std::size_t>::max();
    // --max-events on a lone (un-sharded) run: cap the record range to roughly N
    // events. Workers get an explicit range from the supervisor, so they skip
    // this (guarded on record_end < 0) and never double-apply the cap.
    if (!worker_mode && args.max_events > 0 && args.record_end < 0) {
        const std::uint64_t te = ch.total_events();
        const std::uint64_t tr = ch.total_records();
        if (tr > 0 && static_cast<std::uint64_t>(args.max_events) < te) {
            const double per_rec = static_cast<double>(te) / static_cast<double>(tr);
            const auto cap =
                static_cast<std::size_t>(std::ceil(static_cast<double>(args.max_events) / per_rec));
            rec_end = std::min(rec_end, cap);
        }
    }
    vz::shared_counter prog =
        args.progress_file.empty() ? vz::shared_counter{}
                                   : vz::shared_counter::map(args.progress_file, /*create=*/false);
    std::uint64_t prog_local = 0;
    ch.process(
        banks,
        [&](vz::bank_list& bl, int fi, long ei) {
            filler(bl, fi, ei);
            ++filler.c.n_events;
            if ((++prog_local & 0x3FFFU) == 0) prog.add(0x4000);
        },
        rec_begin, rec_end);
    prog.add(prog_local & 0x3FFFU);

    try {
        write_grid(filler.grid, args.output);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    if (worker_mode) {
        try {
            filler.c.write(args.emit_counts);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error: could not write counts: %s\n", e.what());
            return 1;
        }
    } else {
        print_summary(filler.c, args.output);
    }
    return 0;
}

// Summed event/record counts from the input scan; ok=false (after the first
// open failure is printed) aborts the run.
struct scan_result {
    std::uint64_t total_events = 0;
    std::uint64_t total_records = 0;
    bool ok = true;
};

// Open + index every input to total its events/records, in parallel across
// `n_threads` threads. The opens are independent and I/O-latency bound (a big
// win over sshfs, where a serial scan of a few hundred files dominates the
// run), so each thread pulls the next file off a shared atomic cursor. A
// thread-safe progress bar ticks per file; the first open failure stops the
// scan and is reported. The workers re-open their own shards afterwards.
auto scan_inputs(const std::vector<std::string>& paths, int n_threads, bool quiet) -> scan_result {
    scan_result result;
    const std::size_t n = paths.size();
    if (n == 0) return result;

    std::unique_ptr<progress_tracker> bar;
    if (!quiet && n > 1) {
        progress_tracker::config cfg;
        cfg.label = "Scanning";
        cfg.show_rate = true;
        bar = std::make_unique<progress_tracker>(n, cfg);
        bar->start();
    }

    const unsigned nt = static_cast<unsigned>(
        std::clamp<long>(n_threads, 1, static_cast<long>(n)));
    std::atomic<std::size_t> next{0};
    std::atomic<std::uint64_t> ev{0}, rec{0};
    std::atomic<bool> failed{false};
    std::mutex err_mtx;
    std::string err_path, err_msg;

    auto worker = [&] {
        for (std::size_t i = next.fetch_add(1, std::memory_order_relaxed); i < n;
             i = next.fetch_add(1, std::memory_order_relaxed)) {
            if (failed.load(std::memory_order_relaxed)) return;
            hipo::read_options opt;
            opt.prefetch_records = 0;  // index only; no background decode pipeline
            auto f = hipo::file::open(paths[i], opt);
            if (!f) {
                bool expected = false;
                if (failed.compare_exchange_strong(expected, true)) {
                    std::lock_guard<std::mutex> lk(err_mtx);
                    err_path = paths[i];
                    err_msg = f.error().message;
                }
                return;
            }
            ev.fetch_add(f->event_count(), std::memory_order_relaxed);
            rec.fetch_add(f->record_count(), std::memory_order_relaxed);
            if (bar) bar->increment();
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(nt - 1);
    for (unsigned t = 1; t < nt; ++t) pool.emplace_back(worker);
    worker();  // the calling thread is one of the `nt` scanners
    for (auto& th : pool) th.join();
    if (bar) bar->finish();

    if (failed.load()) {
        std::fprintf(stderr, "error: cannot open '%s': %s\n", err_path.c_str(), err_msg.c_str());
        result.ok = false;
        return result;
    }
    result.total_events = ev.load();
    result.total_records = rec.load();
    return result;
}

// Supervisor: split the work across `jobs` re-exec'd workers and merge their
// partial histogram files into args.output. With at least `jobs` input files
// each worker takes a subset of files; with fewer the work is sharded by HIPO
// record range so one big file still parallelizes. Mirrors hipo2root.
auto run_supervisor(const cli_args& args, char** argv, const std::vector<std::string>& paths,
                    int jobs) -> int {
    const bool by_file = paths.size() >= static_cast<std::size_t>(jobs);

    // Scan (open + index) every input to total its events/records for the
    // worker sharding and the processing bar -- multithreaded, since over sshfs
    // this is the slow part of the run.
    if (!args.quiet) {
        fmt::print("rec-vz-hist: scanning {} input file(s) with {} thread(s)...\n", paths.size(),
                   std::min<std::size_t>(static_cast<std::size_t>(std::max(jobs, 1)), paths.size()));
        std::fflush(stdout);
    }
    const scan_result scan = scan_inputs(paths, jobs, args.quiet);
    if (!scan.ok) return 1;
    const std::uint64_t total_events = scan.total_events;
    const std::uint64_t total_records = scan.total_records;

    // --max-events: cap the processed record range to roughly N events (rounded
    // up to whole HIPO records, since records are the shard unit). Only sensible
    // when sharding by record (a single / few large files); when sharding by
    // file there is no per-file event cap, so warn and process everything.
    std::uint64_t shard_records = total_records;  // records actually handed to workers
    std::uint64_t bar_events = total_events;      // events the progress bar counts up to
    if (args.max_events > 0 && static_cast<std::uint64_t>(args.max_events) < total_events) {
        if (by_file) {
            fmt::print("rec-vz-hist: note: --max-events applies only when sharding by record "
                       "(single / few large files); ignored for {} files\n",
                       paths.size());
        } else if (total_records > 0) {
            const double per_rec =
                static_cast<double>(total_events) / static_cast<double>(total_records);
            shard_records = std::min<std::uint64_t>(
                total_records,
                static_cast<std::uint64_t>(std::ceil(static_cast<double>(args.max_events) / per_rec)));
            bar_events = std::min<std::uint64_t>(
                total_events,
                static_cast<std::uint64_t>(std::llround(static_cast<double>(shard_records) * per_rec)));
            fmt::print("rec-vz-hist: --max-events {} -> ~{} events ({} of {} records)\n",
                       args.max_events, bar_events, shard_records, total_records);
        }
    }

    int n_workers = jobs;
    if (!by_file) {
        n_workers = static_cast<int>(std::min<std::uint64_t>(
            static_cast<std::uint64_t>(jobs), std::max<std::uint64_t>(1, shard_records)));
        if (n_workers <= 1) return run_dump(args, paths, /*worker_mode=*/false);
    }

    const std::string exe = vz::self_exe_path(argv[0]);
    std::string dir;
    try {
        dir = vz::make_temp_dir("rec-vz-hist");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    const std::string progress_path = fmt::format("{}/progress", dir);
    vz::shared_counter progress = vz::shared_counter::map(progress_path, /*create=*/true);

    std::vector<std::string> parts, count_files;
    std::vector<std::vector<std::string>> argvs;
    const std::uint64_t chunk = by_file ? 0 : (shard_records + n_workers - 1) / n_workers;
    for (int i = 0; i < n_workers; ++i) {
        const std::string part = fmt::format("{}/part-{:03}.root", dir, i);
        const std::string cnt = fmt::format("{}/part-{:03}.counts", dir, i);
        parts.push_back(part);
        count_files.push_back(cnt);
        std::vector<std::string> av = {exe,
                                       "--jobs",
                                       "1",
                                       "--worker",
                                       "--quiet",
                                       "--output",
                                       part,
                                       "--emit-counts",
                                       cnt,
                                       "--progress-file",
                                       progress_path,
                                       "--vz-min",
                                       fmt::format("{}", args.vz_min),
                                       "--vz-max",
                                       fmt::format("{}", args.vz_max)};
        if (by_file) {
            for (std::size_t j = i; j < paths.size(); j += static_cast<std::size_t>(n_workers))
                av.push_back(paths[j]);
        } else {
            const std::uint64_t rb = static_cast<std::uint64_t>(i) * chunk;
            const std::uint64_t re = std::min<std::uint64_t>(rb + chunk, shard_records);
            av.insert(av.end(), {"--record-begin", fmt::format("{}", rb), "--record-end",
                                 fmt::format("{}", re)});
            for (const auto& p : paths) av.push_back(p);
        }
        argvs.push_back(std::move(av));
    }

    fmt::print("rec-vz-hist: {} worker process(es) sharded by {} over {} -> {}\n", n_workers,
               by_file ? "file" : "record",
               by_file ? fmt::format("{} file(s)", paths.size())
                       : fmt::format("{} records in {} file(s)", shard_records, paths.size()),
               args.output);

    const auto pids = vz::spawn_workers(argvs);
    bool ok = !pids.empty();
    if (ok) {
        std::unique_ptr<progress_tracker> bar;
        if (!args.quiet && bar_events > 0) {
            progress_tracker::config cfg;
            cfg.label = "Processing";
            cfg.show_eta = true;
            cfg.show_rate = true;
            bar = std::make_unique<progress_tracker>(static_cast<std::size_t>(bar_events), cfg);
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
    if (!args.quiet) {
        fmt::print("rec-vz-hist: merging {} partial file(s) -> {} ...\n", parts.size(),
                   args.output);
        std::fflush(stdout);
    }
    try {
        vz::merge_root_files(parts, args.output);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: merge failed: %s\n", e.what());
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        return 1;
    }

    counts total;
    try {
        for (const auto& c : count_files) total.add_from_file(c);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "warning: could not read worker counts: %s\n", e.what());
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    print_summary(total, args.output);
    return 0;
}

}  // namespace

auto main(int argc, char** argv) -> int {
    // Keep histograms out of the (non-thread-safe) global directory; workers are
    // single-threaded.
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
    cli_args args = read_args(p);
    for (std::size_t s = 0; s < N_SP; ++s) GRIDS[s] = make_grid(SP_PID[s]);
    VZ_HIST_MIN = args.vz_min;  // v_z window (workers re-parse the same flags)
    VZ_HIST_MAX = args.vz_max;

    if (args.inputs.empty()) {
        std::fprintf(stderr, "error: no input files\n");
        return 1;
    }

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int want = args.jobs > 0 ? args.jobs : static_cast<int>(hw);
    vz::ensure_parent_dir(args.output);

    if (args.worker) return run_dump(args, args.inputs, /*worker_mode=*/true);
    print_torus_scale(args.inputs.front());  // top-level only; report the field scale used
    if (want <= 1) return run_dump(args, args.inputs, /*worker_mode=*/false);
    return run_supervisor(args, argv, args.inputs, want);
}
