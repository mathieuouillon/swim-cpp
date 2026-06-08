// CLAS12 v_z / PID diagnostic analysis (C++ port of the Rust analysis).
//
// Reads CLAS12 HIPO files with hipo4, matches reconstructed tracks to MC truth
// (MC::RecMatch) for charged pions/muons, swims them back through the CLAS12
// field, and writes v_z / PID diagnostic histograms to a ROOT file.
//
// Usage:
//   swim-analysis <input>... [--output FILE] [--threads N] [--quiet]
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <glob.h>

#include <fmt/format.h>

#include "TH1.h"
#include "TROOT.h"
#include "accumulator_registry.hpp"
#include "analysis.hpp"
#include "constants.hpp"
#include "field.hpp"
#include "hipo4/chain.h"
#include "hipo4/dictionary.h"
#include "hipo4/reader.h"

namespace {

struct Args {
    std::vector<std::string> inputs;
    std::string output = "vz.root";
    int threads = 0;
    bool quiet = false;
    std::string torus = "Full_torus_r501_phi361_z501_31Mar2021.dat";
    std::string solenoid = "Symm_solenoid_r601_phi1_z1201_21May2019.dat";
    double torus_scale = -1.0;     // inbending (sign to verify)
    double solenoid_scale = -1.0;
    double solenoid_z_shift = -3.0;  // move the solenoid map by -3 cm in z
};

enum class ParseResult { Ok, Help, Error };

void usage() {
    std::fprintf(stderr,
                 "usage: swim-analysis <input>... [--output FILE] [--threads N] [--quiet]\n");
}

ParseResult parse_args(int argc, char** argv, Args& a, std::string& err) {
    auto next = [&](int& i) -> std::optional<std::string> {
        if (i + 1 >= argc) return std::nullopt;
        return std::string(argv[++i]);
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" || arg == "-o") {
            auto v = next(i);
            if (!v) { err = "--output needs a value"; return ParseResult::Error; }
            a.output = *v;
        } else if (arg == "--threads" || arg == "-t" || arg == "-j") {
            auto v = next(i);
            if (!v) { err = "--threads needs a value"; return ParseResult::Error; }
            try { a.threads = std::stoi(*v); } catch (...) { err = "--threads must be an integer"; return ParseResult::Error; }
        } else if (arg == "--torus") {
            auto v = next(i);
            if (!v) { err = "--torus needs a path"; return ParseResult::Error; }
            a.torus = *v;
        } else if (arg == "--solenoid") {
            auto v = next(i);
            if (!v) { err = "--solenoid needs a path"; return ParseResult::Error; }
            a.solenoid = *v;
        } else if (arg == "--torus-scale") {
            auto v = next(i);
            if (!v) { err = "--torus-scale needs a value"; return ParseResult::Error; }
            try { a.torus_scale = std::stod(*v); } catch (...) { err = "--torus-scale must be a number"; return ParseResult::Error; }
        } else if (arg == "--solenoid-scale") {
            auto v = next(i);
            if (!v) { err = "--solenoid-scale needs a value"; return ParseResult::Error; }
            try { a.solenoid_scale = std::stod(*v); } catch (...) { err = "--solenoid-scale must be a number"; return ParseResult::Error; }
        } else if (arg == "--solenoid-z-shift") {
            auto v = next(i);
            if (!v) { err = "--solenoid-z-shift needs a value"; return ParseResult::Error; }
            try { a.solenoid_z_shift = std::stod(*v); } catch (...) { err = "--solenoid-z-shift must be a number"; return ParseResult::Error; }
        } else if (arg == "--quiet" || arg == "-q") {
            a.quiet = true;
        } else if (arg == "-h" || arg == "--help") {
            return ParseResult::Help;
        } else if (arg.size() > 1 && arg[0] == '-' && arg != "-") {
            err = "unknown flag: " + arg;
            return ParseResult::Error;
        } else {
            a.inputs.push_back(arg);
        }
    }
    if (a.inputs.empty()) {
        err = "missing <input> (.hipo files, globs, directories, or @list)";
        return ParseResult::Error;
    }
    return ParseResult::Ok;
}

bool looks_like_glob(const std::string& s) {
    return s.find_first_of("*?[") != std::string::npos;
}

std::vector<std::string> expand_glob(const std::string& pattern) {
    std::vector<std::string> out;
    glob_t g{};
    int rc = glob(pattern.c_str(), 0, nullptr, &g);
    if (rc == 0) {
        for (std::size_t i = 0; i < g.gl_pathc; ++i) out.emplace_back(g.gl_pathv[i]);
    }
    globfree(&g);
    return out;
}

std::string trim(const std::string& s) {
    auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Expand the CLI inputs into a flat list of file paths: `@list` files are read
// line by line, globs are expanded, directories contribute their `*.hipo`
// files, and anything else is taken verbatim.
std::vector<std::string> collect_inputs(const std::vector<std::string>& inputs) {
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
// data). Absent banks map to a -1 index in BankIndex -> nullptr in fill_event.
std::vector<std::string> present_banks(const std::string& first_file) {
    std::vector<std::string> present;
    hipo::reader r;
    r.open(first_file.c_str());
    hipo::dictionary dict;
    r.readDictionary(dict);
    for (const auto& n : WANTED_BANKS) {
        if (dict.hasSchema(n.c_str())) present.push_back(n);
    }
    return present;
}

}  // namespace

int main(int argc, char** argv) {
    // ROOT global state: enable internal locks and stop histograms registering
    // themselves in the (non-thread-safe) global directory. MUST be done before
    // any threads or histograms are created.
    ROOT::EnableThreadSafety();
    TH1::AddDirectory(kFALSE);

    Args args;
    std::string err;
    switch (parse_args(argc, argv, args, err)) {
        case ParseResult::Help:
            usage();
            return 0;
        case ParseResult::Error:
            std::fprintf(stderr, "error: %s\n\n", err.c_str());
            usage();
            return 2;
        case ParseResult::Ok:
            break;
    }

    std::vector<std::string> paths = collect_inputs(args.inputs);
    if (paths.empty()) {
        std::fprintf(stderr, "error: no input files found\n");
        return 1;
    }

    hipo::chain ch(args.threads, /*progress=*/!args.quiet, /*verbose=*/false);
    for (const auto& p : paths) ch.add(p);
    try {
        ch.open(/*validate_all=*/true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not open inputs: %s\n", e.what());
        return 1;
    }

    fmt::print("inputs: {} file(s), {} events\n", ch.size(), ch.total_events());

    // Load the magnetic field maps once (~1 GB torus) and share read-only across
    // the worker threads.
    fmt::print("loading field maps: torus={} (scale {}), solenoid={} (scale {}, z-shift {} cm)\n",
               args.torus, args.torus_scale, args.solenoid, args.solenoid_scale,
               args.solenoid_z_shift);
    std::unique_ptr<vz::CompositeField> field_ptr;
    try {
        field_ptr = std::make_unique<vz::CompositeField>(vz::CompositeField::load(
            args.torus, args.torus_scale, args.solenoid, args.solenoid_scale,
            args.solenoid_z_shift));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not load field maps: %s\n", e.what());
        return 1;
    }
    const vz::Field& field = *field_ptr;

    // Build the banklist from banks present in the first file, then resolve the
    // bank indices (absent -> -1).
    hipo::banklist banks;
    try {
        banks = ch.getBanks(present_banks(paths[0]));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not build banklist: %s\n", e.what());
        return 1;
    }
    vz::BankIndex idx(banks);

    // Parallel event loop: each worker folds into its own Analysis (registry),
    // merged on the main thread afterwards.
    vz::AccumulatorRegistry registry;
    ch.process(
        banks,
        [&](hipo::banklist& bl, int /*file_idx*/, long /*event_idx*/) {
            registry.local().fill_event(bl, idx, field);
        },
        100.0);

    vz::Analysis result = registry.merge_all();
    try {
        result.write(args.output);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not write '%s': %s\n", args.output.c_str(), e.what());
        return 1;
    }

    // Summary, mirroring the Rust report.
    fmt::print("----------------------------------------\n");
    fmt::print("files            : {}\n", ch.size());
    fmt::print("events           : {}\n", result.events());
    fmt::print("fills [truth/reco]:\n");
    for (std::size_t t = 0; t < vz::N_SPECIES; ++t) {
        for (std::size_t r = 0; r < vz::N_SPECIES; ++r) {
            fmt::print("  {} / {} : {}\n", vz::SPECIES_NAME[t], vz::SPECIES_NAME[r],
                       result.fills()[t][r]);
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
    fmt::print("histograms written to {} (Forward Detector only):\n", args.output);
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
    return 0;
}
