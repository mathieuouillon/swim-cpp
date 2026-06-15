// Dump REC::Particle kinematics from CLAS12 HIPO files into a flat ROOT TTree.
//
// One TTree entry per reconstructed particle, holding the PID, momentum
// (px, py, pz), the z vertex (vz), the status word, the DC Region-1 and
// Region-3 trajectory states from REC::Traj (position + direction cosines,
// NaN when absent), the per-event RASTER::position beam offset, plus the
// event index it came from. The RUN::config torus/solenoid scales are printed
// at startup (feed them to vz-swim-hist).
//
// Reads with the hipo4 chain. Bank decoding runs on N worker threads (each
// with its own banklist copy); rows are staged per event and appended to the
// (single) TTree under a mutex, so TTree::Fill stays serial.
//
// Usage:
//   hipo2root <input>... [--output FILE] [--threads N] [--quiet]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "TFile.h"
#include "TROOT.h"
#include "TTree.h"

#include "bank_access.hpp"
#include "constants.hpp"
#include "hipo4/hipo.hpp"
#include "hipo_chain.hpp"

namespace {

struct Args {
    std::vector<std::string> inputs;
    std::string output = "particles.root";
    int threads = 0;          // 0 = auto-detect hardware concurrency
    bool run_config = false;  // only print RUN::config (per input file) and exit
    bool quiet = false;
};

enum class ParseResult { Ok, Help, Error };

void usage() {
    std::fprintf(stderr,
                 "usage: hipo2root <input>... [--output FILE] [--threads N] [--run-config] "
                 "[--quiet]\n"
                 "  --run-config  only print each input's RUN::config (run, torus/solenoid "
                 "scales) and exit\n");
}

ParseResult parse_args(int argc, char** argv, Args& a, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" || arg == "-o") {
            if (i + 1 >= argc) { err = "--output needs a value"; return ParseResult::Error; }
            a.output = argv[++i];
        } else if (arg == "--threads" || arg == "-t" || arg == "-j") {
            if (i + 1 >= argc) { err = "--threads needs a value"; return ParseResult::Error; }
            try { a.threads = std::stoi(argv[++i]); } catch (...) { err = "--threads must be an integer"; return ParseResult::Error; }
        } else if (arg == "--run-config") {
            a.run_config = true;
        } else if (arg == "--quiet" || arg == "-q") {
            a.quiet = true;
        } else if (arg == "-h" || arg == "--help") {
            return ParseResult::Help;
        } else if (arg.size() > 1 && arg[0] == '-') {
            err = "unknown flag: " + arg;
            return ParseResult::Error;
        } else {
            a.inputs.push_back(arg);
        }
    }
    if (a.inputs.empty()) {
        err = "missing <input> (.hipo file(s))";
        return ParseResult::Error;
    }
    return ParseResult::Ok;
}

// Banks the dump reads. REC::Particle is required; REC::Traj and
// RASTER::position are optional (their branches are NaN when absent).
const std::vector<std::string> WANTED_BANKS = {"REC::Particle", "REC::Traj",
                                               "RASTER::position"};

// Keep only the wanted banks present in the first file's dictionary, so the
// banklist construction never throws on a missing schema.
std::vector<std::string> present_banks(const std::string& file) {
    std::vector<std::string> present;
    auto f = hipo::file::open(file);
    if (!f) {
        std::fprintf(stderr, "warning: cannot open '%s' to read dictionary: %s\n", file.c_str(),
                     f.error().message.c_str());
        return present;
    }
    for (const auto& n : WANTED_BANKS) {
        if (f->find_schema(n) != nullptr) present.push_back(n);
    }
    return present;
}

// Print the RUN::config torus/solenoid scale factors from the first file (what
// the reconstruction's swimmer used — pass them to vz-swim-hist as
// --torus-scale/--solenoid-scale).
void print_run_config(const std::string& file) {
    auto fr = hipo::file::open(file);
    if (!fr) {
        std::fprintf(stderr, "warning: cannot open '%s': %s\n", file.c_str(),
                     fr.error().message.c_str());
        return;
    }
    hipo::file& f = *fr;
    auto cfg = f.bank("RUN::config");
    if (!cfg) {
        std::fprintf(stderr, "warning: no RUN::config in '%s'\n", file.c_str());
        return;
    }
    // The bank can be absent in the first events; scan up to the first 100.
    const std::uint64_t n = std::min<std::uint64_t>(100, f.event_count());
    for (std::uint64_t i = 0; i < n; ++i) {
        auto ev = f.event_at(i);
        if (!ev) continue;
        hipo::bank_view c = ev->get(*cfg);
        if (c.rows() > 0) {
            const double torus = c.get<double>("torus", 0);
            const double solenoid = c.get<double>("solenoid", 0);
            fmt::print("RUN::config: run {}, torus scale {}, solenoid scale {}\n",
                       c.get<int>("run", 0), torus, solenoid);
            // Our FieldMap orientation is opposite to cnuphys's for both
            // magnets (see docs/coatjava_vz.md), so the vz-swim-hist flags are:
            fmt::print("  -> vz-swim-hist: --torus-scale {} --solenoid-scale {}\n", -torus,
                       -solenoid);
            return;
        }
    }
    std::fprintf(stderr, "warning: RUN::config empty in the first 100 events of '%s'\n",
                 file.c_str());
}

// Per-event callback, shared by all chain worker threads: decode every
// REC::Particle row into a thread-local staging vector (parallel), then copy
// the rows into the branch buffers and Fill the tree under a mutex (serial —
// TTree::Fill is not thread-safe). A named functor (not a lambda) so the
// branch buffers it owns are addressable by TTree::Branch.
struct ParticleDumper {
    long rec_particle = -1;  // banklist index of REC::Particle
    long rec_traj = -1;      // banklist index of REC::Traj (-1 = absent)
    long raster_pos = -1;    // banklist index of RASTER::position (-1 = absent)
    TTree* tree = nullptr;

    // Branch buffers (one row per particle).
    Long64_t event = 0;
    Int_t pid = 0;
    Float_t px = 0.f, py = 0.f, pz = 0.f, vz = 0.f;
    Int_t status = 0;
    // DC Region-1 / Region-3 states from REC::Traj (detector 6, layers 6/36):
    // position [cm] + direction cosines; NaN when the row is absent.
    Float_t dc_x = 0.f, dc_y = 0.f, dc_z = 0.f;
    Float_t dc_cx = 0.f, dc_cy = 0.f, dc_cz = 0.f;
    Float_t dc3_x = 0.f, dc3_y = 0.f, dc3_z = 0.f;
    Float_t dc3_cx = 0.f, dc3_cy = 0.f, dc3_cz = 0.f;
    // Per-event raster beam position [cm] from RASTER::position; NaN when
    // absent. The reconstruction adds this to the CCDB beam offset.
    Float_t raster_x = 0.f, raster_y = 0.f;

    long long n_particles = 0;  // guarded by fill_mutex
    std::mutex fill_mutex;

    // One decoded REC::Particle row, staged before the locked Fill.
    struct Row {
        Int_t pid;
        Float_t px, py, pz, vz;
        Int_t status;
        Float_t dc_x, dc_y, dc_z, dc_cx, dc_cy, dc_cz;
        Float_t dc3_x, dc3_y, dc3_z, dc3_cx, dc3_cy, dc3_cz;
    };

    void operator()(vz::BankList& bl, int /*file_idx*/, long event_idx) {
        if (rec_particle < 0) return;
        hipo::bank_view rec = bl[rec_particle];
        hipo::bank_view traj = rec_traj >= 0 ? bl[rec_traj] : hipo::bank_view{};
        hipo::bank_view raster = raster_pos >= 0 ? bl[raster_pos] : hipo::bank_view{};
        const int n = static_cast<int>(rec.rows());
        constexpr Float_t fnan = std::numeric_limits<Float_t>::quiet_NaN();

        Float_t ras_x = fnan, ras_y = fnan;
        if (raster.rows() > 0) {
            ras_x = static_cast<Float_t>(raster.get<double>("x", 0));
            ras_y = static_cast<Float_t>(raster.get<double>("y", 0));
        }

        // Decode in parallel into a per-thread staging buffer (reused across
        // events to avoid a per-event allocation).
        thread_local std::vector<Row> rows;
        rows.clear();
        rows.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            Row r;
            r.pid = rec.get<int>("pid", i);
            r.px = static_cast<Float_t>(rec.get<double>("px", i));
            r.py = static_cast<Float_t>(rec.get<double>("py", i));
            r.pz = static_cast<Float_t>(rec.get<double>("pz", i));
            r.vz = static_cast<Float_t>(rec.get<double>("vz", i));
            r.status = rec.get<int>("status", i);
            r.dc_x = r.dc_y = r.dc_z = r.dc_cx = r.dc_cy = r.dc_cz = fnan;
            r.dc3_x = r.dc3_y = r.dc3_z = r.dc3_cx = r.dc3_cy = r.dc3_cz = fnan;
            if (auto k = vz::match_row(traj, i, vz::DET_DC, vz::DC_LAYERS[0])) {
                r.dc_x = static_cast<Float_t>(traj.get<double>("x", *k));
                r.dc_y = static_cast<Float_t>(traj.get<double>("y", *k));
                r.dc_z = static_cast<Float_t>(traj.get<double>("z", *k));
                r.dc_cx = static_cast<Float_t>(traj.get<double>("cx", *k));
                r.dc_cy = static_cast<Float_t>(traj.get<double>("cy", *k));
                r.dc_cz = static_cast<Float_t>(traj.get<double>("cz", *k));
            }
            if (auto k = vz::match_row(traj, i, vz::DET_DC, vz::DC_LAYERS[2])) {
                r.dc3_x = static_cast<Float_t>(traj.get<double>("x", *k));
                r.dc3_y = static_cast<Float_t>(traj.get<double>("y", *k));
                r.dc3_z = static_cast<Float_t>(traj.get<double>("z", *k));
                r.dc3_cx = static_cast<Float_t>(traj.get<double>("cx", *k));
                r.dc3_cy = static_cast<Float_t>(traj.get<double>("cy", *k));
                r.dc3_cz = static_cast<Float_t>(traj.get<double>("cz", *k));
            }
            rows.push_back(r);
        }
        if (rows.empty()) return;

        // Serial section: copy into the branch buffers and Fill.
        std::lock_guard lock(fill_mutex);
        event = static_cast<Long64_t>(event_idx);
        raster_x = ras_x;
        raster_y = ras_y;
        for (const Row& r : rows) {
            pid = r.pid;
            px = r.px;
            py = r.py;
            pz = r.pz;
            vz = r.vz;
            status = r.status;
            dc_x = r.dc_x;
            dc_y = r.dc_y;
            dc_z = r.dc_z;
            dc_cx = r.dc_cx;
            dc_cy = r.dc_cy;
            dc_cz = r.dc_cz;
            dc3_x = r.dc3_x;
            dc3_y = r.dc3_y;
            dc3_z = r.dc3_z;
            dc3_cx = r.dc3_cx;
            dc3_cy = r.dc3_cy;
            dc3_cz = r.dc3_cz;
            tree->Fill();
            ++n_particles;
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    // ROOT global state: enable internal locks before any worker threads exist
    // (TTree::Fill itself is additionally serialized by ParticleDumper's mutex).
    ROOT::EnableThreadSafety();

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

    // --run-config: just report each input's RUN::config (run number, torus and
    // solenoid scale factors) without producing a tree.
    if (args.run_config) {
        for (const auto& p : args.inputs) {
            if (!std::filesystem::exists(p)) {
                std::fprintf(stderr, "error: no such file: %s\n", p.c_str());
                continue;
            }
            fmt::print("{}:\n  ", p);
            print_run_config(p);
        }
        return 0;
    }

    vz::Chain ch(args.threads, /*progress=*/!args.quiet, /*verbose=*/false);
    for (const auto& p : args.inputs) ch.add(p);
    try {
        ch.open(/*validate_all=*/true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not open inputs: %s\n", e.what());
        return 1;
    }

    std::vector<std::string> present = present_banks(args.inputs[0]);
    if (std::find(present.begin(), present.end(), "REC::Particle") == present.end()) {
        std::fprintf(stderr, "error: REC::Particle not found in '%s'\n", args.inputs[0].c_str());
        return 1;
    }

    fmt::print("inputs: {} file(s), {} events\n", ch.size(), ch.total_events());
    print_run_config(args.inputs[0]);

    vz::BankList banks;
    try {
        banks = ch.getBanks(present);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not build banklist: %s\n", e.what());
        return 1;
    }

    TFile fout(args.output.c_str(), "RECREATE");
    if (fout.IsZombie()) {
        std::fprintf(stderr, "error: could not create '%s'\n", args.output.c_str());
        return 1;
    }
    TTree tree("particles", "REC::Particle kinematics");

    // Resolve bank indices (absent optional bank -> -1, like vz::BankIndex).
    auto bank_index = [&](const char* n) -> long {
        try {
            return vz::getBanklistIndex(banks, n);
        } catch (const std::exception&) {
            return -1;
        }
    };

    ParticleDumper dumper;
    dumper.rec_particle = bank_index("REC::Particle");
    dumper.rec_traj = bank_index("REC::Traj");
    dumper.raster_pos = bank_index("RASTER::position");
    dumper.tree = &tree;
    tree.Branch("event", &dumper.event, "event/L");
    tree.Branch("pid", &dumper.pid, "pid/I");
    tree.Branch("px", &dumper.px, "px/F");
    tree.Branch("py", &dumper.py, "py/F");
    tree.Branch("pz", &dumper.pz, "pz/F");
    tree.Branch("vz", &dumper.vz, "vz/F");
    tree.Branch("status", &dumper.status, "status/I");
    tree.Branch("dc_x", &dumper.dc_x, "dc_x/F");
    tree.Branch("dc_y", &dumper.dc_y, "dc_y/F");
    tree.Branch("dc_z", &dumper.dc_z, "dc_z/F");
    tree.Branch("dc_cx", &dumper.dc_cx, "dc_cx/F");
    tree.Branch("dc_cy", &dumper.dc_cy, "dc_cy/F");
    tree.Branch("dc_cz", &dumper.dc_cz, "dc_cz/F");
    tree.Branch("dc3_x", &dumper.dc3_x, "dc3_x/F");
    tree.Branch("dc3_y", &dumper.dc3_y, "dc3_y/F");
    tree.Branch("dc3_z", &dumper.dc3_z, "dc3_z/F");
    tree.Branch("dc3_cx", &dumper.dc3_cx, "dc3_cx/F");
    tree.Branch("dc3_cy", &dumper.dc3_cy, "dc3_cy/F");
    tree.Branch("dc3_cz", &dumper.dc3_cz, "dc3_cz/F");
    tree.Branch("raster_x", &dumper.raster_x, "raster_x/F");
    tree.Branch("raster_y", &dumper.raster_y, "raster_y/F");

    ch.process(banks, dumper, 100.0);

    tree.Write();
    fout.Close();

    fmt::print("wrote {} particles to {}\n", dumper.n_particles, args.output);
    return 0;
}
