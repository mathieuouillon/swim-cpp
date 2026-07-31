// Dump REC::Particle kinematics from CLAS12 HIPO files into a flat ROOT TTree.
//
// One TTree entry per reconstructed particle, holding the PID, momentum
// (px, py, pz), the z vertex (vz), the status word, the DC Region-1 and
// Region-3 trajectory states from REC::Traj (position + direction cosines,
// NaN when absent), the per-event RASTER::position beam offset, the run number,
// and the matched MC truth (true_pid / true_p / true_vz from MC::Particle via
// MC::RecMatch; 0 / NaN in real data) — e.g. for pion reco-vs-truth studies.
//
// With --ccdb FILE the beam position + solenoid z-shift are looked up from a
// CCDB SQLite snapshot by run and saved, with the run number and field scales,
// as ROOT metadata (TParameters) that vz-swim-hist reads back automatically.
//
// Reads with the hipo4 chain. Parallelism is by PROCESS: with --jobs N the
// inputs are sharded across N re-exec'd single-threaded copies, each writing a
// partial TTree, which the supervisor then merges (see process_pool.hpp).
//
// Usage:
//   hipo2root <input>... [--output FILE] [--jobs N] [--ccdb FILE] [--quiet]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <thread>  // std::thread::hardware_concurrency (default worker count only)
#include <vector>

#include <fmt/format.h>

#include "TFile.h"
#include "TTree.h"

#include "argparse/argparse.hpp"
#include "bank_access.hpp"
#include "ccdb.hpp"
#include "constants.hpp"
#include "hipo4/hipo.hpp"
#include "hipo_chain.hpp"
#include "process_pool.hpp"
#include "run_meta.hpp"

namespace {

struct cli_args {
    std::vector<std::string> inputs;
    std::string output = "output/cpp/particles.root";
    int jobs = 0;             // 0 = one worker process per hardware thread
    bool run_config = false;  // only print RUN::config (per input file) and exit
    bool all_species = false;  // keep every row; default keeps only forward e-/pi+/pi-
    bool quiet = false;
    bool worker = false;      // hidden: set by the supervisor on each re-exec
    std::string ccdb;         // CCDB sqlite snapshot; empty = skip the CCDB lookup
    std::string ccdb_variation = "default";  // CCDB variation for the lookup
    long long record_begin = -1;  // hidden: worker's global HIPO record range [begin, end)
    long long record_end = -1;
    std::string progress_file;  // hidden: shared progress-counter path (supervisor -> worker)
};

// Build the command-line parser (external/argparse). --worker is a hidden marker
// the supervisor sets on each re-exec'd child.
auto build_parser() -> argparse::parser {
    argparse::parser p("hipo2root",
                       "Dump REC::Particle kinematics from CLAS12 HIPO files into a flat ROOT TTree.");
    p.add_argument("inputs")
        .nargs(argparse::nargs::one_or_more)
        .metavar("input")
        .help(".hipo file(s)");
    p.add_argument("-o", "--output")
        .default_value("output/cpp/particles.root")
        .help("output ROOT file");
    p.add_argument("-j", "--jobs", "-t", "--threads")
        .default_value(0)
        .help("worker processes, 0 = one per hardware thread (--threads is an alias)");
    p.add_argument("--run-config")
        .flag()
        .help("only print each input's RUN::config (run, torus/solenoid scales) and exit");
    p.add_argument("--all-species")
        .flag()
        .help("keep every REC::Particle row (default: only forward-detector e-/pi+/pi- "
              "passing the |chi2pid|<3, DC-edge, and electron PCAL fiducial cuts)");
    p.add_argument("--ccdb")
        .default_value("")
        .help("CCDB SQLite snapshot; look up beam position + solenoid shift by run and save "
              "them (with the run number) in the output file");
    p.add_argument("--ccdb-variation")
        .default_value("default")
        .help("CCDB variation for the lookup (e.g. rga_spring2018, rgc_spring2023); the "
              "solenoid z-shift in particular is variation-dependent");
    p.add_argument("-q", "--quiet").flag().help("suppress the progress bar");
    p.add_argument("--worker").flag().hidden();  // re-exec'd worker marker
    p.add_argument("--record-begin").default_value(static_cast<long long>(-1)).hidden();
    p.add_argument("--record-end").default_value(static_cast<long long>(-1)).hidden();
    p.add_argument("--progress-file").default_value("").hidden();
    return p;
}

// Banks the dump reads. REC::Particle is required; the rest are optional (absent
// -> their branches are NaN / 0). RUN::config gives the run number; MC::Particle
// + MC::RecMatch give the matched MC truth (e.g. for pions) in simulation.
const std::vector<std::string> WANTED_BANKS = {
    "REC::Particle", "REC::Traj",       "RASTER::position", "REC::Calorimeter",
    "RUN::config",   "MC::Particle",    "MC::RecMatch"};

// Keep only the wanted banks present in the first file's dictionary, so the
// banklist construction never throws on a missing schema.
auto present_banks(const std::string& file) -> std::vector<std::string> {
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
auto print_run_config(const std::string& file) -> void {
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
        // Skip leading scaler/run-0 events (run==0): their RUN::config carries no
        // physics run and would send the CCDB lookup to the wrong (empty) run.
        if (c.rows() > 0 && c.get<int>("run", 0) > 0) {
            const double torus = c.get<double>("torus", 0);
            const double solenoid = c.get<double>("solenoid", 0);
            fmt::print("RUN::config: run {}, torus scale {}, solenoid scale {}\n",
                       c.get<int>("run", 0), torus, solenoid);
            // vz-swim-hist swims in the physical (cnuphys / RUN::config) polarity
            // and reverses the charge, so the scales pass straight through:
            fmt::print("  -> vz-swim-hist: --torus-scale {} --solenoid-scale {}\n", torus,
                       solenoid);
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
struct particle_dumper {
    long rec_particle = -1;  // banklist index of REC::Particle
    long rec_traj = -1;      // banklist index of REC::Traj (-1 = absent)
    long rec_calo = -1;      // banklist index of REC::Calorimeter (-1 = absent)
    long raster_pos = -1;    // banklist index of RASTER::position (-1 = absent)
    long run_config = -1;    // banklist index of RUN::config (-1 = absent)
    long mc_particle = -1;   // banklist index of MC::Particle (-1 = absent)
    long mc_recmatch = -1;   // banklist index of MC::RecMatch (-1 = absent)
    TTree* tree = nullptr;
    bool keep_all = false;   // default: stage only forward-detector e-/pi+/pi-

    // Species the vz analysis swims: electrons and charged pions.
    static auto keep_species(int pid) -> bool {
        return pid == 11 || pid == 211 || pid == -211;
    }

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
    // Run number (RUN::config), constant per event.
    Int_t run = 0;
    // Matched MC truth (MC::Particle via MC::RecMatch); 0 / NaN when unmatched or
    // in real data. Lets pion (and electron) reco be compared to truth.
    Int_t true_pid = 0;
    Float_t true_px = 0.f, true_py = 0.f, true_pz = 0.f, true_vz = 0.f;

    long long n_particles = 0;

    // One decoded REC::Particle row, staged before the Fill.
    struct row {
        Int_t pid;
        Float_t px, py, pz, vz;
        Int_t status;
        Float_t dc_x, dc_y, dc_z, dc_cx, dc_cy, dc_cz;
        Float_t dc3_x, dc3_y, dc3_z, dc3_cx, dc3_cy, dc3_cz;
        Int_t true_pid;
        Float_t true_px, true_py, true_pz, true_vz;
    };

    auto operator()(vz::bank_list& bl, int /*file_idx*/, long event_idx) -> void {
        if (rec_particle < 0) return;
        hipo::bank_view rec = bl[rec_particle];
        hipo::bank_view traj = rec_traj >= 0 ? bl[rec_traj] : hipo::bank_view{};
        hipo::bank_view calo = rec_calo >= 0 ? bl[rec_calo] : hipo::bank_view{};
        hipo::bank_view raster = raster_pos >= 0 ? bl[raster_pos] : hipo::bank_view{};
        const int n = static_cast<int>(rec.rows());
        constexpr Float_t fnan = std::numeric_limits<Float_t>::quiet_NaN();

        Float_t ras_x = fnan, ras_y = fnan;
        if (raster.rows() > 0) {
            ras_x = static_cast<Float_t>(raster.get<double>("x", 0));
            ras_y = static_cast<Float_t>(raster.get<double>("y", 0));
        }

        // Run number (constant per event).
        hipo::bank_view cfg = run_config >= 0 ? bl[run_config] : hipo::bank_view{};
        const Int_t rn = cfg.rows() > 0 ? cfg.get<int>("run", 0) : 0;

        // Map each reco row to its matched MC::Particle index via MC::RecMatch.
        hipo::bank_view mc = mc_particle >= 0 ? bl[mc_particle] : hipo::bank_view{};
        hipo::bank_view mtch = mc_recmatch >= 0 ? bl[mc_recmatch] : hipo::bank_view{};
        std::vector<int> mc_of(static_cast<std::size_t>(n), -1);
        if (mc && mtch) {
            const int mr = static_cast<int>(mtch.rows());
            for (int m = 0; m < mr; ++m) {
                const int pi = mtch.get<int>("pindex", m);
                if (pi >= 0 && pi < n)
                    mc_of[static_cast<std::size_t>(pi)] = mtch.get<int>("mcindex", m);
            }
        }
        const int mc_rows = mc ? static_cast<int>(mc.rows()) : 0;

        // Decode each REC::Particle row into a staging buffer.
        std::vector<row> rows;
        rows.clear();
        rows.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const int pid_i = rec.get<int>("pid", i);
            const int status_i = rec.get<int>("status", i);
            // Keep only the forward-detector electrons and pions the vz analysis
            // swims; skip everything else (other species, central detector)
            // before the costly REC::Traj / MC-truth lookups. --all-species
            // disables this and dumps every row, as before.
            if (!keep_all && (!keep_species(pid_i) || !vz::is_forward(status_i))) continue;
            // Quality selection (default mode only; --all-species keeps every row):
            //  - PID timing: |chi2pid| < 3 for both electrons and pions;
            //  - DC fiducial: REC::Traj edge distance per region (cm), all three
            //    present and above their thresholds (R1/R2 > 5, R3 > 10);
            //  - PCAL fiducial (electrons only): lv, lw > 9 cm.
            if (!keep_all) {
                const double chi2pid = rec.get<double>("chi2pid", i);
                if (!(chi2pid > -3.0 && chi2pid < 3.0)) continue;

                const auto edge1 = vz::layer_f64(traj, i, vz::DET_DC, vz::DC_LAYERS[0], "edge");
                const auto edge2 = vz::layer_f64(traj, i, vz::DET_DC, vz::DC_LAYERS[1], "edge");
                const auto edge3 = vz::layer_f64(traj, i, vz::DET_DC, vz::DC_LAYERS[2], "edge");
                if (!edge1 || !edge2 || !edge3) continue;
                if (!(*edge1 > 5.0 && *edge2 > 5.0 && *edge3 > 10.0)) continue;

                if (pid_i == 11) {
                    const auto lv = vz::layer_f64(calo, i, vz::DET_ECAL, vz::ECAL_PCAL_LAYER, "lv");
                    const auto lw = vz::layer_f64(calo, i, vz::DET_ECAL, vz::ECAL_PCAL_LAYER, "lw");
                    if (!lv || !lw) continue;
                    if (!(*lv > 9.0 && *lw > 9.0)) continue;
                }
            }
            row r;
            r.pid = pid_i;
            r.px = static_cast<Float_t>(rec.get<double>("px", i));
            r.py = static_cast<Float_t>(rec.get<double>("py", i));
            r.pz = static_cast<Float_t>(rec.get<double>("pz", i));
            r.vz = static_cast<Float_t>(rec.get<double>("vz", i));
            r.status = status_i;
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
            r.true_pid = 0;
            r.true_px = r.true_py = r.true_pz = r.true_vz = fnan;
            if (const int mi = mc_of[static_cast<std::size_t>(i)]; mi >= 0 && mi < mc_rows) {
                r.true_pid = mc.get<int>("pid", mi);
                r.true_px = static_cast<Float_t>(mc.get<double>("px", mi));
                r.true_py = static_cast<Float_t>(mc.get<double>("py", mi));
                r.true_pz = static_cast<Float_t>(mc.get<double>("pz", mi));
                r.true_vz = static_cast<Float_t>(mc.get<double>("vz", mi));
            }
            rows.push_back(r);
        }
        if (rows.empty()) return;

        // Copy into the branch buffers and Fill.
        event = static_cast<Long64_t>(event_idx);
        run = rn;
        raster_x = ras_x;
        raster_y = ras_y;
        for (const row& r : rows) {
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
            true_pid = r.true_pid;
            true_px = r.true_px;
            true_py = r.true_py;
            true_pz = r.true_pz;
            true_vz = r.true_vz;
            tree->Fill();
            ++n_particles;
        }
    }
};

// Read RUN::config (run number + field scales, in their physical RUN::config
// polarity) and, when a CCDB snapshot is given, the beam offset + solenoid
// z-shift for that run. Packaged for saving into the output file; vz-swim-hist
// reads it back and swims with the charge reversed to match this polarity.
auto compute_run_meta(const std::string& src_file, const std::string& ccdb_path,
                      const std::string& variation) -> vz::run_meta {
    vz::run_meta m;
    if (auto fr = hipo::file::open(src_file)) {
        hipo::file& f = *fr;
        if (auto cfg = f.bank("RUN::config")) {
            const std::uint64_t n = std::min<std::uint64_t>(100, f.event_count());
            for (std::uint64_t i = 0; i < n; ++i) {
                auto ev = f.event_at(i);
                if (!ev) continue;
                hipo::bank_view c = ev->get(*cfg);
                // Skip leading scaler/run-0 events: run==0 poisons the CCDB
                // lookup (beam/solenoid-shift would come back empty).
                const int rn = c.rows() > 0 ? c.get<int>("run", 0) : 0;
                if (rn > 0) {
                    m.run = rn;
                    m.torus_scale = c.get<double>("torus", 0);
                    m.solenoid_scale = c.get<double>("solenoid", 0);
                    break;
                }
            }
        }
    }
    if (ccdb_path.empty()) return m;
    try {
        ccdb::reader db(ccdb_path);
        // Column names vary between snapshots; try the common spellings.
        const auto first = [&](const std::string& path,
                               std::initializer_list<const char*> cols) -> std::optional<double> {
            for (const char* col : cols)
                if (auto v = db.value(path, m.run, col, variation)) return v;
            return std::nullopt;
        };
        if (auto v = first("/geometry/beam/position", {"x", "x_offset"})) m.beam_x = *v;
        if (auto v = first("/geometry/beam/position", {"y", "y_offset"})) m.beam_y = *v;
        if (auto v = first("/geometry/shifts/solenoid", {"z", "z_shift", "dz"}))
            m.solenoid_z_shift = *v;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "warning: CCDB lookup failed (%s); using defaults\n", e.what());
    }
    return m;
}

// One single-threaded pass over `paths`: dump every REC::Particle row into a
// flat TTree in args.output. In worker mode the chatter is suppressed (the
// supervisor reports); a lone --jobs 1 run prints the inputs, RUN::config, and
// total itself.
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

    if (!worker_mode) {
        fmt::print("inputs: {} file(s), {} events\n", ch.size(), ch.total_events());
        print_run_config(paths[0]);
    }

    vz::bank_list banks;
    try {
        banks = ch.get_banks(present);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not build banklist: %s\n", e.what());
        return 1;
    }

    // Match the merge compression so the supervisor can fast-clone these
    // partials' baskets (raw copy) instead of recompressing the whole tree.
    TFile fout(args.output.c_str(), "RECREATE", "", vz::merge_compression());
    if (fout.IsZombie()) {
        std::fprintf(stderr, "error: could not create '%s'\n", args.output.c_str());
        return 1;
    }
    TTree tree("particles", "REC::Particle kinematics");

    // Resolve bank indices (absent optional bank -> -1, like vz::bank_index).
    auto bank_index = [&](const char* n) -> long {
        try {
            return vz::get_banklist_index(banks, n);
        } catch (const std::exception&) {
            return -1;
        }
    };

    particle_dumper dumper;
    dumper.rec_particle = bank_index("REC::Particle");
    dumper.rec_traj = bank_index("REC::Traj");
    dumper.rec_calo = bank_index("REC::Calorimeter");
    dumper.raster_pos = bank_index("RASTER::position");
    dumper.run_config = bank_index("RUN::config");
    dumper.mc_particle = bank_index("MC::Particle");
    dumper.mc_recmatch = bank_index("MC::RecMatch");
    dumper.keep_all = args.all_species;
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
    tree.Branch("run", &dumper.run, "run/I");
    tree.Branch("true_pid", &dumper.true_pid, "true_pid/I");
    tree.Branch("true_px", &dumper.true_px, "true_px/F");
    tree.Branch("true_py", &dumper.true_py, "true_py/F");
    tree.Branch("true_pz", &dumper.true_pz, "true_pz/F");
    tree.Branch("true_vz", &dumper.true_vz, "true_vz/F");

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
            dumper(bl, fi, ei);
            if ((++prog_local & 0x3FFFU) == 0) prog.add(0x4000);
        },
        rec_begin, rec_end);
    prog.add(prog_local & 0x3FFFU);

    tree.Write();
    // The final (single, non-worker) file carries the run + CCDB metadata; worker
    // part files do not, so it is written once into the merged supervisor output.
    if (!worker_mode) compute_run_meta(paths[0], args.ccdb, args.ccdb_variation).write(fout);
    fout.Close();

    if (!worker_mode)
        fmt::print("wrote {} {} to {}\n", dumper.n_particles,
                   args.all_species ? "particles" : "forward e-/pi+/pi- rows", args.output);
    return 0;
}

// Supervisor: split the work across `jobs` re-exec'd workers and merge their
// partial TTrees into args.output. With at least `jobs` input files each worker
// takes a subset of files; with fewer (e.g. a single big train file) the work
// is sharded by HIPO record range instead, so one file still parallelizes.
auto run_supervisor(const cli_args& args, char** argv, const std::vector<std::string>& paths,
                    int jobs) -> int {
    const bool by_file = paths.size() >= static_cast<std::size_t>(jobs);

    // Open the inputs once: total events (for the aggregate progress bar) and
    // total records (to record-shard one/few files). On a multi-GB train file
    // this index scan takes a moment, so announce it for immediate feedback.
    if (!args.quiet) {
        fmt::print("hipo2root: scanning {} input file(s)...\n", paths.size());
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
        if (n_workers <= 1) return run_dump(args, paths, /*worker_mode=*/false);  // nothing to split
    }

    const std::string exe = vz::self_exe_path(argv[0]);
    std::string dir;
    try {
        dir = vz::make_temp_dir("hipo2root");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    const std::string progress_path = fmt::format("{}/progress", dir);
    vz::shared_counter progress = vz::shared_counter::map(progress_path, /*create=*/true);

    std::vector<std::string> parts;
    std::vector<std::vector<std::string>> argvs;
    const std::uint64_t chunk = by_file ? 0 : (total_records + n_workers - 1) / n_workers;
    for (int i = 0; i < n_workers; ++i) {
        const std::string part = fmt::format("{}/part-{:03}.root", dir, i);
        parts.push_back(part);
        std::vector<std::string> av = {exe,      "--jobs",          "1",      "--worker", "--quiet",
                                       "--output", part, "--progress-file", progress_path};
        if (args.all_species) av.push_back("--all-species");
        if (by_file) {
            for (std::size_t j = i; j < paths.size(); j += static_cast<std::size_t>(n_workers))
                av.push_back(paths[j]);
        } else {
            const std::uint64_t rb = static_cast<std::uint64_t>(i) * chunk;
            const std::uint64_t re = std::min<std::uint64_t>(rb + chunk, total_records);
            av.insert(av.end(), {"--record-begin", fmt::format("{}", rb), "--record-end",
                                 fmt::format("{}", re)});
            for (const auto& p : paths) av.push_back(p);
        }
        argvs.push_back(std::move(av));
    }

    fmt::print("hipo2root: {} worker process(es) sharded by {} over {} -> {}\n", n_workers,
               by_file ? "file" : "record",
               by_file ? fmt::format("{} file(s)", paths.size())
                       : fmt::format("{} records in {} file(s)", total_records, paths.size()),
               args.output);
    if (!args.all_species) fmt::print("  keeping forward-detector e-/pi+/pi- rows only\n");
    print_run_config(paths[0]);

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
    if (!args.quiet) {
        fmt::print("hipo2root: merging {} partial file(s) -> {} ...\n", parts.size(), args.output);
        std::fflush(stdout);
    }
    const auto t_merge = std::chrono::steady_clock::now();
    try {
        vz::merge_root_files(parts, args.output);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: merge failed: %s\n", e.what());
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        return 1;
    }
    if (!args.quiet)
        fmt::print(
            "hipo2root: merged in {:.1f}s\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_merge).count());
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    // Write the run + CCDB metadata into the merged file, and read back the total.
    long long total = 0;
    {
        TFile f(args.output.c_str(), "UPDATE");
        if (!f.IsZombie()) {
            compute_run_meta(paths[0], args.ccdb, args.ccdb_variation).write(f);
            if (auto* t = dynamic_cast<TTree*>(f.Get("particles"))) total = t->GetEntries();
            f.Close();
        }
    }
    fmt::print("wrote {} particles to {}\n", total, args.output);
    return 0;
}

}  // namespace

auto main(int argc, char** argv) -> int {
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
    args.run_config = p.get<bool>("--run-config");
    args.all_species = p.get<bool>("--all-species");
    args.quiet = p.get<bool>("--quiet");
    args.worker = p.get<bool>("--worker");
    args.ccdb = p.get<std::string>("--ccdb");
    args.ccdb_variation = p.get<std::string>("--ccdb-variation");
    args.record_begin = p.get<long long>("--record-begin");
    args.record_end = p.get<long long>("--record-end");
    args.progress_file = p.get<std::string>("--progress-file");

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

    if (args.inputs.empty()) {
        std::fprintf(stderr, "error: no input files\n");
        return 1;
    }

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int want = args.jobs > 0 ? args.jobs : static_cast<int>(hw);
    // Ensure the output directory (default output/cpp/) exists before any write.
    vz::ensure_parent_dir(args.output);

    // A re-exec'd worker dumps its assigned files / record range. A lone run dumps
    // everything. Otherwise the supervisor splits across `want` workers — by file
    // when there are enough, else by HIPO record range so one file parallelizes.
    if (args.worker) return run_dump(args, args.inputs, /*worker_mode=*/true);
    if (want <= 1) return run_dump(args, args.inputs, /*worker_mode=*/false);
    return run_supervisor(args, argv, args.inputs, want);
}
