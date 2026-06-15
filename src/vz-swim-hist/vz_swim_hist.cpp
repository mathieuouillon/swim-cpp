// Charged-track v_z in (p, theta) bins, with a DC-R1 -> beamline swim.
//
// Reads the flat `particles` TTree written by hipo2root, selects Forward-
// Detector tracks of one species by reconstructed pid (--pid, default 11 = e-;
// e.g. 211 / -211 for pions, with the swim charge taken from the pid), swims
// their DC trajectory state back to the beamline through the CLAS12 field (same
// swimmer and field configuration as swim-analysis), and histograms per
// (p, theta) cell:
//   vz_rec_pAA_BB_thCC_DD   EB-reconstructed v_z          [-13, 0] cm
//   vz_swum_pAA_BB_thCC_DD  swum-back v_z                 [-13, 0] cm
//   dvz_pAA_BB_thCC_DD      swum - rec                    [-10, 10] cm
// where AA_BB are the momentum-bin edges in GeV and CC_DD the theta-bin edges
// in degrees, both zero-padded to two digits. All three are filled only when
// the swim converged, so they share the same sample.
//
// Grid: p in [2, 10) GeV, 1-GeV bins (8); theta in [6, 26) deg, 2-deg bins (10).
//
// The swim replicates coatjava's vertex procedure (see docs/coatjava_vz.md):
// it starts by default from the DC Region-3 state (--dc-region 1 reverts to
// Region 1) and targets the DOCA to the beam position x_b = --beam-x +
// per-event RASTER::position x (same for y), as DC tracking's SwimToBeamLine
// does — not the z-axis.
//
// The entry loop is split across PROCESSES, not threads: with --jobs N the
// program re-execs itself N times (see process_pool.hpp), each worker reading a
// contiguous entry range through its own TChain into its own histogram grid and
// partial ROOT file, which the supervisor then merges. Farm-friendly: no ROOT
// thread-safety locking; each worker reloads the read-only field map.
//
// Usage:
//   vz-swim-hist <input.root>... [--output FILE] [--jobs N] [--max-entries N]
//                [--pid PID] [--dc-region 1|3] [--beam-x X] [--beam-y Y]
//                [--torus PATH] [--solenoid PATH]
//                [--torus-scale X] [--solenoid-scale X] [--solenoid-z-shift X]
//                [--torus-z-shift X] [--torus-x-shift X] [--torus-y-shift X]
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>  // std::thread::hardware_concurrency (default worker count only)
#include <vector>

#include <fmt/format.h>

#include "TChain.h"
#include "TFile.h"
#include "TH1D.h"

#include "argparse/argparse.hpp"
#include "constants.hpp"
#include "field.hpp"
#include "process_pool.hpp"
#include "progresstracker.hpp"
#include "run_meta.hpp"
#include "swim.hpp"

namespace {

// (p, theta) grid: 1-GeV momentum bins over [2, 10), 2-deg theta bins over [6, 26).
constexpr std::size_t N_P = 8;
constexpr double P_MIN = 2.0, P_WIDTH = 1.0;
constexpr std::size_t N_TH = 10;
constexpr double TH_MIN = 6.0, TH_WIDTH = 2.0;

// Forward-Detector sectors: 6, each 60 deg wide; sector 1 centred on phi=0.
constexpr std::size_t N_SEC = 6;

// CLAS12 sector index (0..5) from the lab azimuth phi [deg]: sector 1 spans
// [-30, 30) about +x, sector 2 [30, 90), ...
inline auto sector_index(double phi_deg) -> std::size_t {
    double a = std::fmod(phi_deg + 30.0, 360.0);
    if (a < 0.0) a += 360.0;
    return static_cast<std::size_t>(a / 60.0) % N_SEC;
}

// v_z axis: the RG-D dual-target region (as in the reference electron-vz
// figure); dvz axis from constants.
constexpr int VZ_HIST_BINS = 200;
constexpr double VZ_HIST_MIN = -13.0;
constexpr double VZ_HIST_MAX = 0.0;

struct cli_args {
    std::vector<std::string> inputs;
    std::string output = "output/cpp/vz_bins.root";
    int jobs = 0;           // 0 = one worker process per hardware thread
    long long max_entries = -1;  // -1 = all tree entries (particles)
    bool quiet = false;
    // Hidden worker fields, set by the supervisor on each re-exec: a non-empty
    // emit_counts marks a worker, which processes [entry_begin, entry_end),
    // writes its counts sidecar there, and stays silent.
    std::string emit_counts;
    long long entry_begin = -1;
    long long entry_end = -1;
    // Swim configuration, mirroring coatjava's vertex procedure.
    int dc_region = 3;    // start state: DC region 3 (coatjava) or 1
    int pid = 11;         // species to analyze by reco pid: 11 e-, 211 pi+, -211 pi-
    double beam_x = 0.0;  // CCDB /geometry/beam/position x_offset [cm]
    double beam_y = 0.0;  // CCDB /geometry/beam/position y_offset [cm]
    // magnetic_field configuration. NOTE the scale signs are in OUR field_map convention
    // (scale multiplies the map values verbatim), which is the OPPOSITE of
    // RUN::config's for BOTH magnets: run 18614 reports torus +1 / solenoid -1,
    // and torus -1 / solenoid +1 here best reproduces its pass1-reconstructed
    // vz (wrong torus sign sends every swum vz far outside the target region;
    // wrong solenoid sign visibly degrades the rec/swum agreement). Rule:
    // --torus-scale = -(RUN::config torus), --solenoid-scale = -(RUN::config
    // solenoid). See docs/coatjava_vz.md.
    // --solenoid-z-shift: the reconstruction uses the CCDB
    // /geometry/shifts/solenoid z value for the run (-3.0 for 18614).
    std::string torus = "Full_torus_r501_phi361_z501_31Mar2021.dat";
    std::string solenoid = "Symm_solenoid_r601_phi1_z1201_21May2019.dat";
    double torus_scale = -1.0;
    double solenoid_scale = 1.0;
    double solenoid_z_shift = -3.0;
    // Rigid torus-map displacements [cm] (0 = nominal survey position). A
    // transverse (x/y) shift is the diagnostic for a beamline-misalignment-like
    // 1/tan(theta) vz trend the solenoid z-shift cannot remove.
    double torus_z_shift = 0.0;
    double torus_x_shift = 0.0;
    double torus_y_shift = 0.0;
    // DC alignment probe: rigidly translate the DC start state (position only)
    // before swimming [cm]. Tests a DC misalignment as the source of the
    // momentum-independent (geometric) part of the vz theta-walk.
    double dc_x_shift = 0.0;
    double dc_y_shift = 0.0;
    double dc_z_shift = 0.0;
};

// Build the command-line parser (external/argparse). The hidden flags
// (--emit-counts / --entry-begin / --entry-end) are set by the supervisor on
// each re-exec'd worker and kept out of --help.
auto build_parser() -> argparse::parser {
    argparse::parser p("vz-swim-hist",
                       "Electron v_z in (p, theta) bins with a DC -> beamline swim, from the "
                       "flat particles tree written by hipo2root.");
    p.add_argument("inputs")
        .nargs(argparse::nargs::one_or_more)
        .metavar("input.root")
        .help("particles.root tree(s) from hipo2root");
    p.add_argument("-o", "--output")
        .default_value("output/cpp/vz_bins.root")
        .help("output ROOT file");
    p.add_argument("-j", "--jobs", "-t", "--threads")
        .default_value(0)
        .help("worker processes, 0 = one per hardware thread (--threads is an alias)");
    p.add_argument("-n", "--max-entries", "--max-events")
        .default_value(static_cast<long long>(-1))
        .help("process only the first N particles (-1 = all)");
    p.add_argument("--dc-region")
        .default_value(3)
        .choices({1, 3})
        .help("DC start state: region 1 or 3 (coatjava uses 3)");
    p.add_argument("--pid")
        .default_value(11)
        .help("species to analyze by reconstructed pid (11 = e-, 211 = pi+, -211 = pi-)");
    p.add_argument("--beam-x").default_value(0.0).help("CCDB beam x offset [cm]");
    p.add_argument("--beam-y").default_value(0.0).help("CCDB beam y offset [cm]");
    p.add_argument("--torus")
        .default_value("Full_torus_r501_phi361_z501_31Mar2021.dat")
        .help("torus field map");
    p.add_argument("--solenoid")
        .default_value("Symm_solenoid_r601_phi1_z1201_21May2019.dat")
        .help("solenoid field map");
    p.add_argument("--torus-scale").default_value(-1.0).help("torus field scale");
    p.add_argument("--solenoid-scale").default_value(1.0).help("solenoid field scale");
    p.add_argument("--solenoid-z-shift").default_value(-3.0).help("solenoid map z-shift [cm]");
    p.add_argument("--torus-z-shift").default_value(0.0).help("torus map z-shift [cm]");
    p.add_argument("--torus-x-shift").default_value(0.0).help("torus map x-shift [cm]");
    p.add_argument("--torus-y-shift").default_value(0.0).help("torus map y-shift [cm]");
    p.add_argument("--dc-x-shift").default_value(0.0).help("DC start-state x-shift [cm]");
    p.add_argument("--dc-y-shift").default_value(0.0).help("DC start-state y-shift [cm]");
    p.add_argument("--dc-z-shift").default_value(0.0).help("DC start-state z-shift [cm]");
    p.add_argument("-q", "--quiet").flag().help("suppress the progress bar");
    p.add_argument("--emit-counts").default_value("").hidden();
    p.add_argument("--entry-begin").default_value(static_cast<long long>(-1)).hidden();
    p.add_argument("--entry-end").default_value(static_cast<long long>(-1)).hidden();
    return p;
}

// Copy every parsed value into the cli_args the rest of the program reads.
auto read_args(const argparse::parser& p) -> cli_args {
    cli_args a;
    a.inputs = p.get<std::vector<std::string>>("inputs");
    a.output = p.get<std::string>("--output");
    a.jobs = p.get<int>("--jobs");
    a.max_entries = p.get<long long>("--max-entries");
    a.quiet = p.get<bool>("--quiet");
    a.emit_counts = p.get<std::string>("--emit-counts");
    a.entry_begin = p.get<long long>("--entry-begin");
    a.entry_end = p.get<long long>("--entry-end");
    a.dc_region = p.get<int>("--dc-region");
    a.pid = p.get<int>("--pid");
    a.beam_x = p.get<double>("--beam-x");
    a.beam_y = p.get<double>("--beam-y");
    a.torus = p.get<std::string>("--torus");
    a.solenoid = p.get<std::string>("--solenoid");
    a.torus_scale = p.get<double>("--torus-scale");
    a.solenoid_scale = p.get<double>("--solenoid-scale");
    a.solenoid_z_shift = p.get<double>("--solenoid-z-shift");
    a.torus_z_shift = p.get<double>("--torus-z-shift");
    a.torus_x_shift = p.get<double>("--torus-x-shift");
    a.torus_y_shift = p.get<double>("--torus-y-shift");
    a.dc_x_shift = p.get<double>("--dc-x-shift");
    a.dc_y_shift = p.get<double>("--dc-y-shift");
    a.dc_z_shift = p.get<double>("--dc-z-shift");
    return a;
}

// Default the field config (field scales, solenoid z-shift, beam offset) from the
// input file's run metadata — hipo2root writes it from RUN::config + CCDB — for
// any of those the user did not pass explicitly on the command line. Silent when
// the file carries no metadata (an older particles.root). Workers receive the
// already-resolved values from the supervisor via argv, so this runs only in the
// user-facing (lone / supervisor) process.
auto apply_run_meta(cli_args& a, const argparse::parser& p) -> void {
    if (a.inputs.empty()) return;
    TFile f(a.inputs.front().c_str());
    if (f.IsZombie()) return;
    vz::run_meta m;
    m.torus_scale = a.torus_scale;
    m.solenoid_scale = a.solenoid_scale;
    m.solenoid_z_shift = a.solenoid_z_shift;
    m.beam_x = a.beam_x;
    m.beam_y = a.beam_y;
    if (!vz::run_meta::read(f, m)) return;
    if (!p.is_used("--torus-scale")) a.torus_scale = m.torus_scale;
    if (!p.is_used("--solenoid-scale")) a.solenoid_scale = m.solenoid_scale;
    if (!p.is_used("--solenoid-z-shift")) a.solenoid_z_shift = m.solenoid_z_shift;
    if (!p.is_used("--beam-x")) a.beam_x = m.beam_x;
    if (!p.is_used("--beam-y")) a.beam_y = m.beam_y;
    if (!a.quiet)
        fmt::print("run {} field config from {} metadata: torus_scale={} solenoid_scale={} "
                   "solenoid_z_shift={} beam=({}, {})\n",
                   m.run, a.inputs.front(), a.torus_scale, a.solenoid_scale, a.solenoid_z_shift,
                   a.beam_x, a.beam_y);
}

// `pAA_BB_thCC_DD` name fragment for cell (pb, tb), edges zero-padded.
auto cell_tag(std::size_t pb, std::size_t tb) -> std::string {
    const int plo = static_cast<int>(std::lround(P_MIN + pb * P_WIDTH));
    const int phi = static_cast<int>(std::lround(P_MIN + (pb + 1) * P_WIDTH));
    const int tlo = static_cast<int>(std::lround(TH_MIN + tb * TH_WIDTH));
    const int thi = static_cast<int>(std::lround(TH_MIN + (tb + 1) * TH_WIDTH));
    return fmt::format("p{:02d}_{:02d}_th{:02d}_{:02d}", plo, phi, tlo, thi);
}

// `sS_pAA_BB_thCC_DD` name fragment for a (sector, p, theta) cell.
auto sector_tag(std::size_t s, std::size_t pb, std::size_t tb) -> std::string {
    return fmt::format("s{}_{}", s + 1, cell_tag(pb, tb));
}

// The (p x theta) grid plus a (sector x theta) grid (p-integrated over the
// selection), three histograms per cell. Each worker owns one (filled without
// locking); the workers' grids are merged into the main one at the end.
struct cell_grid {
    std::array<std::array<std::unique_ptr<TH1D>, N_TH>, N_P> rec, swum, dvz;
    // (sector x p x theta): lets us test whether the per-sector vz modulation
    // is momentum-dependent (torus-coil field ~ 1/p) or flat (DC alignment).
    using sec_grid =
        std::array<std::array<std::array<std::unique_ptr<TH1D>, N_TH>, N_P>, N_SEC>;
    sec_grid sec_rec, sec_swum, sec_dvz;

    cell_grid() {
        for (std::size_t pb = 0; pb < N_P; ++pb) {
            for (std::size_t tb = 0; tb < N_TH; ++tb) {
                const std::string tag = cell_tag(pb, tb);
                const std::string range = fmt::format(
                    "e^{{-}}, {}<p<{} GeV, {}<#theta<{}#circ", P_MIN + pb * P_WIDTH,
                    P_MIN + (pb + 1) * P_WIDTH, TH_MIN + tb * TH_WIDTH,
                    TH_MIN + (tb + 1) * TH_WIDTH);
                rec[pb][tb] = std::make_unique<TH1D>(
                    ("vz_rec_" + tag).c_str(), (range + ";v_{z} (rec) [cm];counts").c_str(),
                    VZ_HIST_BINS, VZ_HIST_MIN, VZ_HIST_MAX);
                swum[pb][tb] = std::make_unique<TH1D>(
                    ("vz_swum_" + tag).c_str(), (range + ";v_{z} (swum) [cm];counts").c_str(),
                    VZ_HIST_BINS, VZ_HIST_MIN, VZ_HIST_MAX);
                dvz[pb][tb] = std::make_unique<TH1D>(
                    ("dvz_" + tag).c_str(),
                    (range + ";v_{z}^{swum} - v_{z}^{rec} [cm];counts").c_str(),
                    vz::SWUM_DVZ_BINS, vz::SWUM_DVZ_MIN, vz::SWUM_DVZ_MAX);
            }
        }
        // Per-sector grid, momentum-resolved (sector x p x theta).
        for (std::size_t s = 0; s < N_SEC; ++s) {
            for (std::size_t pb = 0; pb < N_P; ++pb) {
                for (std::size_t tb = 0; tb < N_TH; ++tb) {
                    const std::string tag = sector_tag(s, pb, tb);
                    const std::string range = fmt::format(
                        "e^{{-}}, sector {}, {}<p<{} GeV, {}<#theta<{}#circ", s + 1,
                        P_MIN + pb * P_WIDTH, P_MIN + (pb + 1) * P_WIDTH, TH_MIN + tb * TH_WIDTH,
                        TH_MIN + (tb + 1) * TH_WIDTH);
                    sec_rec[s][pb][tb] = std::make_unique<TH1D>(
                        ("vzsec_rec_" + tag).c_str(), (range + ";v_{z} (rec) [cm];counts").c_str(),
                        VZ_HIST_BINS, VZ_HIST_MIN, VZ_HIST_MAX);
                    sec_swum[s][pb][tb] = std::make_unique<TH1D>(
                        ("vzsec_swum_" + tag).c_str(),
                        (range + ";v_{z} (swum) [cm];counts").c_str(), VZ_HIST_BINS, VZ_HIST_MIN,
                        VZ_HIST_MAX);
                    sec_dvz[s][pb][tb] = std::make_unique<TH1D>(
                        ("vzsec_dvz_" + tag).c_str(),
                        (range + ";v_{z}^{swum} - v_{z}^{rec} [cm];counts").c_str(),
                        vz::SWUM_DVZ_BINS, vz::SWUM_DVZ_MIN, vz::SWUM_DVZ_MAX);
                }
            }
        }
    }

    auto merge(const cell_grid& other) -> void {
        for (std::size_t pb = 0; pb < N_P; ++pb) {
            for (std::size_t tb = 0; tb < N_TH; ++tb) {
                rec[pb][tb]->Add(other.rec[pb][tb].get());
                swum[pb][tb]->Add(other.swum[pb][tb].get());
                dvz[pb][tb]->Add(other.dvz[pb][tb].get());
            }
        }
        for (std::size_t s = 0; s < N_SEC; ++s) {
            for (std::size_t pb = 0; pb < N_P; ++pb) {
                for (std::size_t tb = 0; tb < N_TH; ++tb) {
                    sec_rec[s][pb][tb]->Add(other.sec_rec[s][pb][tb].get());
                    sec_swum[s][pb][tb]->Add(other.sec_swum[s][pb][tb].get());
                    sec_dvz[s][pb][tb]->Add(other.sec_dvz[s][pb][tb].get());
                }
            }
        }
    }
};

// Charge [e] of a particle from its PDG id, for the species this analysis can
// select via --pid (e-, pi+/-, mu+/-, K+/-, p); falls back to the id sign.
auto charge_of(int pid) -> double {
    switch (pid) {
        case 11: case 13: case -211: case -321: case -2212: return -1.0;  // e- mu- pi- K- pbar
        case -11: case -13: case 211: case 321: case 2212: return 1.0;    // e+ mu+ pi+ K+ p
        default: return pid > 0 ? 1.0 : -1.0;
    }
}

// One worker: read entries [begin, end) through a private TChain, swim, and
// fill the private cell_grid. Shares only the read-only field map and the
// (thread-safe) progress tracker.
struct swim_worker {
    const cli_args* args = nullptr;
    const vz::magnetic_field* field = nullptr;
    Long64_t begin = 0, end = 0;
    progress_tracker* progress = nullptr;  // nullptr = no progress display

    cell_grid grid;
    // Cut flow.
    long long n_fd = 0;  // Forward-Detector tracks of the selected species (--pid)
    long long n_in_grid = 0;      // ... inside the (p, theta) grid
    long long n_with_dc = 0;      // ... with the chosen DC trajectory state
    long long n_swim_ok = 0;      // ... whose swim converged (-> filled)
    double sum_swum_vz = 0.0;     // sum of converged swum vz (config sanity check)

    auto operator()() -> void {
        TChain tree("particles");
        for (const auto& p : args->inputs) tree.Add(p.c_str());

        Int_t pid = 0, status = 0;
        Float_t px = 0.f, py = 0.f, pz = 0.f, vz_rec = 0.f;
        Float_t dc_x = 0.f, dc_y = 0.f, dc_z = 0.f, dc_cx = 0.f, dc_cy = 0.f, dc_cz = 0.f;
        Float_t raster_x = 0.f, raster_y = 0.f;
        tree.SetBranchAddress("pid", &pid);
        tree.SetBranchAddress("status", &status);
        tree.SetBranchAddress("px", &px);
        tree.SetBranchAddress("py", &py);
        tree.SetBranchAddress("pz", &pz);
        tree.SetBranchAddress("vz", &vz_rec);
        // Start state: DC region 3 (the dc3_* branches, coatjava's choice) or
        // region 1 (dc_*).
        const char* pre = args->dc_region == 3 ? "dc3" : "dc";
        auto br = [&](const char* col) { return std::string(pre) + "_" + col; };
        tree.SetBranchAddress(br("x").c_str(), &dc_x);
        tree.SetBranchAddress(br("y").c_str(), &dc_y);
        tree.SetBranchAddress(br("z").c_str(), &dc_z);
        tree.SetBranchAddress(br("cx").c_str(), &dc_cx);
        tree.SetBranchAddress(br("cy").c_str(), &dc_cy);
        tree.SetBranchAddress(br("cz").c_str(), &dc_cz);
        tree.SetBranchAddress("raster_x", &raster_x);
        tree.SetBranchAddress("raster_y", &raster_y);

        const double q = charge_of(args->pid);  // swim charge for the selected species

        constexpr Long64_t PROGRESS_CHUNK = 8192;  // amortize the atomic add
        Long64_t since_report = 0;
        for (Long64_t i = begin; i < end; ++i) {
            tree.GetEntry(i);
            if (progress && ++since_report == PROGRESS_CHUNK) {
                progress->add(static_cast<std::size_t>(PROGRESS_CHUNK));
                since_report = 0;
            }
            if (pid != args->pid) continue;
            if (!vz::is_forward(status)) continue;
            ++n_fd;

            const double p = std::sqrt(double(px) * px + double(py) * py + double(pz) * pz);
            if (p <= 0.0) continue;
            const double theta = std::acos(std::clamp(double(pz) / p, -1.0, 1.0)) * vz::RAD2DEG;
            const double fp = (p - P_MIN) / P_WIDTH;
            const double ft = (theta - TH_MIN) / TH_WIDTH;
            if (fp < 0.0 || fp >= N_P || ft < 0.0 || ft >= N_TH) continue;
            const auto pb = static_cast<std::size_t>(fp);
            const auto tb = static_cast<std::size_t>(ft);
            ++n_in_grid;

            if (!std::isfinite(dc_x) || !std::isfinite(dc_cx)) continue;  // no DC row
            ++n_with_dc;

            // Beam target: CCDB offset (CLI) + per-event raster, as in
            // coatjava's DCTBEngine.
            double x_b = args->beam_x, y_b = args->beam_y;
            if (std::isfinite(raster_x) && std::isfinite(raster_y)) {
                x_b += raster_x;
                y_b += raster_y;
            }

            // DC alignment probe: rigidly translate the start position.
            const std::array<double, 3> pos = {dc_x + args->dc_x_shift, dc_y + args->dc_y_shift,
                                               dc_z + args->dc_z_shift};
            const std::array<double, 3> mom = {p * dc_cx, p * dc_cy, p * dc_cz};
            const vz::swim_result sw =
                vz::swim_back_to_beamline(*field, pos, mom, /*q=*/q, x_b, y_b);
            if (sw.status != vz::swim_status::converged || sw.doca_rho >= vz::SWUM_MAX_DOCA_RHO)
                continue;
            ++n_swim_ok;

            sum_swum_vz += sw.vz;

            grid.rec[pb][tb]->Fill(vz_rec);
            grid.swum[pb][tb]->Fill(sw.vz);
            grid.dvz[pb][tb]->Fill(sw.vz - vz_rec);

            // Same fills, binned by sector (from the momentum azimuth).
            const double phi = std::atan2(double(py), double(px)) * vz::RAD2DEG;
            const std::size_t sec = sector_index(phi);
            grid.sec_rec[sec][pb][tb]->Fill(vz_rec);
            grid.sec_swum[sec][pb][tb]->Fill(sw.vz);
            grid.sec_dvz[sec][pb][tb]->Fill(sw.vz - vz_rec);
        }
        if (progress && since_report > 0) progress->add(static_cast<std::size_t>(since_report));
    }
};

// Cut-flow counters the histogram file does not hold; the supervisor sums one
// per worker. Plain text, one value per line, in declaration order.
struct swim_counts {
    long long n_entries = 0;
    long long n_fd = 0;
    long long n_in_grid = 0;
    long long n_with_dc = 0;
    long long n_swim_ok = 0;
    double sum_swum_vz = 0.0;

    auto write(const std::string& path) const -> void {
        std::ofstream out(path);
        if (!out) throw std::runtime_error(fmt::format("cannot write counts '{}'", path));
        out << n_entries << '\n'
            << n_fd << '\n'
            << n_in_grid << '\n'
            << n_with_dc << '\n'
            << n_swim_ok << '\n'
            << sum_swum_vz << '\n';
    }
    auto add_from_file(const std::string& path) -> void {
        std::ifstream in(path);
        if (!in) throw std::runtime_error(fmt::format("cannot read counts '{}'", path));
        long long ne = 0, nf = 0, ng = 0, nd = 0, no = 0;
        double sv = 0.0;
        in >> ne >> nf >> ng >> nd >> no >> sv;
        n_entries += ne;
        n_fd += nf;
        n_in_grid += ng;
        n_with_dc += nd;
        n_swim_ok += no;
        sum_swum_vz += sv;
    }
};

// Write a finished grid to a ROOT file (zstd-compressed, matching the merge).
auto write_grid(const cell_grid& grid, const std::string& path) -> void {
    TFile fout(path.c_str(), "RECREATE");
    if (fout.IsZombie()) throw std::runtime_error(fmt::format("cannot create '{}'", path));
    fout.SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kZSTD);
    fout.SetCompressionLevel(5);
    for (std::size_t pb = 0; pb < N_P; ++pb)
        for (std::size_t tb = 0; tb < N_TH; ++tb) {
            grid.rec[pb][tb]->Write();
            grid.swum[pb][tb]->Write();
            grid.dvz[pb][tb]->Write();
        }
    for (std::size_t s = 0; s < N_SEC; ++s)
        for (std::size_t pb = 0; pb < N_P; ++pb)
            for (std::size_t tb = 0; tb < N_TH; ++tb) {
                grid.sec_rec[s][pb][tb]->Write();
                grid.sec_swum[s][pb][tb]->Write();
                grid.sec_dvz[s][pb][tb]->Write();
            }
    fout.Close();
}

// The run report (cut flow + histogram tally), printed by the user-facing
// process (a lone --jobs 1 run, or the supervisor after merging the workers).
auto print_summary(const cli_args& a, const swim_counts& c, const std::string& output) -> void {
    fmt::print("----------------------------------------\n");
    fmt::print("particles          : {}\n", c.n_entries);
    fmt::print("FD tracks (pid {:>4}): {}\n", a.pid, c.n_fd);
    fmt::print("  in (p,theta) grid: {}\n", c.n_in_grid);
    fmt::print("  with DC-R{} state : {}\n", a.dc_region, c.n_with_dc);
    fmt::print("  swim converged   : {}  (filled)\n", c.n_swim_ok);
    if (c.n_swim_ok > 0) {
        const double mean_vz = c.sum_swum_vz / static_cast<double>(c.n_swim_ok);
        fmt::print("  mean swum vz     : {:.2f} cm{}\n", mean_vz,
                   (mean_vz < VZ_HIST_MIN || mean_vz > VZ_HIST_MAX)
                       ? "  ** OUTSIDE the histogram range -- check field scales/shifts **"
                       : "");
    }
    fmt::print("histograms written to {}: {} ((p,theta) {} + (sector,p,theta) {}) x 3\n", output,
               (N_P * N_TH + N_SEC * N_P * N_TH) * 3, N_P * N_TH, N_SEC * N_P * N_TH);
}

// The swim-config flags shared by every worker re-exec (everything except the
// per-worker output/counts/entry-range and the input files).
auto base_worker_argv(const std::string& exe, const cli_args& a) -> std::vector<std::string> {
    return {exe,
            "--jobs", "1", "--quiet",
            "--pid", fmt::format("{}", a.pid),
            "--dc-region", fmt::format("{}", a.dc_region),
            "--beam-x", fmt::format("{}", a.beam_x),
            "--beam-y", fmt::format("{}", a.beam_y),
            "--torus", a.torus, "--solenoid", a.solenoid,
            "--torus-scale", fmt::format("{}", a.torus_scale),
            "--solenoid-scale", fmt::format("{}", a.solenoid_scale),
            "--solenoid-z-shift", fmt::format("{}", a.solenoid_z_shift),
            "--torus-z-shift", fmt::format("{}", a.torus_z_shift),
            "--torus-x-shift", fmt::format("{}", a.torus_x_shift),
            "--torus-y-shift", fmt::format("{}", a.torus_y_shift),
            "--dc-x-shift", fmt::format("{}", a.dc_x_shift),
            "--dc-y-shift", fmt::format("{}", a.dc_y_shift),
            "--dc-z-shift", fmt::format("{}", a.dc_z_shift)};
}

// Probe the input tree once: validate every file and return the entry count.
// Returns -1 on the first file missing a 'particles' tree.
auto probe_entries(const cli_args& args) -> Long64_t {
    TChain tree("particles");
    for (const auto& p : args.inputs) {
        if (tree.Add(p.c_str(), /*nentries=*/-1) == 0) {
            std::fprintf(stderr, "error: no 'particles' tree in '%s'\n", p.c_str());
            return -1;
        }
    }
    return tree.GetEntries();
}

// Run one entry range [begin, end) serially into a single grid (no threads).
// Suppresses the progress bar in worker mode (the supervisor reports instead).
auto run_worker(const cli_args& args, Long64_t begin, Long64_t end, bool worker_mode) -> int {
    if (!worker_mode)
        fmt::print("loading field maps: torus={} (scale {}, shift x={} y={} z={} cm), "
                   "solenoid={} (scale {}, z-shift {} cm)\n",
                   args.torus, args.torus_scale, args.torus_x_shift, args.torus_y_shift,
                   args.torus_z_shift, args.solenoid, args.solenoid_scale, args.solenoid_z_shift);
    std::unique_ptr<vz::composite_field> field_ptr;
    try {
        field_ptr = std::make_unique<vz::composite_field>(vz::composite_field::load(
            args.torus, args.torus_scale, args.solenoid, args.solenoid_scale,
            args.solenoid_z_shift, args.torus_z_shift, args.torus_x_shift, args.torus_y_shift));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not load field maps: %s\n", e.what());
        return 1;
    }

    std::unique_ptr<progress_tracker> progress;
    if (!args.quiet && !worker_mode && end > begin) {
        progress_tracker::config config;
        config.label = "Swimming";
        config.show_eta = true;
        config.show_rate = true;
        progress = std::make_unique<progress_tracker>(static_cast<std::size_t>(end - begin), config);
        progress->start();
    }

    swim_worker worker;
    worker.args = &args;
    worker.field = field_ptr.get();
    worker.progress = progress.get();
    worker.begin = begin;
    worker.end = end;
    worker();
    if (progress) progress->finish();

    try {
        write_grid(worker.grid, args.output);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    swim_counts c{end - begin,         worker.n_fd, worker.n_in_grid,
                  worker.n_with_dc,    worker.n_swim_ok,     worker.sum_swum_vz};
    if (worker_mode) {
        try {
            c.write(args.emit_counts);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error: could not write counts: %s\n", e.what());
            return 1;
        }
    } else {
        print_summary(args, c, args.output);
    }
    return 0;
}

// Supervisor: shard [0, n_entries) into contiguous ranges across `n_workers`
// re-exec'd workers, merge their partial ROOT files, and print the summed
// summary.
auto run_supervisor(const cli_args& args, char** argv, Long64_t n_entries, Long64_t total_entries,
                    int n_workers) -> int {
    const std::string exe = vz::self_exe_path(argv[0]);
    std::string dir;
    try {
        dir = vz::make_temp_dir("vz-swim-hist");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    const Long64_t chunk = (n_entries + n_workers - 1) / n_workers;
    const std::vector<std::string> base = base_worker_argv(exe, args);
    std::vector<std::string> parts, counts;
    std::vector<std::vector<std::string>> argvs;
    for (int w = 0; w < n_workers; ++w) {
        const Long64_t begin = std::min<Long64_t>(static_cast<Long64_t>(w) * chunk, n_entries);
        const Long64_t end = std::min<Long64_t>(begin + chunk, n_entries);
        const std::string part = fmt::format("{}/part-{:03}.root", dir, w);
        const std::string cnt = fmt::format("{}/part-{:03}.counts", dir, w);
        parts.push_back(part);
        counts.push_back(cnt);
        std::vector<std::string> av = base;
        av.insert(av.end(), {"--output", part, "--emit-counts", cnt, "--entry-begin",
                             fmt::format("{}", begin), "--entry-end", fmt::format("{}", end)});
        for (const auto& p : args.inputs) av.push_back(p);
        argvs.push_back(std::move(av));
    }

    fmt::print("vz-swim-hist: {} worker process(es) over {}{} particles -> {}\n", n_workers,
               n_entries < total_entries ? fmt::format("{} of ", n_entries) : std::string(),
               total_entries, args.output);
    fmt::print("swim: from DC region {}, beam offset ({}, {}) cm + per-event raster\n",
               args.dc_region, args.beam_x, args.beam_y);

    if (!vz::run_workers(argvs)) {
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

    swim_counts total;
    try {
        for (const auto& c : counts) total.add_from_file(c);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "warning: could not read worker counts: %s\n", e.what());
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    print_summary(args, total, args.output);
    return 0;
}

}  // namespace

auto main(int argc, char** argv) -> int {
    // Keep histograms out of the (non-thread-safe) global directory. No
    // EnableThreadSafety: workers are single-threaded.
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
    const bool worker_mode = !args.emit_counts.empty();
    // Resolve the field config from the input file's run metadata (RUN::config +
    // CCDB, written by hipo2root) for anything not set on the CLI. Workers inherit
    // the resolved values from the supervisor's argv, so skip it there.
    if (!worker_mode) apply_run_meta(args, p);

    const Long64_t total_entries = probe_entries(args);
    if (total_entries < 0) return 1;
    Long64_t n_entries = total_entries;
    if (args.max_entries >= 0 && args.max_entries < n_entries)
        n_entries = static_cast<Long64_t>(args.max_entries);  // process only the first N

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int want = args.jobs > 0 ? args.jobs : static_cast<int>(hw);
    const int n_workers = static_cast<int>(std::min<Long64_t>(want, std::max<Long64_t>(1, n_entries)));

    // Ensure the output directory (default output/cpp/) exists before any write.
    vz::ensure_parent_dir(args.output);

    // A worker (explicit entry range) or any run resolving to a single shard
    // processes in-process; otherwise supervise N re-exec'd workers.
    if (worker_mode) {
        const Long64_t begin = args.entry_begin >= 0 ? static_cast<Long64_t>(args.entry_begin) : 0;
        const Long64_t end = args.entry_end >= 0 ? static_cast<Long64_t>(args.entry_end) : n_entries;
        return run_worker(args, begin, end, /*worker_mode=*/true);
    }
    if (n_workers <= 1) return run_worker(args, 0, n_entries, /*worker_mode=*/false);
    return run_supervisor(args, argv, n_entries, total_entries, n_workers);
}
