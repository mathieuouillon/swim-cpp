// Electron v_z in (p, theta) bins, with a DC-R1 -> beamline swim.
//
// Reads the flat `particles` TTree written by hipo2root, selects Forward-
// Detector electrons (pid == 11), swims their DC trajectory state back to the
// beamline through the CLAS12 field (same swimmer and field configuration as
// swim-analysis), and histograms per (p, theta) cell:
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
// Region 1) and targets the DOCA to the beam position xB = --beam-x +
// per-event RASTER::position x (same for y), as DC tracking's SwimToBeamLine
// does — not the z-axis.
//
// The entry loop is parallel: the tree is split into N contiguous entry
// ranges, each worker reads through its own TChain and fills its own
// histogram grid (merged on the main thread afterwards), so no ROOT object
// is shared between threads. The shared field map is read-only.
//
// Usage:
//   vz-swim-hist <input.root>... [--output FILE] [--threads N] [--max-entries N]
//                [--dc-region 1|3] [--beam-x X] [--beam-y Y]
//                [--torus PATH] [--solenoid PATH]
//                [--torus-scale X] [--solenoid-scale X] [--solenoid-z-shift X]
//                [--torus-z-shift X] [--torus-x-shift X] [--torus-y-shift X]
//                [--vz-cot-coeff C0] [--vz-cot-p-coeff C1]
//                   (vz -> vz + (C0 + C1/p)*cot(theta) theta-walk correction)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include "TChain.h"
#include "TFile.h"
#include "TH1D.h"
#include "TROOT.h"

#include "constants.hpp"
#include "field.hpp"
#include "progresstracker.hpp"
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
inline std::size_t sector_index(double phi_deg) {
    double a = std::fmod(phi_deg + 30.0, 360.0);
    if (a < 0.0) a += 360.0;
    return static_cast<std::size_t>(a / 60.0) % N_SEC;
}

// v_z axis: the RG-D dual-target region (as in the reference electron-vz
// figure); dvz axis from constants.
constexpr int VZ_HIST_BINS = 200;
constexpr double VZ_HIST_MIN = -13.0;
constexpr double VZ_HIST_MAX = 0.0;

struct Args {
    std::vector<std::string> inputs;
    std::string output = "vz_bins.root";
    int threads = 0;        // 0 = auto-detect hardware concurrency
    long long max_entries = -1;  // -1 = all tree entries (particles)
    bool quiet = false;
    // Swim configuration, mirroring coatjava's vertex procedure.
    int dc_region = 3;    // start state: DC region 3 (coatjava) or 1
    double beam_x = 0.0;  // CCDB /geometry/beam/position x_offset [cm]
    double beam_y = 0.0;  // CCDB /geometry/beam/position y_offset [cm]
    // vz theta-walk correction: the swum (and rec) vz has a bias ~ a(p)*cot(theta)
    // (same for both targets) with a MOMENTUM-DEPENDENT coefficient
    // a(p) = c0 + c1/p:  c0 = geometric/alignment floor, c1/p = a 1/p field term
    // (dominant at low p). vz -> vz + (c0 + c1/p)*cot(theta) flattens all p.
    // Run 18614: c0 ~ 0.10, c1 ~ 0.52. Both default 0 = off.
    double vz_cot_coeff = 0.0;    // c0 [cm]
    double vz_cot_p_coeff = 0.0;  // c1 [cm*GeV]
    // Field configuration. NOTE the scale signs are in OUR FieldMap convention
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

enum class ParseResult { Ok, Help, Error };

void usage() {
    std::fprintf(stderr,
                 "usage: vz-swim-hist <input.root>... [--output FILE] [--threads N] "
                 "[--max-entries N] [--dc-region 1|3] [--beam-x X] [--beam-y Y] "
                 "[--torus PATH] [--solenoid PATH] [--torus-scale X] [--solenoid-scale X] "
                 "[--solenoid-z-shift X] [--torus-z-shift X] [--torus-x-shift X] "
                 "[--torus-y-shift X] [--dc-x-shift X] [--dc-y-shift X] [--dc-z-shift X] "
                 "[--vz-cot-coeff C0] [--vz-cot-p-coeff C1] [--quiet]\n");
}

ParseResult parse_args(int argc, char** argv, Args& a, std::string& err) {
    auto next = [&](int& i) -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" || arg == "-o") {
            auto v = next(i);
            if (!v) { err = "--output needs a value"; return ParseResult::Error; }
            a.output = v;
        } else if (arg == "--threads" || arg == "-t" || arg == "-j") {
            auto v = next(i);
            if (!v) { err = "--threads needs a value"; return ParseResult::Error; }
            try { a.threads = std::stoi(v); } catch (...) { err = "--threads must be an integer"; return ParseResult::Error; }
        } else if (arg == "--dc-region") {
            auto v = next(i);
            if (!v) { err = "--dc-region needs a value"; return ParseResult::Error; }
            try { a.dc_region = std::stoi(v); } catch (...) { err = "--dc-region must be 1 or 3"; return ParseResult::Error; }
            if (a.dc_region != 1 && a.dc_region != 3) { err = "--dc-region must be 1 or 3"; return ParseResult::Error; }
        } else if (arg == "--beam-x") {
            auto v = next(i);
            if (!v) { err = "--beam-x needs a value"; return ParseResult::Error; }
            try { a.beam_x = std::stod(v); } catch (...) { err = "--beam-x must be a number"; return ParseResult::Error; }
        } else if (arg == "--beam-y") {
            auto v = next(i);
            if (!v) { err = "--beam-y needs a value"; return ParseResult::Error; }
            try { a.beam_y = std::stod(v); } catch (...) { err = "--beam-y must be a number"; return ParseResult::Error; }
        } else if (arg == "--torus") {
            auto v = next(i);
            if (!v) { err = "--torus needs a path"; return ParseResult::Error; }
            a.torus = v;
        } else if (arg == "--solenoid") {
            auto v = next(i);
            if (!v) { err = "--solenoid needs a path"; return ParseResult::Error; }
            a.solenoid = v;
        } else if (arg == "--torus-scale") {
            auto v = next(i);
            if (!v) { err = "--torus-scale needs a value"; return ParseResult::Error; }
            try { a.torus_scale = std::stod(v); } catch (...) { err = "--torus-scale must be a number"; return ParseResult::Error; }
        } else if (arg == "--solenoid-scale") {
            auto v = next(i);
            if (!v) { err = "--solenoid-scale needs a value"; return ParseResult::Error; }
            try { a.solenoid_scale = std::stod(v); } catch (...) { err = "--solenoid-scale must be a number"; return ParseResult::Error; }
        } else if (arg == "--solenoid-z-shift") {
            auto v = next(i);
            if (!v) { err = "--solenoid-z-shift needs a value"; return ParseResult::Error; }
            try { a.solenoid_z_shift = std::stod(v); } catch (...) { err = "--solenoid-z-shift must be a number"; return ParseResult::Error; }
        } else if (arg == "--torus-z-shift") {
            auto v = next(i);
            if (!v) { err = "--torus-z-shift needs a value"; return ParseResult::Error; }
            try { a.torus_z_shift = std::stod(v); } catch (...) { err = "--torus-z-shift must be a number"; return ParseResult::Error; }
        } else if (arg == "--torus-x-shift") {
            auto v = next(i);
            if (!v) { err = "--torus-x-shift needs a value"; return ParseResult::Error; }
            try { a.torus_x_shift = std::stod(v); } catch (...) { err = "--torus-x-shift must be a number"; return ParseResult::Error; }
        } else if (arg == "--torus-y-shift") {
            auto v = next(i);
            if (!v) { err = "--torus-y-shift needs a value"; return ParseResult::Error; }
            try { a.torus_y_shift = std::stod(v); } catch (...) { err = "--torus-y-shift must be a number"; return ParseResult::Error; }
        } else if (arg == "--dc-x-shift") {
            auto v = next(i);
            if (!v) { err = "--dc-x-shift needs a value"; return ParseResult::Error; }
            try { a.dc_x_shift = std::stod(v); } catch (...) { err = "--dc-x-shift must be a number"; return ParseResult::Error; }
        } else if (arg == "--dc-y-shift") {
            auto v = next(i);
            if (!v) { err = "--dc-y-shift needs a value"; return ParseResult::Error; }
            try { a.dc_y_shift = std::stod(v); } catch (...) { err = "--dc-y-shift must be a number"; return ParseResult::Error; }
        } else if (arg == "--dc-z-shift") {
            auto v = next(i);
            if (!v) { err = "--dc-z-shift needs a value"; return ParseResult::Error; }
            try { a.dc_z_shift = std::stod(v); } catch (...) { err = "--dc-z-shift must be a number"; return ParseResult::Error; }
        } else if (arg == "--max-entries" || arg == "--max-events" || arg == "-n") {
            auto v = next(i);
            if (!v) { err = "--max-entries needs a value"; return ParseResult::Error; }
            try { a.max_entries = std::stoll(v); } catch (...) { err = "--max-entries must be an integer"; return ParseResult::Error; }
        } else if (arg == "--vz-cot-coeff" || arg == "--vz-theta-correction") {
            auto v = next(i);
            if (!v) { err = "--vz-cot-coeff needs a value"; return ParseResult::Error; }
            try { a.vz_cot_coeff = std::stod(v); } catch (...) { err = "--vz-cot-coeff must be a number"; return ParseResult::Error; }
        } else if (arg == "--vz-cot-p-coeff") {
            auto v = next(i);
            if (!v) { err = "--vz-cot-p-coeff needs a value"; return ParseResult::Error; }
            try { a.vz_cot_p_coeff = std::stod(v); } catch (...) { err = "--vz-cot-p-coeff must be a number"; return ParseResult::Error; }
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
        err = "missing <input.root> (the particles tree from hipo2root)";
        return ParseResult::Error;
    }
    return ParseResult::Ok;
}

// `pAA_BB_thCC_DD` name fragment for cell (pb, tb), edges zero-padded.
std::string cell_tag(std::size_t pb, std::size_t tb) {
    const int plo = static_cast<int>(std::lround(P_MIN + pb * P_WIDTH));
    const int phi = static_cast<int>(std::lround(P_MIN + (pb + 1) * P_WIDTH));
    const int tlo = static_cast<int>(std::lround(TH_MIN + tb * TH_WIDTH));
    const int thi = static_cast<int>(std::lround(TH_MIN + (tb + 1) * TH_WIDTH));
    return fmt::format("p{:02d}_{:02d}_th{:02d}_{:02d}", plo, phi, tlo, thi);
}

// `sS_pAA_BB_thCC_DD` name fragment for a (sector, p, theta) cell.
std::string sector_tag(std::size_t s, std::size_t pb, std::size_t tb) {
    return fmt::format("s{}_{}", s + 1, cell_tag(pb, tb));
}

// The (p x theta) grid plus a (sector x theta) grid (p-integrated over the
// selection), three histograms per cell. Each worker owns one (filled without
// locking); the workers' grids are merged into the main one at the end.
struct CellGrid {
    std::array<std::array<std::unique_ptr<TH1D>, N_TH>, N_P> rec, swum, dvz;
    // (sector x p x theta): lets us test whether the per-sector vz modulation
    // is momentum-dependent (torus-coil field ~ 1/p) or flat (DC alignment).
    using SecGrid =
        std::array<std::array<std::array<std::unique_ptr<TH1D>, N_TH>, N_P>, N_SEC>;
    SecGrid sec_rec, sec_swum, sec_dvz;

    CellGrid() {
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

    void merge(const CellGrid& other) {
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

// One worker: read entries [begin, end) through a private TChain, swim, and
// fill the private CellGrid. Shares only the read-only field map and the
// (thread-safe) progress tracker.
struct SwimWorker {
    const Args* args = nullptr;
    const vz::Field* field = nullptr;
    Long64_t begin = 0, end = 0;
    ProgressTracker* progress = nullptr;  // nullptr = no progress display

    CellGrid grid;
    // Cut flow.
    long long n_electron_fd = 0;  // FD electrons
    long long n_in_grid = 0;      // ... inside the (p, theta) grid
    long long n_with_dc = 0;      // ... with the chosen DC trajectory state
    long long n_swim_ok = 0;      // ... whose swim converged (-> filled)
    double sum_swum_vz = 0.0;     // sum of converged swum vz (config sanity check)

    void operator()() {
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

        constexpr Long64_t PROGRESS_CHUNK = 8192;  // amortize the atomic add
        Long64_t since_report = 0;
        for (Long64_t i = begin; i < end; ++i) {
            tree.GetEntry(i);
            if (progress && ++since_report == PROGRESS_CHUNK) {
                progress->add(static_cast<std::size_t>(PROGRESS_CHUNK));
                since_report = 0;
            }
            if (pid != 11) continue;
            if (!vz::is_forward(status)) continue;
            ++n_electron_fd;

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
            double xB = args->beam_x, yB = args->beam_y;
            if (std::isfinite(raster_x) && std::isfinite(raster_y)) {
                xB += raster_x;
                yB += raster_y;
            }

            // DC alignment probe: rigidly translate the start position.
            const std::array<double, 3> pos = {dc_x + args->dc_x_shift, dc_y + args->dc_y_shift,
                                               dc_z + args->dc_z_shift};
            const std::array<double, 3> mom = {p * dc_cx, p * dc_cy, p * dc_cz};
            const vz::SwimResult sw =
                vz::swim_back_to_beamline(*field, pos, mom, /*q=*/-1.0, xB, yB);
            if (sw.status != vz::SwimStatus::Converged || sw.doca_rho >= vz::SWUM_MAX_DOCA_RHO)
                continue;
            ++n_swim_ok;

            // vz theta-walk correction: vz -> vz + (c0 + c1/p)*cot(theta),
            // applied to both rec and swum (so dvz is unchanged). c0 is the
            // geometric floor, c1/p the 1/p field term. cot(theta) = pz / pT.
            const double pT = std::sqrt(double(px) * px + double(py) * py);
            const double corr =
                pT > 0.0 ? (args->vz_cot_coeff + args->vz_cot_p_coeff / p) * (double(pz) / pT)
                         : 0.0;
            const double vz_rec_c = vz_rec + corr;
            const double vz_swum_c = sw.vz + corr;
            sum_swum_vz += vz_swum_c;

            grid.rec[pb][tb]->Fill(vz_rec_c);
            grid.swum[pb][tb]->Fill(vz_swum_c);
            grid.dvz[pb][tb]->Fill(vz_swum_c - vz_rec_c);

            // Same fills, binned by sector (from the momentum azimuth).
            const double phi = std::atan2(double(py), double(px)) * vz::RAD2DEG;
            const std::size_t sec = sector_index(phi);
            grid.sec_rec[sec][pb][tb]->Fill(vz_rec_c);
            grid.sec_swum[sec][pb][tb]->Fill(vz_swum_c);
            grid.sec_dvz[sec][pb][tb]->Fill(vz_swum_c - vz_rec_c);
        }
        if (progress && since_report > 0) progress->add(static_cast<std::size_t>(since_report));
    }
};

}  // namespace

int main(int argc, char** argv) {
    // ROOT global state: enable internal locks and keep histograms out of the
    // (non-thread-safe) global directory. MUST precede any threads/histograms.
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

    // Field maps, same configuration as swim-analysis. Loaded once, shared
    // read-only across the workers.
    fmt::print("loading field maps: torus={} (scale {}, shift x={} y={} z={} cm), "
               "solenoid={} (scale {}, z-shift {} cm)\n",
               args.torus, args.torus_scale, args.torus_x_shift, args.torus_y_shift,
               args.torus_z_shift, args.solenoid, args.solenoid_scale, args.solenoid_z_shift);
    if (args.dc_x_shift != 0.0 || args.dc_y_shift != 0.0 || args.dc_z_shift != 0.0)
        fmt::print("DC start-state shift: x={} y={} z={} cm\n", args.dc_x_shift, args.dc_y_shift,
                   args.dc_z_shift);
    std::unique_ptr<vz::CompositeField> field_ptr;
    try {
        field_ptr = std::make_unique<vz::CompositeField>(vz::CompositeField::load(
            args.torus, args.torus_scale, args.solenoid, args.solenoid_scale,
            args.solenoid_z_shift, args.torus_z_shift, args.torus_x_shift, args.torus_y_shift));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: could not load field maps: %s\n", e.what());
        return 1;
    }

    // Probe the inputs once on the main thread for validation + entry count.
    Long64_t n_entries = 0;
    {
        TChain tree("particles");
        for (const auto& p : args.inputs) {
            if (tree.Add(p.c_str(), /*nentries=*/-1) == 0) {
                std::fprintf(stderr, "error: no 'particles' tree in '%s'\n", p.c_str());
                return 1;
            }
        }
        n_entries = tree.GetEntries();
    }

    const Long64_t total_entries = n_entries;
    if (args.max_entries >= 0 && args.max_entries < n_entries)
        n_entries = static_cast<Long64_t>(args.max_entries);  // process only the first N

    const int n_threads = args.threads > 0
                              ? args.threads
                              : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    if (n_entries < total_entries)
        fmt::print("inputs: {} file(s), {} of {} particles ({} threads)\n", args.inputs.size(),
                   n_entries, total_entries, n_threads);
    else
        fmt::print("inputs: {} file(s), {} particles ({} threads)\n", args.inputs.size(), n_entries,
                   n_threads);
    fmt::print("swim: from DC region {}, beam offset ({}, {}) cm + per-event raster"
               ", vz correction (c0={} + c1={}/p)*cot(theta)\n",
               args.dc_region, args.beam_x, args.beam_y, args.vz_cot_coeff, args.vz_cot_p_coeff);

    // Progress bar over tree entries (instant rate + ETA from hipo4).
    std::unique_ptr<ProgressTracker> progress;
    if (!args.quiet && n_entries > 0) {
        ProgressTracker::Config config;
        config.label = "Swimming";
        config.show_eta = true;
        config.show_rate = true;
        progress = std::make_unique<ProgressTracker>(static_cast<std::size_t>(n_entries), config);
        progress->start();
    }

    // Contiguous entry ranges, one worker (private TChain + CellGrid) each.
    std::vector<SwimWorker> workers(static_cast<std::size_t>(n_threads));
    const Long64_t chunk = (n_entries + n_threads - 1) / n_threads;
    std::vector<std::thread> pool;
    pool.reserve(workers.size());
    for (std::size_t w = 0; w < workers.size(); ++w) {
        workers[w].args = &args;
        workers[w].field = field_ptr.get();
        workers[w].progress = progress.get();
        workers[w].begin = std::min<Long64_t>(static_cast<Long64_t>(w) * chunk, n_entries);
        workers[w].end = std::min<Long64_t>(workers[w].begin + chunk, n_entries);
        pool.emplace_back(std::ref(workers[w]));
    }
    for (auto& t : pool) t.join();
    if (progress) progress->finish();

    // Merge the workers' grids and cut flows into the first one.
    CellGrid& grid = workers[0].grid;
    long long n_electron_fd = 0, n_in_grid = 0, n_with_dc = 0, n_swim_ok = 0;
    double sum_swum_vz = 0.0;
    for (std::size_t w = 0; w < workers.size(); ++w) {
        if (w > 0) grid.merge(workers[w].grid);
        n_electron_fd += workers[w].n_electron_fd;
        n_in_grid += workers[w].n_in_grid;
        n_with_dc += workers[w].n_with_dc;
        n_swim_ok += workers[w].n_swim_ok;
        sum_swum_vz += workers[w].sum_swum_vz;
    }

    TFile fout(args.output.c_str(), "RECREATE");
    if (fout.IsZombie()) {
        std::fprintf(stderr, "error: could not create '%s'\n", args.output.c_str());
        return 1;
    }
    for (std::size_t pb = 0; pb < N_P; ++pb) {
        for (std::size_t tb = 0; tb < N_TH; ++tb) {
            grid.rec[pb][tb]->Write();
            grid.swum[pb][tb]->Write();
            grid.dvz[pb][tb]->Write();
        }
    }
    for (std::size_t s = 0; s < N_SEC; ++s) {
        for (std::size_t pb = 0; pb < N_P; ++pb) {
            for (std::size_t tb = 0; tb < N_TH; ++tb) {
                grid.sec_rec[s][pb][tb]->Write();
                grid.sec_swum[s][pb][tb]->Write();
                grid.sec_dvz[s][pb][tb]->Write();
            }
        }
    }
    fout.Close();

    fmt::print("----------------------------------------\n");
    fmt::print("particles          : {}\n", n_entries);
    fmt::print("FD electrons       : {}\n", n_electron_fd);
    fmt::print("  in (p,theta) grid: {}\n", n_in_grid);
    fmt::print("  with DC-R{} state : {}\n", args.dc_region, n_with_dc);
    fmt::print("  swim converged   : {}  (filled)\n", n_swim_ok);
    if (n_swim_ok > 0) {
        const double mean_vz = sum_swum_vz / static_cast<double>(n_swim_ok);
        fmt::print("  mean swum vz     : {:.2f} cm{}\n", mean_vz,
                   (mean_vz < VZ_HIST_MIN || mean_vz > VZ_HIST_MAX)
                       ? "  ** OUTSIDE the histogram range -- check field scales/shifts **"
                       : "");
    }
    fmt::print("histograms written to {}: {} ((p,theta) {} + (sector,p,theta) {}) x 3\n",
               args.output, (N_P * N_TH + N_SEC * N_P * N_TH) * 3, N_P * N_TH,
               N_SEC * N_P * N_TH);
    return 0;
}
