// Binning, species, region, and correlation-variable definitions for the
// v_z / PID diagnostic analysis (port of the Rust constants.rs, itself a port
// of the C++ RHist version).
#pragma once

#include <array>
#include <cstddef>
#include <numbers>
#include <optional>
#include <string_view>

namespace vz {

inline constexpr double RAD2DEG = 180.0 / std::numbers::pi;

// Charged-pion mass (GeV), for the expected-beta / delta-beta computation.
inline constexpr double PION_MASS = 0.139570;

// Speed of light in cm/ns, for time-of-flight residuals.
inline constexpr double C_CM_PER_NS = 29.9792458;

// CLAS12 detector IDs (org.jlab.detector.DetectorType ordinals) and layers,
// used to pick the right rows from the per-detector response banks.
inline constexpr int DET_DC = 6;
inline constexpr int DET_ECAL = 7;
inline constexpr int DET_FTOF = 12;
inline constexpr int DET_HTCC = 15;
inline constexpr std::array<int, 3> DC_LAYERS = {6, 18, 36};  // DC regions 1/2/3
inline constexpr int FTOF_PREF_LAYER = 2;                     // FTOF panel 1B

// Momentum binning: 12 bins of 0.5 GeV over [0, 6) GeV.
inline constexpr std::size_t N_MOM_BINS = 12;
inline constexpr double MOM_MIN = 0.0;
inline constexpr double MOM_MAX = 6.0;
inline constexpr double MOM_WIDTH = (MOM_MAX - MOM_MIN) / static_cast<double>(N_MOM_BINS);

// v_z axis (CLAS12 vertex z is in cm).
inline constexpr int VZ_BINS = 200;
inline constexpr double VZ_MIN = -20.0;
inline constexpr double VZ_MAX = 0.0;

// Delta v_z = reco_vz - truth_vz residual axis (cm), centered on 0.
inline constexpr int DVZ_BINS = 200;
inline constexpr double DVZ_MIN = -5.0;
inline constexpr double DVZ_MAX = 5.0;

// Polar angle theta axis (deg) for the v_z-vs-theta 2D diagnostic.
inline constexpr int THETA_BINS = 120;
inline constexpr double THETA_MIN = 0.0;
inline constexpr double THETA_MAX = 60.0;

// True if a REC::Particle status word denotes a Forward-Detector track:
// |status| in [2000, 4000). This analysis keeps Forward-Detector tracks only.
inline constexpr bool is_forward(int status) {
    int a = status < 0 ? -status : status;
    return a >= 2000 && a < 4000;
}

// Particle-species index used for both the MC-truth tag and the reconstructed
// pid (pions then muons): indexes the truth x reco confusion-matrix grid.
inline constexpr std::size_t N_SPECIES = 4;
inline constexpr std::array<std::string_view, N_SPECIES> SPECIES_TAG = {"pip", "pim", "mup", "mum"};
inline constexpr std::array<std::string_view, N_SPECIES> SPECIES_NAME = {"pi+", "pi-", "mu+", "mu-"};
inline constexpr std::array<std::string_view, N_SPECIES> SPECIES_TEX = {
    "#pi^{+}", "#pi^{-}", "#mu^{+}", "#mu^{-}"};

// Species index for a pid (pi+ 211, pi- -211, mu+ -13, mu- 13), else nullopt.
inline constexpr std::optional<std::size_t> species_index(int pid) {
    switch (pid) {
        case 211: return 0;   // pi+
        case -211: return 1;  // pi-
        case -13: return 2;   // mu+
        case 13: return 3;    // mu-
        default: return std::nullopt;
    }
}

// A track variable: defines a 1-D histogram axis.
struct Var {
    std::string_view tag;
    int nbins;
    double lo;
    double hi;
};

// 1-D distributions filled for reco pions in the PT_BANDS momentum bands.
// Order MUST match the `vals` array in Analysis::fill_event.
inline constexpr std::array<Var, 28> TRACK_VARS = {{
    {"p", 120, 0.0, 6.0},
    {"pt", 120, 0.0, 6.0},
    {"theta", 120, 0.0, 60.0},
    {"phi", 180, -180.0, 180.0},
    {"vx", 100, -2.0, 2.0},
    {"vy", 100, -2.0, 2.0},
    {"vz", VZ_BINS, VZ_MIN, VZ_MAX},
    {"dbeta", 200, -0.01, 0.01},
    {"chi2pid", 200, -5.0, 5.0},
    {"vt", 200, -2.0, 2.0},
    {"trkchi2", 200, 0.0, 100.0},
    {"trkndf", 60, 0.0, 60.0},
    {"trkredchi2", 100, 0.0, 20.0},
    {"edge_dc1", 120, -10.0, 50.0},
    {"edge_dc2", 120, -10.0, 50.0},
    {"edge_dc3", 120, -10.0, 50.0},
    {"dc_path", 100, 0.0, 700.0},
    {"sigp", 100, 0.0, 0.05},
    {"sigtx", 100, 0.0, 0.003},
    {"sigty", 100, 0.0, 0.003},
    {"sigtheta", 100, 0.0, 5.0},
    {"sigphi", 100, 0.0, 20.0},
    {"ftof_chi2", 100, 0.0, 1.0},
    {"ftof_dt", 200, -1.0, 1.0},
    {"ftof_dedx", 100, 0.0, 20.0},
    {"htcc_nphe", 50, 0.0, 50.0},
    {"ecal_e", 100, 0.0, 2.0},
    {"ecal_sf", 100, 0.0, 0.4},
}};

inline constexpr std::size_t N_TRACK_VARS = TRACK_VARS.size();

// Resolution variables histogrammed per fine momentum bin across the full
// truth x reco grid (like v_z). Order MUST match res_vals in fill_event.
inline constexpr std::array<Var, 3> RES_VARS = {{
    {"sigtx", 100, 0.0, 0.003},
    {"sigty", 100, 0.0, 0.003},
    {"sigtheta", 100, 0.0, 5.0},  // mrad
}};

inline constexpr std::size_t N_RES_VARS = RES_VARS.size();

// Track-variable distributions are filled for pions (pi+, pi- = the first two
// entries of the species grid) only.
inline constexpr std::size_t N_PION_SPECIES = 2;

// Swum-vz residual axis (cm) and the closest-approach quality cut (cm).
inline constexpr int SWUM_DVZ_BINS = 200;
inline constexpr double SWUM_DVZ_MIN = -10.0;
inline constexpr double SWUM_DVZ_MAX = 10.0;
inline constexpr double SWUM_MAX_DOCA_RHO = 5.0;

// A momentum band for the track-variable distributions. Low- and high-p pions
// are histogrammed separately so their shapes can be overlaid for comparison.
struct PtBand {
    std::string_view tag;
    double lo;
    double hi;
};

inline constexpr std::array<PtBand, 2> PT_BANDS = {{
    {"lo", 0.0, 1.0},
    {"hi", 3.0, 6.0},
}};

inline constexpr std::size_t N_PT_BANDS = PT_BANDS.size();

// Index of the momentum band a track falls in, or nullopt if between bands.
inline constexpr std::optional<std::size_t> pt_band(double p) {
    for (std::size_t i = 0; i < PT_BANDS.size(); ++i) {
        if (p >= PT_BANDS[i].lo && p < PT_BANDS[i].hi) return i;
    }
    return std::nullopt;
}

}  // namespace vz
