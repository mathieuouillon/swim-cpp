// Per-detector response-bank readers over a `hipo::bank_view` (an empty view —
// rows()==0 — means the bank is absent in this run/event, the C++ analog of
// Rust's `Option<&Bank>` being None). Banks may have several rows per particle,
// so rows are selected by `pindex` plus an optional `detector` and/or `layer`.
// Port of the helpers in analysis.rs.
//
// Column reads use the new hipo4 convert-on-read getter bank_view::get<int>
// (promotes Byte/Short/Int) and get<double> (handles Float and Double) —
// matching the Rust RowView i32()/f64() getters that convert across the stored
// wire width.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>

#include "constants.hpp"
#include "hipo4/hipo.hpp"

namespace vz {

inline constexpr double DNAN = std::numeric_limits<double>::quiet_NaN();

/// row index of the first row matching `pindex` (and `det`/`layer` when given).
inline auto match_row(hipo::bank_view bank, int pindex, std::optional<int> det,
                      std::optional<int> layer) -> std::optional<int> {
    if (!bank) return std::nullopt;
    const int n = static_cast<int>(bank.rows());
    for (int k = 0; k < n; ++k) {
        if (bank.get<int>("pindex", k) != pindex) continue;
        if (det && bank.get<int>("detector", k) != *det) continue;
        if (layer && bank.get<int>("layer", k) != *layer) continue;
        return k;
    }
    return std::nullopt;
}

/// Float column from the first row matching `pindex` (+ optional `det`).
inline auto first_f64(hipo::bank_view bank, int pindex, std::optional<int> det,
                      const char* col) -> std::optional<double> {
    if (!bank) return std::nullopt;
    auto k = match_row(bank, pindex, det, std::nullopt);
    if (!k) return std::nullopt;
    return bank.get<double>(col, *k);
}

/// Float column at a specific `det` + `layer` for `pindex`.
inline auto layer_f64(hipo::bank_view bank, int pindex, int det, int layer,
                      const char* col) -> std::optional<double> {
    if (!bank) return std::nullopt;
    auto k = match_row(bank, pindex, det, layer);
    if (!k) return std::nullopt;
    return bank.get<double>(col, *k);
}

/// Sum of a float column over all rows matching `pindex` (+ optional `det`);
/// nullopt if no row matched (distinguishes "absent" from "zero deposit").
inline auto sum_f64(hipo::bank_view bank, int pindex, std::optional<int> det,
                    const char* col) -> std::optional<double> {
    if (!bank) return std::nullopt;
    double sum = 0.0;
    bool found = false;
    const int n = static_cast<int>(bank.rows());
    for (int k = 0; k < n; ++k) {
        if (bank.get<int>("pindex", k) != pindex) continue;
        if (det && bank.get<int>("detector", k) != *det) continue;
        sum += bank.get<double>(col, k);
        found = true;
    }
    if (!found) return std::nullopt;
    return sum;
}

/// Covariance-derived resolutions from REC::CovMat (5x5 DC Kalman state x,y,tx,
/// ty,q/p at the last superlayer): (sigma_p/p, sigma(tx), sigma(ty),
/// sigma_theta [mrad], sigma_phi [mrad]). sigma_theta/phi propagate the slope
/// covariance through the (theta,phi) Jacobian at the lab direction.
struct cov_res {
    double sigp;
    double sigtx;
    double sigty;
    double sigtheta;
    double sigphi;
};

inline auto covariance_resolutions(hipo::bank_view cov, int pindex, double p, double px,
                                   double py, double pz) -> cov_res {
    const double nan = DNAN;
    auto c33o = first_f64(cov, pindex, std::nullopt, "C33");
    auto c44o = first_f64(cov, pindex, std::nullopt, "C44");
    auto c34o = first_f64(cov, pindex, std::nullopt, "C34");
    auto c55o = first_f64(cov, pindex, std::nullopt, "C55");

    double sigp = c55o ? p * std::sqrt(std::max(0.0, *c55o)) : nan;
    double sigtx = c33o ? std::sqrt(std::max(0.0, *c33o)) : nan;
    double sigty = c44o ? std::sqrt(std::max(0.0, *c44o)) : nan;
    double sigtheta = nan;
    double sigphi = nan;

    if (c33o && c44o && c34o && pz > 0.0) {
        const double c33 = *c33o, c44 = *c44o, c34 = *c34o;
        const double a = px / pz;
        const double b = py / pz;
        const double rho2 = a * a + b * b;
        if (rho2 > 1e-9) {
            const double rho = std::sqrt(rho2);
            // theta = atan(rho), phi = atan2(b, a)
            const double dth_da = a / (rho * (1.0 + rho2));
            const double dth_db = b / (rho * (1.0 + rho2));
            const double dph_da = -b / rho2;
            const double dph_db = a / rho2;
            const double var_th =
                dth_da * dth_da * c33 + dth_db * dth_db * c44 + 2.0 * dth_da * dth_db * c34;
            const double var_ph =
                dph_da * dph_da * c33 + dph_db * dph_db * c44 + 2.0 * dph_da * dph_db * c34;
            sigtheta = std::sqrt(std::max(0.0, var_th)) * 1000.0;  // mrad
            sigphi = std::sqrt(std::max(0.0, var_ph)) * 1000.0;    // mrad
        }
    }
    return {sigp, sigtx, sigty, sigtheta, sigphi};
}

struct pos_mom {
    std::array<double, 3> pos;  // cm
    std::array<double, 3> mom;  // GeV
};

/// True (position [cm], momentum [GeV]) of the primary `target_pid` at its first
/// DC hit (detector==DC, mpid==0, lowest avgT). MC::True positions avgX/Y/Z are
/// mm -> /10 cm, momenta MeV -> /1000 GeV.
inline auto mc_true_dc_state(hipo::bank_view mctrue, int target_pid) -> std::optional<pos_mom> {
    if (!mctrue) return std::nullopt;
    double best_t = std::numeric_limits<double>::infinity();
    std::optional<pos_mom> best;
    const int n = static_cast<int>(mctrue.rows());
    for (int k = 0; k < n; ++k) {
        if (mctrue.get<int>("detector", k) != DET_DC) continue;
        if (mctrue.get<int>("mpid", k) != 0) continue;
        if (mctrue.get<int>("pid", k) != target_pid) continue;
        const double t = mctrue.get<double>("avgT", k);
        if (t < best_t) {
            best_t = t;
            pos_mom pm;
            pm.pos = {mctrue.get<double>("avgX", k) / 10.0, mctrue.get<double>("avgY", k) / 10.0,
                      mctrue.get<double>("avgZ", k) / 10.0};
            pm.mom = {mctrue.get<double>("px", k) / 1000.0, mctrue.get<double>("py", k) / 1000.0,
                      mctrue.get<double>("pz", k) / 1000.0};
            best = pm;
        }
    }
    return best;
}

/// Reconstructed (position [cm], momentum [GeV]) at DC region 1 from REC::Traj:
/// the row with detector==DC, layer==DC_LAYERS[0]. Position is already cm; the
/// momentum is the (cx,cy,cz) cosines scaled by the reconstructed |p|.
inline auto rec_traj_dc_state(hipo::bank_view traj, int pindex, double p) -> std::optional<pos_mom> {
    if (!traj) return std::nullopt;
    auto k = match_row(traj, pindex, DET_DC, DC_LAYERS[0]);
    if (!k) return std::nullopt;
    pos_mom pm;
    pm.pos = {traj.get<double>("x", *k), traj.get<double>("y", *k), traj.get<double>("z", *k)};
    const double cx = traj.get<double>("cx", *k);
    const double cy = traj.get<double>("cy", *k);
    const double cz = traj.get<double>("cz", *k);
    pm.mom = {p * cx, p * cy, p * cz};
    return pm;
}

struct p_bin {
    double lo;
    double hi;
    int lo10;
    int hi10;
};

/// Momentum-bin edges and the `pNN_NN` label fragment (tenths of a GeV).
inline auto pbin(std::size_t b) -> p_bin {
    const double lo = MOM_MIN + static_cast<double>(b) * MOM_WIDTH;
    const double hi = lo + MOM_WIDTH;
    return {lo, hi, static_cast<int>(std::lround(lo * 10.0)), static_cast<int>(std::lround(hi * 10.0))};
}

}  // namespace vz
