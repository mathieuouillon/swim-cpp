// Swim a charged particle through the CLAS12 field using Boost.odeint:
// adaptive-step Dormand-Prince RK5(4) with dense output, plus a scalar
// root-find to stop at the closest approach to the beamline. Port of swim.rs.
//
// State y = [x, y, z, ux, uy, uz] vs arc length s [cm]; dr/ds = u,
// du/ds = (K0*q/p) * (u x B), B in kG. K0 from 1/R[m] = 0.299792458*q*B[T]/p[GeV]
// with 1 kG = 0.1 T and 1 m = 100 cm => K0 = 0.299792458e-3 [1/(kG*cm)].
// Closest approach to the z-axis is where the radial velocity x*ux + y*uy = 0.
#pragma once

#include <array>

#include "field.hpp"

namespace vz {

inline constexpr double K0 = 0.299792458e-3;

enum class SwimStatus { Converged, MaxPath, NoMinimum };

struct SwimResult {
    double vz;        // z at the closest approach (= swum vz); NaN on failure
    double doca_rho;  // sqrt(x^2 + y^2) at the closest approach
    double path;      // arc length s at the stop (diagnostic, not histogrammed)
    SwimStatus status;
};

/// Swim a charged track BACKWARD from its DC-region-1 state to the beamline
/// (closest approach to x=0,y=0). `pos_cm`/`mom_gev` are the position and
/// momentum at DC R1; `q` is +1 (pi+) or -1 (pi-). Reversing the direction
/// (u = -p_hat) while keeping q retraces the particle's incoming path. Returns
/// the z at the closest approach.
SwimResult swim_back_to_beamline(const Field& field, const std::array<double, 3>& pos_cm,
                                 const std::array<double, 3>& mom_gev, double q);

}  // namespace vz
