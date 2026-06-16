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

enum class swim_status { converged, max_path, no_minimum };

struct swim_result {
    double vz;        // z at the closest approach (= swum vz); NaN on failure
    double doca_rho;  // sqrt(x^2 + y^2) at the closest approach
    double path;      // arc length s at the stop (diagnostic, not histogrammed)
    swim_status status;
};

/// Swim a charged track BACKWARD from its DC state to the beamline: the first
/// closest approach to the line x=x_b, y=y_b (the beam position in cm; default
/// the z-axis — coatjava's SwimToBeamLine uses the CCDB beam offset plus the
/// per-event raster) reached inside the target region z < 2 m. `pos_cm`/
/// `mom_gev` are the position and momentum at the DC; `q` is the swim charge.
/// The direction is reversed here (u = -p_hat); to retrace the incoming path
/// the CALLER passes the reversed charge together with the physical (CCDB /
/// cnuphys) field polarity, as coatjava does (vz-swim-hist: -charge_of(pid),
/// torus +1, solenoid -1). Returns the z at the closest approach; doca_rho is
/// the distance to (x_b, y_b) there.
auto swim_back_to_beamline(const magnetic_field& field, const std::array<double, 3>& pos_cm,
                           const std::array<double, 3>& mom_gev, double q,
                           double x_b = 0.0, double y_b = 0.0) -> swim_result;

}  // namespace vz
