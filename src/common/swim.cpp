#include "swim.hpp"

#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>
#include <cmath>
#include <limits>
#include <utility>

namespace vz {

namespace odeint = boost::numeric::odeint;

namespace {
// A stiff (high-p) track meets the beamline at its FIRST closest approach -- the
// long-validated electron behavior, capped at STIFF_MAX_PATH. A curly low-p pion
// spirals, with closest-approach minima far out (> 1 m) before the vertex, so
// below STIFF_P_GEV we keep swimming (out to CURLY_MAX_PATH) and take the
// approach NEAREST the beamline, stopping once within VERTEX_RHO of it. Electrons
// are only swum at p >= 2 GeV (their grid floor), so their results are unchanged.
constexpr double STIFF_P_GEV = 2.0;
constexpr double STIFF_MAX_PATH_CM = 800.0;
constexpr double CURLY_MAX_PATH_CM = 3000.0;
constexpr double VERTEX_RHO_CM = 3.0;
constexpr double RTOL = 1e-6;
constexpr double ATOL = 1e-6;
using state_type = std::array<double, 6>;

auto fail() -> swim_result {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, 0.0, swim_status::no_minimum};
}

// Radial velocity (x-x_b)*ux + (y-y_b)*uy w.r.t. the beam axis at (x_b, y_b);
// zero at the closest approach to that line.
inline auto radial_vel(const state_type& y, double x_b, double y_b) -> double {
    return (y[0] - x_b) * y[3] + (y[1] - y_b) * y[4];
}

// Distance to the beam axis at (x_b, y_b).
inline auto beam_rho(const state_type& y, double x_b, double y_b) -> double {
    return std::sqrt((y[0] - x_b) * (y[0] - x_b) + (y[1] - y_b) * (y[1] - y_b));
}
}  // namespace

auto swim_back_to_beamline(const magnetic_field& field, const std::array<double, 3>& pos_cm,
                           const std::array<double, 3>& mom_gev, double q, double x_b,
                           double y_b) -> swim_result {
    const double p = std::sqrt(mom_gev[0] * mom_gev[0] + mom_gev[1] * mom_gev[1] +
                               mom_gev[2] * mom_gev[2]);
    if (!(p > 1e-6)) return fail();
    const double kq = K0 * q / p;

    // dy/ds = [u, kq*(u x B)], B in kG from the field map (lab Cartesian).
    auto system = [&field, kq](const state_type& y, state_type& dy, double /*s*/) {
        const std::array<double, 3> b = field.b_kgauss(y[0], y[1], y[2]);
        dy[0] = y[3];
        dy[1] = y[4];
        dy[2] = y[5];
        dy[3] = kq * (y[4] * b[2] - y[5] * b[1]);
        dy[4] = kq * (y[5] * b[0] - y[3] * b[2]);
        dy[5] = kq * (y[3] * b[1] - y[4] * b[0]);
    };

    state_type y = {pos_cm[0],        pos_cm[1],        pos_cm[2],
                    -mom_gev[0] / p,  -mom_gev[1] / p,  -mom_gev[2] / p};

    auto stepper =
        odeint::make_dense_output(ATOL, RTOL, odeint::runge_kutta_dopri5<state_type>());
    stepper.initialize(y, 0.0, 0.1);

    const bool stiff = (p >= STIFF_P_GEV);
    const double max_path = stiff ? STIFF_MAX_PATH_CM : CURLY_MAX_PATH_CM;

    double g_prev = radial_vel(y, x_b, y_b);
    state_type y_end{};
    double best_rho = std::numeric_limits<double>::infinity();  // nearest approach (curly search)
    double best_z = 0.0, best_s = 0.0;

    try {
        while (stepper.current_time() < max_path) {
            const double t0 = stepper.current_time();
            stepper.do_step(system);  // one accepted adaptive step
            const double t1 = stepper.current_time();

            const bool capped = (t1 >= max_path);
            const double t_end = capped ? max_path : t1;
            stepper.calc_state(t_end, y_end);
            const double g_cur = radial_vel(y_end, x_b, y_b);

            if ((g_prev < 0.0) != (g_cur < 0.0)) {  // a rho extremum (closest/farthest approach)
                auto G = [&stepper, x_b, y_b](double t) {
                    state_type s;
                    stepper.calc_state(t, s);
                    return radial_vel(s, x_b, y_b);
                };
                std::uintmax_t maxit = 60;
                auto tol = [](double a, double b) { return std::fabs(b - a) < 1e-9; };
                auto br = boost::math::tools::bisect(G, t0, t_end, tol, maxit);
                const double s_star = 0.5 * (br.first + br.second);
                state_type ys;
                stepper.calc_state(s_star, ys);
                const double rho = beam_rho(ys, x_b, y_b);
                // Stiff track: the first closest approach IS the vertex -- return it
                // exactly as before (electrons unchanged). Curly track: keep the
                // nearest approach and keep swimming until genuinely at the beamline.
                if (stiff) return {ys[2], rho, s_star, swim_status::converged};
                if (rho < best_rho) {
                    best_rho = rho;
                    best_z = ys[2];
                    best_s = s_star;
                }
                if (rho < VERTEX_RHO_CM) return {best_z, best_rho, best_s, swim_status::converged};
            }
            if (capped) {
                if (stiff)
                    return {y_end[2], beam_rho(y_end, x_b, y_b), max_path, swim_status::max_path};
                break;
            }
            g_prev = g_cur;
        }
    } catch (...) {
        return fail();
    }

    if (std::isfinite(best_rho)) return {best_z, best_rho, best_s, swim_status::converged};
    return {y_end[2], beam_rho(y_end, x_b, y_b), stepper.current_time(), swim_status::max_path};
}

}  // namespace vz
