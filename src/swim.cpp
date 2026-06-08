#include "swim.hpp"

#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>
#include <cmath>
#include <limits>
#include <utility>

namespace vz {

namespace odeint = boost::numeric::odeint;

namespace {
constexpr double MAX_PATH_CM = 800.0;
constexpr double RTOL = 1e-6;
constexpr double ATOL = 1e-6;
using state_type = std::array<double, 6>;

SwimResult fail() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, 0.0, SwimStatus::NoMinimum};
}

// Radial velocity x*ux + y*uy; zero at the closest approach to the z-axis.
inline double radial_vel(const state_type& y) { return y[0] * y[3] + y[1] * y[4]; }
}  // namespace

SwimResult swim_back_to_beamline(const Field& field, const std::array<double, 3>& pos_cm,
                                 const std::array<double, 3>& mom_gev, double q) {
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

    double g_prev = radial_vel(y);
    state_type y_end{};

    try {
        while (stepper.current_time() < MAX_PATH_CM) {
            const double t0 = stepper.current_time();
            stepper.do_step(system);  // one accepted adaptive step
            const double t1 = stepper.current_time();

            // Clamp evaluation to the 800 cm cap (the dense interpolant is valid
            // across the just-accepted [t0, t1]).
            const bool capped = (t1 >= MAX_PATH_CM);
            const double t_end = capped ? MAX_PATH_CM : t1;
            stepper.calc_state(t_end, y_end);
            const double g_cur = radial_vel(y_end);

            if ((g_prev < 0.0) != (g_cur < 0.0)) {  // radial-velocity sign change
                auto G = [&stepper](double t) {
                    state_type s;
                    stepper.calc_state(t, s);
                    return s[0] * s[3] + s[1] * s[4];
                };
                std::uintmax_t maxit = 60;
                auto tol = [](double a, double b) { return std::fabs(b - a) < 1e-9; };
                auto br = boost::math::tools::bisect(G, t0, t_end, tol, maxit);
                const double s_star = 0.5 * (br.first + br.second);
                state_type ys;
                stepper.calc_state(s_star, ys);
                return {ys[2], std::sqrt(ys[0] * ys[0] + ys[1] * ys[1]), s_star,
                        SwimStatus::Converged};
            }
            if (capped) {
                return {y_end[2], std::sqrt(y_end[0] * y_end[0] + y_end[1] * y_end[1]),
                        MAX_PATH_CM, SwimStatus::MaxPath};
            }
            g_prev = g_cur;
        }
    } catch (...) {
        return fail();
    }

    return {y_end[2], std::sqrt(y_end[0] * y_end[0] + y_end[1] * y_end[1]),
            stepper.current_time(), SwimStatus::MaxPath};
}

}  // namespace vz
