// Unit tests for the swimmer (port of the Rust swim.rs test, plus a helix case).
#include <array>
#include <cmath>
#include <cstdio>

#include "field.hpp"
#include "swim.hpp"

using namespace vz;

namespace {

struct uniform : magnetic_field {
    double bz;
    explicit uniform(double b) : bz(b) {}
    auto b_kgauss(double, double, double) const -> std::array<double, 3> override {
        return {0.0, 0.0, bz};
    }
};

int failures = 0;
auto check(bool c, const char* msg) -> void {
    if (!c) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

}  // namespace

auto main() -> int {
    // No field -> straight line. Start at (10,0,100) cm, momentum (2,0,5);
    // swimming back (u = -p_hat) heads toward the beamline. Closest approach is
    // at x=0: z = 100 - (5/sqrt(29))*(10/(2/sqrt(29))) = 75 -- inside the z < 2 m
    // target window the swimmer requires. (Charge sign is irrelevant with B=0.)
    {
        uniform field(0.0);
        swim_result r = swim_back_to_beamline(field, {10.0, 0.0, 100.0}, {2.0, 0.0, 5.0}, 1.0);
        std::fprintf(stderr, "zero-field: status=%d vz=%.6f doca=%.3e path=%.3f\n",
                     static_cast<int>(r.status), r.vz, r.doca_rho, r.path);
        check(r.status == swim_status::converged, "zero-field converged");
        check(r.doca_rho < 1e-2, "zero-field doca ~ 0");
        check(std::fabs(r.vz - 75.0) < 0.1, "zero-field vz ~ 75");
    }

    // uniform Bz -> helix: should still reach a closest approach within max path
    // and inside the target window.
    {
        uniform field(10.0);  // 10 kG
        swim_result r = swim_back_to_beamline(field, {10.0, 0.0, 100.0}, {0.3, 0.0, 1.0}, 1.0);
        std::fprintf(stderr, "helix: status=%d vz=%.6f doca=%.3e path=%.3f\n",
                     static_cast<int>(r.status), r.vz, r.doca_rho, r.path);
        check(r.status == swim_status::converged, "helix converged");
        check(r.path < 900.0, "helix path < 900");
        check(std::isfinite(r.vz), "helix vz finite");
        check(r.doca_rho < 11.0, "helix doca within start radius");
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "all swim tests passed\n");
    return 0;
}
