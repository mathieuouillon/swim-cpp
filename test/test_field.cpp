// Field-map loading test against the real clas12-cmag solenoid map (8.7 MB; the
// 1.1 GB torus is exercised by the end-to-end run instead). Skips (passes) if
// the map is not present so `meson test` works without the large files.
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <optional>

#include "field.hpp"

using namespace vz;

int main() {
    const std::string solenoid =
        "/Users/mathieuouillon/farm/Symm_solenoid_r601_phi1_z1201_21May2019.dat";

    if (!std::filesystem::exists(solenoid)) {
        std::fprintf(stderr, "SKIP: solenoid map not found, skipping field test\n");
        return 0;
    }

    int failures = 0;
    try {
        // Loads only if the header parses and the byte-size check
        // (80 + 12*nphi*nrho*nz) passes. The solenoid has a single phi node, so
        // this exercises the n==1 axis path in the trilinear interpolation.
        FieldMap map = FieldMap::load(solenoid, -1.0, -3.0);

        auto b = map.b_cart(5.0, 0.0, 0.0);
        if (!b) {
            std::fprintf(stderr, "FAIL: solenoid field nullopt at an in-volume point\n");
            ++failures;
        } else {
            std::fprintf(stderr, "B_solenoid(5,0,0) = (%.4f, %.4f, %.4f) kG\n", (*b)[0], (*b)[1],
                         (*b)[2]);
            bool finite = std::isfinite((*b)[0]) && std::isfinite((*b)[1]) && std::isfinite((*b)[2]);
            if (!finite) {
                std::fprintf(stderr, "FAIL: solenoid field not finite\n");
                ++failures;
            }
        }

        // Far out of the grid box -> nullopt.
        if (map.b_cart(0.0, 0.0, 1.0e6)) {
            std::fprintf(stderr, "FAIL: solenoid field non-nullopt out of volume\n");
            ++failures;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: field-map load threw: %s\n", e.what());
        ++failures;
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "field tests passed\n");
    return 0;
}
