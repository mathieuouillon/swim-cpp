// CLAS12 magnetic-field map reader (port of the clas12-cmag binary format) and
// a composite torus + solenoid field. Port of the Rust field.rs.
//
// On-disk format: big-endian, 80-byte header of 20 u32/f32 words
//   [0]magic=0x0ced [1]gridCS(0=cyl) [2]fieldCS(0=cyl,1=cart) [3]lengthUnits(0=cm)
//   [4]angleUnits(0=deg) [5]fieldUnits(0=kG) [6..8]phi(min,max,n) [9..11]rho
//   [12..14]z [15..19]reserved. Then 3*f32 per grid node, phi slowest / z fastest:
//   idx(iphi,irho,iz) = ((iphi*nrho + irho)*nz + iz)*3.
// Components per node are Cartesian (Bx,By,Bz) when fieldCS==1, else cylindrical
// (Bphi,Brho,Bz). All fields here are kG, cm, degrees.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vz {

/// A field the swimmer can query: lab-frame B in kG at a point in cm.
struct magnetic_field {
    virtual ~magnetic_field() = default;
    virtual auto b_kgauss(double x, double y, double z) const -> std::array<double, 3> = 0;
};

class field_map {
public:
    /// Load and parse a clas12-cmag binary map. Throws std::runtime_error on any
    /// format/IO error. `scale` multiplies the field; the shifts rigidly
    /// translate the map (lab point is looked up at x-x_shift, y-y_shift,
    /// z-z_shift). NOTE a transverse shift only translates the lookup point;
    /// the returned Cartesian components are not re-rotated (exact for a
    /// rigidly displaced magnet).
    static auto load(const std::string& path, double scale, double z_shift,
                     double x_shift = 0.0, double y_shift = 0.0) -> field_map;

    /// Lab-frame Cartesian field in kG (scale applied), or nullopt out of volume.
    auto b_cart(double x, double y, double z) const -> std::optional<std::array<double, 3>>;

private:
    struct axis {
        double min = 0.0;
        std::size_t n = 0;
        double step = 0.0;

        static auto make(double mn, double mx, std::size_t count) -> axis;
        /// Lower node index and fractional offset in [0,1], or nullopt if out of
        /// range. A single-point axis (n==1) always returns (0, 0.0).
        auto frac(double q) const -> std::optional<std::pair<std::size_t, double>>;
    };

    field_map() = default;

    /// Trilinear-interpolated native components at (phi[deg], rho[cm], z[cm]),
    /// or nullopt if outside the grid box.
    auto interp(double phi, double rho, double z) const -> std::optional<std::array<double, 3>>;

    std::uint32_t field_cs_ = 0;
    axis phi_{};
    axis rho_{};
    axis z_{};
    std::vector<float> data_;  // 3 components per node
    double scale_ = 1.0;
    double x_shift_ = 0.0;
    double y_shift_ = 0.0;
    double z_shift_ = 0.0;
};

/// Composite (torus + solenoid) field; vector sum, out-of-volume maps add 0.
class composite_field : public magnetic_field {
public:
    static auto load(const std::string& torus_path, double torus_scale,
                     const std::string& solenoid_path, double solenoid_scale,
                     double solenoid_z_shift, double torus_z_shift = 0.0,
                     double torus_x_shift = 0.0, double torus_y_shift = 0.0) -> composite_field;

    auto b_kgauss(double x, double y, double z) const -> std::array<double, 3> override;

private:
    composite_field(field_map torus, field_map solenoid)
        : torus_(std::move(torus)), solenoid_(std::move(solenoid)) {}

    field_map torus_;
    field_map solenoid_;
};

}  // namespace vz
