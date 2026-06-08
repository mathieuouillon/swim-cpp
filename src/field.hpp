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
struct Field {
    virtual ~Field() = default;
    virtual std::array<double, 3> b_kgauss(double x, double y, double z) const = 0;
};

class FieldMap {
public:
    /// Load and parse a clas12-cmag binary map. Throws std::runtime_error on any
    /// format/IO error. `scale` multiplies the field; `z_shift` rigidly
    /// translates the map along z (lab z is looked up at z - z_shift).
    static FieldMap load(const std::string& path, double scale, double z_shift);

    /// Lab-frame Cartesian field in kG (scale applied), or nullopt out of volume.
    std::optional<std::array<double, 3>> b_cart(double x, double y, double z) const;

private:
    struct Axis {
        double min = 0.0;
        std::size_t n = 0;
        double step = 0.0;

        static Axis make(double mn, double mx, std::size_t count);
        /// Lower node index and fractional offset in [0,1], or nullopt if out of
        /// range. A single-point axis (n==1) always returns (0, 0.0).
        std::optional<std::pair<std::size_t, double>> frac(double q) const;
    };

    FieldMap() = default;

    /// Trilinear-interpolated native components at (phi[deg], rho[cm], z[cm]),
    /// or nullopt if outside the grid box.
    std::optional<std::array<double, 3>> interp(double phi, double rho, double z) const;

    std::uint32_t field_cs_ = 0;
    Axis phi_{};
    Axis rho_{};
    Axis z_{};
    std::vector<float> data_;  // 3 components per node
    double scale_ = 1.0;
    double z_shift_ = 0.0;
};

/// Composite (torus + solenoid) field; vector sum, out-of-volume maps add 0.
class CompositeField : public Field {
public:
    static CompositeField load(const std::string& torus_path, double torus_scale,
                               const std::string& solenoid_path, double solenoid_scale,
                               double solenoid_z_shift);

    std::array<double, 3> b_kgauss(double x, double y, double z) const override;

private:
    CompositeField(FieldMap torus, FieldMap solenoid)
        : torus_(std::move(torus)), solenoid_(std::move(solenoid)) {}

    FieldMap torus_;
    FieldMap solenoid_;
};

}  // namespace vz
