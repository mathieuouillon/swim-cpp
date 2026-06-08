#include "field.hpp"

#include <bit>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <numbers>
#include <stdexcept>
#include <string>

namespace vz {

namespace {
constexpr std::uint32_t MAGIC = 0x0ced;
constexpr std::size_t HEADER_BYTES = 80;
constexpr std::uint32_t FIELD_CS_CYL = 0;

inline double lerp(double a, double b, double t) { return a * (1.0 - t) + b * t; }
}  // namespace

FieldMap::Axis FieldMap::Axis::make(double mn, double mx, std::size_t count) {
    Axis a;
    a.min = mn;
    a.n = count;
    a.step = count > 1 ? (mx - mn) / (static_cast<double>(count) - 1.0) : 0.0;
    return a;
}

std::optional<std::pair<std::size_t, double>> FieldMap::Axis::frac(double q) const {
    if (n == 1) return std::make_pair(std::size_t{0}, 0.0);
    double f = (q - min) / step;
    if (f < 0.0 || f > static_cast<double>(n - 1)) return std::nullopt;
    std::size_t i = static_cast<std::size_t>(std::floor(f));
    if (i > n - 2) i = n - 2;
    return std::make_pair(i, f - static_cast<double>(i));
}

FieldMap FieldMap::load(const std::string& path, double scale, double z_shift) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open field map: " + path);
    const std::streamsize size = in.tellg();
    if (size < static_cast<std::streamsize>(HEADER_BYTES)) {
        throw std::runtime_error("field map shorter than header: " + path);
    }
    in.seekg(0);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    if (!in.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("failed to read field map: " + path);
    }

    auto word = [&](std::size_t i) -> std::uint32_t {
        return static_cast<std::uint32_t>(bytes[i]) | (static_cast<std::uint32_t>(bytes[i + 1]) << 8) |
               (static_cast<std::uint32_t>(bytes[i + 2]) << 16) |
               (static_cast<std::uint32_t>(bytes[i + 3]) << 24);
    };
    auto bswap = [](std::uint32_t v) -> std::uint32_t {
        return ((v & 0x000000ffu) << 24) | ((v & 0x0000ff00u) << 8) |
               ((v & 0x00ff0000u) >> 8) | ((v & 0xff000000u) >> 24);
    };

    // `word` reads little-endian; the file is big-endian, so byte-swap unless the
    // magic says it is already little-endian.
    std::uint32_t le0 = word(0);
    bool big;
    if (bswap(le0) == MAGIC) {
        big = true;
    } else if (le0 == MAGIC) {
        big = false;
    } else {
        throw std::runtime_error("bad magic word in field map: " + path);
    }

    auto ru = [&](std::size_t off) -> std::uint32_t {
        std::uint32_t v = word(off);
        return big ? bswap(v) : v;
    };
    auto rf = [&](std::size_t off) -> double {
        std::uint32_t bits = ru(off);
        return static_cast<double>(std::bit_cast<float>(bits));
    };

    std::uint32_t grid_cs = ru(4);
    std::uint32_t field_cs = ru(8);
    std::uint32_t length_units = ru(12);
    std::uint32_t angle_units = ru(16);
    std::uint32_t field_units = ru(20);
    if (grid_cs != 0 || length_units != 0 || angle_units != 0 || field_units != 0) {
        throw std::runtime_error("expected cylindrical grid in cm/deg/kG: " + path);
    }

    Axis phi = Axis::make(rf(24), rf(28), static_cast<std::size_t>(ru(32)));
    Axis rho = Axis::make(rf(36), rf(40), static_cast<std::size_t>(ru(44)));
    Axis z = Axis::make(rf(48), rf(52), static_cast<std::size_t>(ru(56)));

    std::size_t n_nodes = phi.n * rho.n * z.n;
    std::size_t expect = HEADER_BYTES + 12 * n_nodes;
    if (bytes.size() != expect) {
        throw std::runtime_error("field map size " + std::to_string(bytes.size()) +
                                 " != expected " + std::to_string(expect) + ": " + path);
    }

    std::size_t n = 3 * n_nodes;
    FieldMap m;
    m.field_cs_ = field_cs;
    m.phi_ = phi;
    m.rho_ = rho;
    m.z_ = z;
    m.scale_ = scale;
    m.z_shift_ = z_shift;
    m.data_.reserve(n);
    std::size_t off = HEADER_BYTES;
    for (std::size_t k = 0; k < n; ++k) {
        m.data_.push_back(std::bit_cast<float>(ru(off)));
        off += 4;
    }
    return m;
}

std::optional<std::array<double, 3>> FieldMap::interp(double phi, double rho, double z) const {
    auto fp_opt = phi_.frac(phi);
    if (!fp_opt) return std::nullopt;
    auto fr_opt = rho_.frac(rho);
    if (!fr_opt) return std::nullopt;
    auto fz_opt = z_.frac(z);
    if (!fz_opt) return std::nullopt;

    auto [ip, fp] = *fp_opt;
    auto [ir, fr] = *fr_opt;
    auto [iz, fz] = *fz_opt;
    const std::size_t nrho = rho_.n;
    const std::size_t nz = z_.n;
    const std::size_t ip1 = phi_.n > 1 ? ip + 1 : ip;
    const std::size_t ir1 = rho_.n > 1 ? ir + 1 : ir;
    const std::size_t iz1 = z_.n > 1 ? iz + 1 : iz;

    auto node = [&](std::size_t aphi, std::size_t arho, std::size_t az, std::size_t c) -> double {
        return static_cast<double>(data_[((aphi * nrho + arho) * nz + az) * 3 + c]);
    };

    std::array<double, 3> out{};
    for (std::size_t c = 0; c < 3; ++c) {
        double c00 = lerp(node(ip, ir, iz, c), node(ip, ir, iz1, c), fz);
        double c01 = lerp(node(ip, ir1, iz, c), node(ip, ir1, iz1, c), fz);
        double c10 = lerp(node(ip1, ir, iz, c), node(ip1, ir, iz1, c), fz);
        double c11 = lerp(node(ip1, ir1, iz, c), node(ip1, ir1, iz1, c), fz);
        double c0 = lerp(c00, c01, fr);
        double c1 = lerp(c10, c11, fr);
        out[c] = lerp(c0, c1, fp);
    }
    return out;
}

std::optional<std::array<double, 3>> FieldMap::b_cart(double x, double y, double z) const {
    double rho = std::sqrt(x * x + y * y);
    double phi = std::atan2(y, x) * (180.0 / std::numbers::pi);
    if (phi < 0.0) phi += 360.0;
    auto c_opt = interp(phi, rho, z - z_shift_);
    if (!c_opt) return std::nullopt;
    const std::array<double, 3>& c = *c_opt;

    std::array<double, 3> v{};
    if (field_cs_ == FIELD_CS_CYL) {
        double phi_rad = phi * (std::numbers::pi / 180.0);
        double s = std::sin(phi_rad);
        double co = std::cos(phi_rad);
        // c = [Bphi, Brho, Bz] -> lab Cartesian.
        v = {c[1] * co - c[0] * s, c[1] * s + c[0] * co, c[2]};
    } else {
        v = c;  // already [Bx, By, Bz]
    }
    return std::array<double, 3>{v[0] * scale_, v[1] * scale_, v[2] * scale_};
}

CompositeField CompositeField::load(const std::string& torus_path, double torus_scale,
                                    const std::string& solenoid_path, double solenoid_scale,
                                    double solenoid_z_shift) {
    return CompositeField(FieldMap::load(torus_path, torus_scale, 0.0),
                          FieldMap::load(solenoid_path, solenoid_scale, solenoid_z_shift));
}

std::array<double, 3> CompositeField::b_kgauss(double x, double y, double z) const {
    std::array<double, 3> t = torus_.b_cart(x, y, z).value_or(std::array<double, 3>{0.0, 0.0, 0.0});
    std::array<double, 3> s =
        solenoid_.b_cart(x, y, z).value_or(std::array<double, 3>{0.0, 0.0, 0.0});
    return {t[0] + s[0], t[1] + s[1], t[2] + s[2]};
}

}  // namespace vz
