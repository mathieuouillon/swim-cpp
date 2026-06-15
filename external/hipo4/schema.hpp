//******************************************************************************
//* HIPO reading core — bank schemas                                           *
//* Part of the modern C++23 reading API (see hipo4/hipo.hpp).                 *
//******************************************************************************
#pragma once

#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "error.hpp"

namespace hipo {

namespace detail {
    /// Process-unique, never-reused id used to key the get<"...">
    /// resolution caches. Address-based keying would be vulnerable to
    /// allocator address reuse (ABA) after an object is destroyed.
    [[nodiscard]] inline std::uint64_t next_cache_id() noexcept {
        static std::atomic<std::uint64_t> counter{1};
        return counter.fetch_add(1, std::memory_order_relaxed);
    }
} // namespace detail

/// Element type of a bank column. Values match the on-disk HIPO type ids
/// (legacy kByte..kLong; note there is no 6 or 7).
enum class dtype : std::uint8_t {
    i8  = 1,
    i16 = 2,
    i32 = 3,
    f32 = 4,
    f64 = 5,
    i64 = 8,
};

[[nodiscard]] constexpr std::size_t size_of(dtype t) noexcept {
    switch (t) {
        case dtype::i8:  return 1;
        case dtype::i16: return 2;
        case dtype::i32: return 4;
        case dtype::f32: return 4;
        case dtype::f64: return 8;
        case dtype::i64: return 8;
    }
    return 0;
}

[[nodiscard]] constexpr std::string_view to_string(dtype t) noexcept {
    switch (t) {
        case dtype::i8:  return "byte";
        case dtype::i16: return "short";
        case dtype::i32: return "int";
        case dtype::f32: return "float";
        case dtype::f64: return "double";
        case dtype::i64: return "long";
    }
    return "?";
}

namespace detail {
    // Specialized on fundamental types so that both 64-bit integer types
    // (long and long long — int64_t aliases one or the other depending on
    // the platform) are accepted.
    template <class T> struct dtype_for;
    template <> struct dtype_for<signed char> { static constexpr dtype value = dtype::i8;  };
    template <> struct dtype_for<short>       { static constexpr dtype value = dtype::i16; };
    template <> struct dtype_for<int>         { static constexpr dtype value = dtype::i32; };
    template <> struct dtype_for<float>       { static constexpr dtype value = dtype::f32; };
    template <> struct dtype_for<double>      { static constexpr dtype value = dtype::f64; };
    template <> struct dtype_for<long>        { static constexpr dtype value = sizeof(long) == 8 ? dtype::i64 : dtype::i32; };
    template <> struct dtype_for<long long>   { static constexpr dtype value = dtype::i64; };
} // namespace detail

/// C++ types that can be stored in a bank column.
template <class T>
concept bank_element = requires { detail::dtype_for<T>::value; };

/// The HIPO dtype corresponding to a C++ element type.
template <bank_element T>
inline constexpr dtype dtype_of = detail::dtype_for<T>::value;

/// One column of a bank: name, element type, and the byte offset of the
/// column within a single row (prefix sum of the preceding column sizes).
/// Bank payloads are column-major: the cell (column c, row r) of a bank
/// with R rows lives at byte `R * c.offset + r * size_of(c.type)`.
struct column_info {
    std::string   name;
    std::uint32_t offset{};
    dtype         type{};
};

/// Declares one column when building a schema programmatically.
struct column_decl {
    std::string_view name;
    dtype            type;
};

/// Parsed schema of one bank: its name, (group,item) identifier and columns.
class bank_schema {
  public:
    bank_schema() = default;

    // Copies and moves get a FRESH cache id: the destination is a new
    // identity as far as the get<"..."> resolution caches are concerned.
    bank_schema(const bank_schema& o)
        : name_(o.name_), cols_(o.cols_), by_name_(o.by_name_), row_size_(o.row_size_),
          group_(o.group_), item_(o.item_) {}
    bank_schema(bank_schema&& o) noexcept
        : name_(std::move(o.name_)), cols_(std::move(o.cols_)),
          by_name_(std::move(o.by_name_)), row_size_(o.row_size_), group_(o.group_),
          item_(o.item_) {}
    bank_schema& operator=(const bank_schema& o) {
        if (this != &o) {
            name_     = o.name_;
            cols_     = o.cols_;
            by_name_  = o.by_name_;
            row_size_ = o.row_size_;
            group_    = o.group_;
            item_     = o.item_;
            cache_id_ = detail::next_cache_id();
        }
        return *this;
    }
    bank_schema& operator=(bank_schema&& o) noexcept {
        if (this != &o) {
            name_     = std::move(o.name_);
            cols_     = std::move(o.cols_);
            by_name_  = std::move(o.by_name_);
            row_size_ = o.row_size_;
            group_    = o.group_;
            item_     = o.item_;
            cache_id_ = detail::next_cache_id();
        }
        return *this;
    }

    /// Identity key for the cached accessors; unique per schema object
    /// lifetime, never reused.
    [[nodiscard]] std::uint64_t cache_id() const noexcept { return cache_id_; }

    /// Builds a schema programmatically (the writing-side equivalent of
    /// parse()):
    ///     auto s = hipo::bank_schema::make("event::particle", 100, 1,
    ///                  {{"pid", hipo::dtype::i16}, {"px", hipo::dtype::f32}});
    [[nodiscard]] static result<bank_schema> make(std::string_view name, std::uint16_t group,
                                                  std::uint16_t item,
                                                  std::span<const column_decl> columns);
    [[nodiscard]] static result<bank_schema> make(std::string_view name, std::uint16_t group,
                                                  std::uint16_t item,
                                                  std::initializer_list<column_decl> columns) {
        return make(name, group, item, std::span(columns.begin(), columns.size()));
    }

    [[nodiscard]] std::string_view name()  const noexcept { return name_; }
    [[nodiscard]] std::uint16_t    group() const noexcept { return group_; }
    [[nodiscard]] std::uint16_t    item()  const noexcept { return item_; }

    [[nodiscard]] std::span<const column_info> columns() const noexcept { return cols_; }

    /// Total bytes occupied by one row (sum of all column sizes).
    [[nodiscard]] std::uint32_t row_size() const noexcept { return row_size_; }

    /// Index of the column with the given name, or nullopt.
    [[nodiscard]] std::optional<std::uint32_t> index_of(std::string_view name) const noexcept;

    /// Parses a dictionary entry of the form
    /// `{name/group/item}{col1/T,col2/T,...}` with T one of B,S,I,F,D,L.
    [[nodiscard]] static result<bank_schema> parse(std::string_view dictionary_entry);

    /// Serializes back to the dictionary-entry format accepted by parse()
    /// (this is what gets stored in a file's dictionary record).
    [[nodiscard]] std::string schema_string() const;

    /// JSON variant stored alongside (legacy-compatible layout).
    [[nodiscard]] std::string schema_string_json() const;

  private:
    std::string                name_;
    std::vector<column_info>   cols_;
    std::vector<std::uint32_t> by_name_;   // column indices sorted by name
    std::uint32_t              row_size_ = 0;
    std::uint16_t              group_    = 0;
    std::uint16_t              item_     = 0;
    std::uint64_t              cache_id_ = detail::next_cache_id();
};

} // namespace hipo
