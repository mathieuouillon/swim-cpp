//******************************************************************************
//* HIPO reading core — zero-copy views over decoded records                   *
//* Part of the modern C++23 reading API (see hipo4/hipo.hpp).                 *
//*                                                                            *
//* All types in this header are non-owning views into a decoded record        *
//* (hipo::record_data). They are valid as long as the record they were       *
//* obtained from is: for `hipo::file` iteration that means until the          *
//* iterator advances past the end of the current record.                      *
//******************************************************************************
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <format>
#include <iterator>
#include <ranges>
#include <span>

#include "constants.h"
#include "error.hpp"
#include "schema.hpp"

namespace hipo {

class file;
class bank_view;

/// A string usable as a template parameter: turns each bank/column NAME
/// into a distinct call site, which is what lets get<"px", float>(row)
/// resolve once and cache the result (see bank_view::col and
/// event_view::get).
template <std::size_t N>
struct fixed_string {
    char value[N]{};

    constexpr fixed_string(const char (&s)[N]) noexcept { std::copy_n(s, N, value); }

    [[nodiscard]] constexpr std::string_view view() const noexcept { return {value, N - 1}; }
};

/// Tokens produced by the user-defined literals in hipo::literals
/// ("REC::Particle"_bank, "px"_f32, ...). Usable with operator[] on
/// event_view and bank_view.
template <fixed_string Name>
struct bank_tag {
    static constexpr auto name = Name;
};

template <fixed_string Name, bank_element T>
struct column_tag {
    static constexpr auto name = Name;
    using type                 = T;
};

namespace detail {
    /// Unaligned load. Bank columns are packed with no alignment guarantee
    /// (a byte column shifts everything after it), so all element access
    /// goes through memcpy; compilers lower this to a plain load.
    template <class T>
    [[nodiscard]] inline T load(const std::byte* p) noexcept {
        T v;
        std::memcpy(&v, p, sizeof(T));
        return v;
    }
} // namespace detail

/// Contiguous, typed, read-only view of one bank column (column-major
/// payloads make every column a contiguous byte range).
template <bank_element T>
class column_view {
  public:
    column_view() = default;
    column_view(const std::byte* base, std::uint32_t n) noexcept : base_(base), n_(n) {}

    [[nodiscard]] std::uint32_t size()  const noexcept { return n_; }
    [[nodiscard]] bool          empty() const noexcept { return n_ == 0; }

    [[nodiscard]] T operator[](std::uint32_t i) const noexcept {
        return detail::load<T>(base_ + std::size_t{i} * sizeof(T));
    }

    /// Bulk-copies all elements to `dst` (must have room for size()).
    /// Single memcpy — the fast path for filling user containers.
    void copy_to(T* dst) const noexcept {
        if (n_ != 0) std::memcpy(dst, base_, std::size_t{n_} * sizeof(T));
    }

    class iterator {
      public:
        using iterator_concept  = std::random_access_iterator_tag;
        using iterator_category = std::input_iterator_tag; // reference is a prvalue
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using reference         = T;

        iterator() = default;
        explicit iterator(const std::byte* p) noexcept : p_(p) {}

        [[nodiscard]] T operator*() const noexcept { return detail::load<T>(p_); }
        [[nodiscard]] T operator[](difference_type n) const noexcept {
            return detail::load<T>(p_ + n * static_cast<difference_type>(sizeof(T)));
        }

        iterator& operator++() noexcept { p_ += sizeof(T); return *this; }
        iterator  operator++(int) noexcept { auto t = *this; ++*this; return t; }
        iterator& operator--() noexcept { p_ -= sizeof(T); return *this; }
        iterator  operator--(int) noexcept { auto t = *this; --*this; return t; }
        iterator& operator+=(difference_type n) noexcept { p_ += n * static_cast<difference_type>(sizeof(T)); return *this; }
        iterator& operator-=(difference_type n) noexcept { p_ -= n * static_cast<difference_type>(sizeof(T)); return *this; }

        [[nodiscard]] friend iterator operator+(iterator it, difference_type n) noexcept { return it += n; }
        [[nodiscard]] friend iterator operator+(difference_type n, iterator it) noexcept { return it += n; }
        [[nodiscard]] friend iterator operator-(iterator it, difference_type n) noexcept { return it -= n; }
        [[nodiscard]] friend difference_type operator-(iterator a, iterator b) noexcept {
            return (a.p_ - b.p_) / static_cast<difference_type>(sizeof(T));
        }
        [[nodiscard]] friend bool operator==(iterator a, iterator b) noexcept { return a.p_ == b.p_; }
        [[nodiscard]] friend auto operator<=>(iterator a, iterator b) noexcept { return a.p_ <=> b.p_; }

      private:
        const std::byte* p_ = nullptr;
    };

    [[nodiscard]] iterator begin() const noexcept { return iterator(base_); }
    [[nodiscard]] iterator end()   const noexcept { return iterator(base_ + std::size_t{n_} * sizeof(T)); }

  private:
    const std::byte* base_ = nullptr;
    std::uint32_t    n_    = 0;
};

/// A resolved, typed column accessor. Resolve once with
/// `bank_handle::col<T>("px")`, then read cells with `px(bank, row)` —
/// no name lookup, no type dispatch on the hot path.
template <bank_element T>
class column {
  public:
    column() = default;

    [[nodiscard]] inline T operator()(const bank_view& b, std::uint32_t row) const noexcept;

    /// Index of the column in the schema.
    [[nodiscard]] std::uint32_t index() const noexcept { return index_; }

    /// Byte offset of the column within one row (cell address math:
    /// rows * row_offset() + row * sizeof(T)). Used by writers.
    [[nodiscard]] std::uint32_t row_offset() const noexcept { return offset_; }

  private:
    friend class bank_handle;
    column(std::uint32_t offset, std::uint32_t index) noexcept : offset_(offset), index_(index) {}

    std::uint32_t offset_ = 0;
    std::uint32_t index_  = 0;
    template <bank_element> friend class column_view;
    friend class bank_view;
};

/// A resolved column accessor that converts on read (any numeric column
/// can be read as any bank_element type). Slightly slower than column<T>;
/// useful for generic tooling.
class column_any {
  public:
    column_any() = default;

    template <bank_element T>
    [[nodiscard]] inline T as(const bank_view& b, std::uint32_t row) const noexcept;

    [[nodiscard]] dtype         type()  const noexcept { return type_; }
    [[nodiscard]] std::uint32_t index() const noexcept { return index_; }

  private:
    friend class bank_handle;
    friend class bank_view;
    column_any(std::uint32_t offset, std::uint32_t index, dtype t) noexcept
        : offset_(offset), index_(index), type_(t) {}

    std::uint32_t offset_ = 0;
    std::uint32_t index_  = 0;
    dtype         type_{};
};

/// Read-only view of one bank inside one event. Trivially copyable value
/// type; an absent bank is represented by an empty view (rows() == 0).
class bank_view {
  public:
    bank_view() = default;
    bank_view(const std::byte* data, std::uint32_t size, const bank_schema* schema) noexcept
        : data_(data), size_(size), schema_(schema),
          rows_(schema != nullptr && schema->row_size() != 0 ? size / schema->row_size() : 0) {}

    [[nodiscard]] std::uint32_t rows()  const noexcept { return rows_; }
    [[nodiscard]] bool          empty() const noexcept { return rows_ == 0; }
    [[nodiscard]] explicit operator bool() const noexcept { return rows_ != 0; }

    /// Schema of this bank; nullptr only for default-constructed views.
    [[nodiscard]] const bank_schema* schema() const noexcept { return schema_; }

    /// Range of row indices [0, rows()), for range-for and ranges pipelines.
    [[nodiscard]] auto row_ids() const noexcept { return std::views::iota(0u, rows_); }

    /// Typed contiguous view of one column.
    template <bank_element T>
    [[nodiscard]] column_view<T> col(column<T> c) const noexcept {
        return {data_ + std::size_t{rows_} * c.offset_, rows_};
    }

    /// Typed contiguous view from a convert-on-read accessor.
    /// Precondition: c.type() == dtype_of<T> (exact match — use as<T>()
    /// otherwise). Useful for bulk copies in generic code.
    template <bank_element T>
    [[nodiscard]] inline column_view<T> col(const column_any& c) const noexcept;

    /// Cached name-based column view: resolves "px" against this bank's
    /// schema on the FIRST call at this call site, then reuses the offset
    /// while the schema stays the same (per-file re-resolution is
    /// automatic). Handle-speed access without pre-loop declarations:
    ///
    ///     for (float v : b.col<"px", float>()) ...
    ///
    /// Throws hipo::io_error (unknown_column / type_mismatch) on first
    /// use if the column is absent or T does not match the stored type.
    /// Thread-safe.
    template <fixed_string Name, bank_element T>
    [[nodiscard]] column_view<T> col() const {
        if (schema_ == nullptr) return {}; // absent bank => empty column
        // Keyed on the schema's never-reused cache id (NOT its address —
        // a new schema can reuse a destroyed one's allocation).
        static std::atomic<std::uint64_t> cached{0};
        static std::atomic<std::uint32_t> offset{0};
        if (cached.load(std::memory_order_acquire) != schema_->cache_id()) [[unlikely]] {
            offset.store(resolve_column(Name.view(), dtype_of<T>), std::memory_order_relaxed);
            cached.store(schema_->cache_id(), std::memory_order_release);
        }
        return {data_ + std::size_t{rows_} * offset.load(std::memory_order_relaxed), rows_};
    }

    /// Cached name-based cell access (same mechanism as col<Name, T>()):
    ///
    ///     sum += b.get<"px", float>(row);
    template <fixed_string Name, bank_element T>
    [[nodiscard]] T get(std::uint32_t row) const {
        return col<Name, T>()[row];
    }

    /// Token form of col<Name, T>(): `b["px"_f32]` (see hipo::literals).
    template <fixed_string Name, bank_element T>
    [[nodiscard]] column_view<T> operator[](column_tag<Name, T>) const {
        return col<Name, T>();
    }

    /// Convenience cell access by column name with numeric conversion.
    /// Resolves the name on every call — use column<T> in hot loops.
    /// Throws hipo::io_error (errc::unknown_column) on a bad name.
    template <bank_element T>
    [[nodiscard]] T get(std::string_view column_name, std::uint32_t row) const;

    /// Raw column-major payload of the bank.
    [[nodiscard]] std::span<const std::byte> payload() const noexcept { return {data_, size_}; }

    /// Start of the payload (column-major block).
    [[nodiscard]] const std::byte* data() const noexcept { return data_; }

  private:
    /// Resolution slow path of col<Name, T>(); returns the column's row
    /// offset or throws io_error with a precise message.
    [[nodiscard]] std::uint32_t resolve_column(std::string_view name, dtype want) const {
        const auto idx = schema_->index_of(name);
        if (!idx)
            throw io_error({errc::unknown_column,
                            std::format("bank \"{}\" has no column \"{}\"", schema_->name(),
                                        name)});
        const auto& ci = schema_->columns()[*idx];
        if (ci.type != want)
            throw io_error({errc::type_mismatch,
                            std::format("column \"{}\" of bank \"{}\" is {}, requested {}",
                                        name, schema_->name(), to_string(ci.type),
                                        to_string(want))});
        return ci.offset;
    }

    const std::byte*   data_   = nullptr;
    std::uint32_t      size_   = 0;
    const bank_schema* schema_ = nullptr;
    std::uint32_t      rows_   = 0;
};

template <bank_element T>
inline T column<T>::operator()(const bank_view& b, std::uint32_t row) const noexcept {
    return detail::load<T>(b.data() + std::size_t{b.rows()} * offset_ + std::size_t{row} * sizeof(T));
}

template <bank_element T>
inline column_view<T> bank_view::col(const column_any& c) const noexcept {
    // exact-type precondition; misuse would reinterpret the column bytes
    return {data_ + std::size_t{rows_} * c.offset_, rows_};
}

template <bank_element T>
inline T column_any::as(const bank_view& b, std::uint32_t row) const noexcept {
    const std::byte* cell =
        b.data() + std::size_t{b.rows()} * offset_ + std::size_t{row} * size_of(type_);
    switch (type_) {
        case dtype::i8:  return static_cast<T>(detail::load<std::int8_t>(cell));
        case dtype::i16: return static_cast<T>(detail::load<std::int16_t>(cell));
        case dtype::i32: return static_cast<T>(detail::load<std::int32_t>(cell));
        case dtype::f32: return static_cast<T>(detail::load<float>(cell));
        case dtype::f64: return static_cast<T>(detail::load<double>(cell));
        case dtype::i64: return static_cast<T>(detail::load<std::int64_t>(cell));
    }
    return T{};
}

template <bank_element T>
T bank_view::get(std::string_view column_name, std::uint32_t row) const {
    if (schema_ == nullptr)
        throw io_error({errc::unknown_column,
                        std::format("bank_view::get: no schema (column \"{}\")", column_name)});
    const auto idx = schema_->index_of(column_name);
    if (!idx)
        throw io_error({errc::unknown_column,
                        std::format("bank_view::get: bank \"{}\" has no column \"{}\"",
                                    schema_->name(), column_name)});
    const auto& ci = schema_->columns()[*idx];
    return column_any(ci.offset, *idx, ci.type).as<T>(*this, row);
}

/// A resolved reference to a bank's schema within an open file. Obtain
/// with `file::bank("REC::Particle")`, then use it to extract bank views
/// from events and to resolve typed columns.
class bank_handle {
  public:
    bank_handle() = default;
    /// Build a handle directly from a schema (must outlive the handle).
    explicit bank_handle(const bank_schema& schema) noexcept : schema_(&schema) {}

    [[nodiscard]] explicit operator bool() const noexcept { return schema_ != nullptr; }
    [[nodiscard]] const bank_schema& schema() const noexcept { return *schema_; }
    [[nodiscard]] const bank_schema* schema_ptr() const noexcept { return schema_; }

    /// Resolves a typed column. The C++ type must match the stored dtype
    /// exactly (schema drift is caught here, not silently on read).
    template <bank_element T>
    [[nodiscard]] result<column<T>> col(std::string_view name) const {
        const auto resolved = resolve(name);
        if (!resolved) return std::unexpected(resolved.error());
        const auto [offset, index, type] = *resolved;
        if (type != dtype_of<T>)
            return std::unexpected(
                error{errc::type_mismatch,
                      std::format("column \"{}\" of bank \"{}\" is {}, requested {}", name,
                                  schema_->name(), to_string(type), to_string(dtype_of<T>))});
        return column<T>(offset, index);
    }

    /// Resolves a column with convert-on-read semantics.
    [[nodiscard]] result<column_any> col_any(std::string_view name) const {
        const auto resolved = resolve(name);
        if (!resolved) return std::unexpected(resolved.error());
        const auto [offset, index, type] = *resolved;
        return column_any(offset, index, type);
    }

  private:
    struct resolved_column {
        std::uint32_t offset;
        std::uint32_t index;
        dtype         type;
    };

    [[nodiscard]] result<resolved_column> resolve(std::string_view name) const {
        if (schema_ == nullptr)
            return std::unexpected(error{errc::unknown_bank, "bank_handle: empty handle"});
        const auto idx = schema_->index_of(name);
        if (!idx)
            return std::unexpected(
                error{errc::unknown_column, std::format("bank \"{}\" has no column \"{}\"",
                                                        schema_->name(), name)});
        const auto& ci = schema_->columns()[*idx];
        return resolved_column{ci.offset, *idx, ci.type};
    }

    const bank_schema* schema_ = nullptr;
};

/// Read-only view of one event: the 16-byte event header followed by a
/// sequence of structures (banks).
class event_view {
  public:
    event_view() = default;
    event_view(std::span<const std::byte> data, const file* owner) noexcept
        : data_(data), owner_(owner) {}

    /// Raw bytes of the event (including the 16-byte header).
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return data_; }

    /// The event tag (e.g. used to mark scaler/special events).
    [[nodiscard]] std::uint32_t tag() const noexcept {
        if (data_.size() < EH_TAG_OFFSET + 4) return 0;
        return detail::load<std::uint32_t>(data_.data() + EH_TAG_OFFSET);
    }

    /// Extracts the bank identified by `h`; empty view when not present.
    [[nodiscard]] bank_view get(const bank_handle& h) const noexcept {
        if (!h) return {};
        return find(h.schema());
    }

    /// Extracts several banks in a single pass over the event.
    template <class... Hs>
        requires(sizeof...(Hs) >= 2 && (std::same_as<Hs, bank_handle> && ...))
    [[nodiscard]] std::array<bank_view, sizeof...(Hs)> get(const Hs&... hs) const noexcept {
        std::array<const bank_schema*, sizeof...(Hs)> schemas{hs.schema_ptr()...};
        std::array<bank_view, sizeof...(Hs)>          out{};
        std::size_t remaining = 0;
        for (const auto* s : schemas)
            if (s != nullptr) ++remaining;

        const std::byte* base = data_.data();
        const std::size_t n   = data_.size();
        std::size_t       pos = EVENT_HEADER_SIZE;
        while (remaining != 0 && pos + BANK_STRUCTURE_SIZE <= n) {
            const auto group = detail::load<std::uint16_t>(base + pos);
            const auto item  = detail::load<std::uint8_t>(base + pos + 2);
            const auto word  = detail::load<std::uint32_t>(base + pos + 4);
            const auto len   = word & STRUCT_SIZE_MASK;
            if (pos + BANK_STRUCTURE_SIZE + len > n) break;
            for (std::size_t i = 0; i < schemas.size(); ++i) {
                if (schemas[i] != nullptr && schemas[i]->group() == group &&
                    schemas[i]->item() == item) {
                    out[i]     = bank_view(base + pos + BANK_STRUCTURE_SIZE, len, schemas[i]);
                    schemas[i] = nullptr;
                    --remaining;
                    break;
                }
            }
            pos += BANK_STRUCTURE_SIZE + len;
        }
        return out;
    }

    /// Extracts a bank by name, looked up in the owning file's dictionary.
    /// Convenience for scripts; empty view if the name or bank is absent.
    /// (Defined in file.hpp.)
    [[nodiscard]] inline bank_view get(std::string_view bank_name) const noexcept;

    /// Cached name-based bank extraction: resolves the schema against the
    /// owning file's dictionary on the FIRST call at this call site, then
    /// reuses it while the file stays the same (switching files
    /// re-resolves automatically). Handle-speed access without pre-loop
    /// declarations:
    ///
    ///     auto b = ev.get<"REC::Particle">();
    ///
    /// Throws hipo::io_error (unknown_bank) on first use if the file has
    /// no such schema. A bank absent from this particular event still
    /// yields an empty view, as with handles. Thread-safe.
    /// (Defined in file.hpp.)
    template <fixed_string Name>
    [[nodiscard]] inline bank_view get() const;

    /// Token form: `ev["REC::Particle"_bank]` (see hipo::literals).
    template <fixed_string Name>
    [[nodiscard]] bank_view operator[](bank_tag<Name>) const {
        return get<Name>();
    }

    /// One raw structure of the event, for generic tooling.
    struct structure_ref {
        std::uint16_t              group;
        std::uint8_t               item;
        std::uint8_t               type;
        std::span<const std::byte> payload;
    };

    class structure_iterator {
      public:
        using value_type        = structure_ref;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        structure_iterator() = default;
        structure_iterator(const std::byte* base, std::size_t pos, std::size_t size) noexcept
            : base_(base), pos_(pos), size_(size) {
            normalize();
        }

        [[nodiscard]] structure_ref operator*() const noexcept {
            const auto word = detail::load<std::uint32_t>(base_ + pos_ + 4);
            return {detail::load<std::uint16_t>(base_ + pos_),
                    detail::load<std::uint8_t>(base_ + pos_ + 2),
                    detail::load<std::uint8_t>(base_ + pos_ + 3),
                    {base_ + pos_ + BANK_STRUCTURE_SIZE, word & STRUCT_SIZE_MASK}};
        }

        structure_iterator& operator++() noexcept {
            const auto word = detail::load<std::uint32_t>(base_ + pos_ + 4);
            pos_ += BANK_STRUCTURE_SIZE + (word & STRUCT_SIZE_MASK);
            normalize();
            return *this;
        }
        void operator++(int) noexcept { ++*this; }

        [[nodiscard]] friend bool operator==(const structure_iterator& a,
                                             std::default_sentinel_t) noexcept {
            return a.pos_ >= a.size_;
        }

      private:
        void normalize() noexcept {
            // Stop also when the next structure would overrun the event.
            if (pos_ + BANK_STRUCTURE_SIZE > size_) { pos_ = size_; return; }
            const auto word = detail::load<std::uint32_t>(base_ + pos_ + 4);
            if (pos_ + BANK_STRUCTURE_SIZE + (word & STRUCT_SIZE_MASK) > size_) pos_ = size_;
        }

        const std::byte* base_ = nullptr;
        std::size_t      pos_  = 0;
        std::size_t      size_ = 0;
    };

    /// Forward range over the raw structures of the event.
    [[nodiscard]] auto structures() const noexcept {
        return std::ranges::subrange(
            structure_iterator(data_.data(), EVENT_HEADER_SIZE, data_.size()),
            std::default_sentinel);
    }

    /// Zero-copy view of a string structure (e.g. configuration entries,
    /// schema records) identified by (group,item); nullopt when absent.
    [[nodiscard]] std::optional<std::string_view> string(std::uint16_t group,
                                                         std::uint16_t item) const noexcept {
        for (const auto& s : structures())
            if (s.group == group && s.item == item)
                return std::string_view(reinterpret_cast<const char*>(s.payload.data()),
                                        s.payload.size());
        return std::nullopt;
    }

  private:
    [[nodiscard]] bank_view find(const bank_schema& s) const noexcept {
        const std::byte* base = data_.data();
        const std::size_t n   = data_.size();
        std::size_t       pos = EVENT_HEADER_SIZE;
        const auto group = s.group();
        const auto item  = s.item();
        while (pos + BANK_STRUCTURE_SIZE <= n) {
            const auto g    = detail::load<std::uint16_t>(base + pos);
            const auto it   = detail::load<std::uint8_t>(base + pos + 2);
            const auto word = detail::load<std::uint32_t>(base + pos + 4);
            const auto len  = word & STRUCT_SIZE_MASK;
            if (pos + BANK_STRUCTURE_SIZE + len > n) break;
            if (g == group && it == item)
                return bank_view(base + pos + BANK_STRUCTURE_SIZE, len, &s);
            pos += BANK_STRUCTURE_SIZE + len;
        }
        return {};
    }

    std::span<const std::byte> data_{};
    const file*                owner_ = nullptr;
};

static_assert(std::ranges::random_access_range<column_view<float>>);
static_assert(std::input_iterator<event_view::structure_iterator>);

/// Opt-in literal suffixes for the cached accessors:
///
///     using namespace hipo::literals;
///     auto b = ev["REC::Particle"_bank];
///     for (float v : b["px"_f32]) ...
///     double t = b["time"_f64][row];
namespace literals {
    template <fixed_string S> consteval auto operator""_bank() { return bank_tag<S>{}; }
    template <fixed_string S> consteval auto operator""_i8()  { return column_tag<S, signed char>{}; }
    template <fixed_string S> consteval auto operator""_i16() { return column_tag<S, std::int16_t>{}; }
    template <fixed_string S> consteval auto operator""_i32() { return column_tag<S, std::int32_t>{}; }
    template <fixed_string S> consteval auto operator""_i64() { return column_tag<S, long long>{}; }
    template <fixed_string S> consteval auto operator""_f32() { return column_tag<S, float>{}; }
    template <fixed_string S> consteval auto operator""_f64() { return column_tag<S, double>{}; }
} // namespace literals

} // namespace hipo
