//******************************************************************************
//* HIPO writing core — file writer                                            *
//* Part of the modern C++23 API (see hipo4/hipo.hpp).                         *
//*                                                                            *
//* Quick start:                                                               *
//*                                                                            *
//*     auto schema = hipo::bank_schema::make("event::particle", 100, 1,       *
//*                       {{"pid", hipo::dtype::i16},                          *
//*                        {"px", hipo::dtype::f32}}).value();                 *
//*     hipo::file_writer w("out.hipo");                                       *
//*     auto particle = w.add_schema(schema).value();                          *
//*     auto px       = particle.col<float>("px").value();                     *
//*                                                                            *
//*     hipo::event_builder ev;                                                *
//*     hipo::bank_data b(particle.schema(), nrows);                           *
//*     b.set(px, row, 1.5f);                                                  *
//*     ev.add(b);                                                             *
//*     w.write(ev).value();                                                   *
//*     w.close().value();                                                     *
//*                                                                            *
//* Records are LZ4-compressed and written on a background thread by           *
//* default (mirroring the reader's prefetch pipeline); errors surface on      *
//* the next write() or at close(). The produced files are byte-compatible     *
//* with the classic hipo 4.x writer.                                          *
//******************************************************************************
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "constants.h"
#include "error.hpp"
#include "schema.hpp"
#include "views.hpp"

namespace hipo {

/// Options for creating a file.
struct write_options {
    enum class codec : std::uint8_t {
        none = 0, ///< store records uncompressed
        lz4  = 1, ///< LZ4 fast (the CLAS12 default)
    };

    codec compression = codec::lz4;

    /// Record flush threshold: a record is sealed once it holds this many
    /// (uncompressed) event bytes.
    std::uint32_t record_bytes = 8 * 1024 * 1024;

    /// Compress and write records on a background thread while the caller
    /// keeps building events. Errors surface on a later write()/close().
    bool background = true;
};

/// An owned, mutable bank: a column-major payload for `rows` rows of
/// `schema`. Fill it with typed column handles, add it to an
/// event_builder, reuse it for the next event.
class bank_data {
  public:
    /// The schema must outlive the bank_data.
    bank_data(const bank_schema& schema, std::uint32_t rows)
        : schema_(&schema), rows_(rows),
          data_(std::size_t{rows} * schema.row_size(), std::byte{}) {}

    [[nodiscard]] const bank_schema& schema() const noexcept { return *schema_; }
    [[nodiscard]] std::uint32_t      rows() const noexcept { return rows_; }

    /// Resizes the bank (cells of surviving rows keep their values only
    /// when shrinking; new rows are zeroed).
    void resize(std::uint32_t rows) {
        rows_ = rows;
        data_.assign(std::size_t{rows} * schema_->row_size(), std::byte{});
    }

    /// Typed cell write through a resolved column handle.
    template <bank_element T>
    void set(column<T> c, std::uint32_t row, T value) noexcept {
        std::memcpy(data_.data() + std::size_t{rows_} * c.row_offset() +
                        std::size_t{row} * sizeof(T),
                    &value, sizeof(T));
    }

    /// Name-based cell write with numeric conversion to the stored type.
    /// Convenient for scripts; use set() in hot loops.
    template <bank_element T>
    result<void> put(std::string_view column_name, std::uint32_t row, T value) {
        const auto idx = schema_->index_of(column_name);
        if (!idx)
            return std::unexpected(
                error{errc::unknown_column,
                      std::format("bank \"{}\" has no column \"{}\"", schema_->name(),
                                  column_name)});
        const auto& ci   = schema_->columns()[*idx];
        const auto* base = data_.data() + std::size_t{rows_} * ci.offset +
                           std::size_t{row} * size_of(ci.type);
        const auto store = [&](auto v) { std::memcpy(const_cast<std::byte*>(base), &v, sizeof(v)); };
        switch (ci.type) {
            case dtype::i8:  store(static_cast<std::int8_t>(value)); break;
            case dtype::i16: store(static_cast<std::int16_t>(value)); break;
            case dtype::i32: store(static_cast<std::int32_t>(value)); break;
            case dtype::f32: store(static_cast<float>(value)); break;
            case dtype::f64: store(static_cast<double>(value)); break;
            case dtype::i64: store(static_cast<std::int64_t>(value)); break;
        }
        return {};
    }

    /// Read-back view over the current contents.
    [[nodiscard]] bank_view view() const noexcept {
        return {data_.data(), static_cast<std::uint32_t>(data_.size()), schema_};
    }

    [[nodiscard]] std::span<const std::byte> payload() const noexcept { return data_; }

  private:
    const bank_schema*     schema_;
    std::uint32_t          rows_;
    std::vector<std::byte> data_;
};

/// Assembles one event: the 16-byte event header plus a sequence of
/// structures. Reusable (clear() between events).
class event_builder {
  public:
    event_builder() { clear(); }

    /// Empties the event and resets the tag to 0.
    void clear() {
        buf_.resize(EVENT_HEADER_SIZE);
        buf_[0] = std::byte{'E'};
        buf_[1] = std::byte{'V'};
        buf_[2] = std::byte{'N'};
        buf_[3] = std::byte{'T'};
        write_u32(EH_SIZE_OFFSET, EVENT_HEADER_SIZE);
        write_u32(EH_TAG_OFFSET, 0);
        write_u32(EH_RESERVED_OFFSET, 0);
    }

    /// Events with different tags are routed into different records by the
    /// writer (which is what makes read_options::tags filtering work).
    void set_tag(std::uint32_t tag) noexcept { write_u32(EH_TAG_OFFSET, tag); }

    [[nodiscard]] std::uint32_t tag() const noexcept {
        std::uint32_t v;
        std::memcpy(&v, buf_.data() + EH_TAG_OFFSET, sizeof(v));
        return v;
    }

    /// Appends a bank (type-11 structure).
    void add(const bank_data& bank) {
        add_raw(bank.schema().group(), static_cast<std::uint8_t>(bank.schema().item()), 11,
                bank.payload());
    }

    /// Appends a string structure (type 6) — e.g. metadata strings.
    void add_string(std::uint16_t group, std::uint8_t item, std::string_view text) {
        add_raw(group, item, 6, std::as_bytes(std::span(text.data(), text.size())));
    }

    /// Appends an arbitrary structure.
    void add_raw(std::uint16_t group, std::uint8_t item, std::uint8_t type,
                 std::span<const std::byte> payload) {
        const std::size_t pos = buf_.size();
        buf_.resize(pos + BANK_STRUCTURE_SIZE + payload.size());
        std::memcpy(buf_.data() + pos, &group, sizeof(group));
        buf_[pos + 2] = std::byte{item};
        buf_[pos + 3] = std::byte{type};
        const auto word =
            static_cast<std::uint32_t>(payload.size()) & STRUCT_SIZE_MASK; // format byte 0
        std::memcpy(buf_.data() + pos + 4, &word, sizeof(word));
        if (!payload.empty())
            std::memcpy(buf_.data() + pos + BANK_STRUCTURE_SIZE, payload.data(), payload.size());
        write_u32(EH_SIZE_OFFSET, static_cast<std::uint32_t>(buf_.size()));
    }

    /// The complete event bytes (header included).
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return buf_; }

    /// View over the event being built (e.g. to re-check banks added).
    [[nodiscard]] event_view view() const noexcept { return {bytes(), nullptr}; }

  private:
    void write_u32(int offset, std::uint32_t v) noexcept {
        std::memcpy(buf_.data() + offset, &v, sizeof(v));
    }

    std::vector<std::byte> buf_;
};

namespace detail {
    struct write_pipeline; // defined in file_writer.cpp
    struct sealed_record;  // defined in file_writer.cpp
}

/// Writes a HIPO file: dictionary + configuration records, tagged data
/// records, and the trailer index. Move-only; close() (or destruction)
/// finalizes the file.
class file_writer {
  public:
    [[nodiscard]] static result<file_writer> open(const std::filesystem::path& path,
                                                  write_options options = {});

    /// Throws hipo::io_error when the file cannot be created.
    explicit file_writer(const std::filesystem::path& path, write_options options = {});

    file_writer(file_writer&&) noexcept;
    file_writer& operator=(file_writer&&) noexcept;
    file_writer(const file_writer&)            = delete;
    file_writer& operator=(const file_writer&) = delete;
    ~file_writer(); ///< best-effort close(); call close() to check errors

    /// Registers a bank schema; must happen before the first write().
    /// Returns a handle usable for column resolution and bank_data.
    result<bank_handle> add_schema(bank_schema schema);

    /// Handle over an already-registered schema.
    [[nodiscard]] result<bank_handle> bank(std::string_view name) const;

    /// Adds a key/value pair to the file configuration (stored with the
    /// dictionary; read back via file::user_config). Before first write().
    result<void> config(std::string_view key, std::string_view value);

    /// Appends one event.
    result<void> write(const event_builder& ev) { return write_bytes(ev.bytes(), ev.tag()); }

    /// Appends an event read from another file, verbatim (the skim path).
    result<void> write(event_view ev) { return write_bytes(ev.bytes(), ev.tag()); }

    [[nodiscard]] std::uint64_t event_count() const noexcept { return events_written_; }

    /// Flushes pending records, writes the trailer index, finalizes the
    /// header and closes the file. Idempotent.
    result<void> close();

  private:
    file_writer(); // use open() or the throwing constructor

    result<void> write_bytes(std::span<const std::byte> event_bytes, std::uint32_t tag);
    result<void> start(); // write file header + dictionary record (lazy)

    std::deque<bank_schema>                          schemas_; // stable addresses
    std::vector<std::pair<std::string, std::string>> configs_;
    std::unique_ptr<detail::write_pipeline>          pipe_;
    // per-tag record accumulators (std::map tolerates the incomplete type)
    std::map<std::uint32_t, detail::sealed_record>   accums_;
    write_options                                    opt_;
    std::filesystem::path                            path_;
    std::uint64_t                                    events_written_ = 0;
    bool                                             started_        = false;
    bool                                             closed_         = false;
};

} // namespace hipo
