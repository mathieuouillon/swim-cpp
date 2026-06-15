//******************************************************************************
//* HIPO reading core — file reader                                            *
//* Part of the modern C++23 reading API (see hipo4/hipo.hpp).                 *
//*                                                                            *
//* Quick start:                                                               *
//*                                                                            *
//*     hipo::file f("run.hipo");                                              *
//*     auto particle = f.bank("REC::Particle").value();                       *
//*     auto px       = particle.col<float>("px").value();                     *
//*     for (hipo::event_view ev : f)                                          *
//*         for (float v : ev.get(particle).col(px))                           *
//*             sum += v;                                                      *
//*                                                                            *
//* Reads local files via mmap (pread fallback) and, by default, decodes       *
//* the next records on a background thread while the caller processes the     *
//* current one. Views handed out during iteration stay valid while the        *
//* iterator remains within the same record.                                   *
//******************************************************************************
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "error.hpp"
#include "schema.hpp"
#include "views.hpp"

namespace hipo {

namespace detail {
    class input_source {
      public:
        input_source() = default;
        ~input_source();
        input_source(input_source&&) noexcept;
        input_source& operator=(input_source&&) noexcept;
        input_source(const input_source&)            = delete;
        input_source& operator=(const input_source&) = delete;

        [[nodiscard]] static result<input_source> open(const std::filesystem::path& path,
                                                       bool prefer_mmap);

        [[nodiscard]] std::uint64_t size()      const noexcept { return size_; }
        [[nodiscard]] bool          is_mapped() const noexcept { return map_ != nullptr; }

        /// Pointer into the mapping; only valid when is_mapped().
        [[nodiscard]] const std::byte* map_at(std::uint64_t offset) const noexcept {
            return map_ + offset;
        }

        /// Positional read (thread-safe); used when not mapped.
        [[nodiscard]] result<void> read_at(std::uint64_t offset, std::byte* dst,
                                           std::size_t n) const;

      private:
        const std::byte* map_  = nullptr;
        std::uint64_t    size_ = 0;
        int              fd_   = -1;
    };

    struct prefetcher; // defined in prefetch.cpp
} // namespace detail

/// Options for opening a file.
struct read_options {
    /// Number of decoded records the background pipeline keeps ahead of
    /// the consumer (ring depth). 0 disables background decoding entirely
    /// (fully synchronous reader). Each in-flight record holds its
    /// decompressed payload in memory (typically tens of MB total).
    unsigned prefetch_records = 3;

    /// Worker threads decoding records in the background while you
    /// iterate (records are independent; events are still delivered in
    /// order). 0 = automatic (about half the hardware threads, capped
    /// at 4). Only used when prefetch_records > 0.
    unsigned decode_threads = 0;

    /// When non-empty, only records whose tag (uid) is in this list are
    /// read; everything else is skipped at the index level.
    std::vector<std::int64_t> tags{};

    /// Map the file into memory (default). Set false to force pread.
    bool use_mmap = true;
};

/// One decoded record: decompressed payload plus the event offset table.
/// Reusable decode target — `file::read_record` fills it without
/// reallocating when capacities suffice. Events are zero-copy views into
/// this object (or directly into the file mapping for uncompressed
/// records), valid until the next read_record into the same object.
class record_data {
  public:
    record_data() = default;

    [[nodiscard]] std::uint32_t event_count() const noexcept { return nevents_; }

    /// View of event `i` (precondition: i < event_count()).
    [[nodiscard]] event_view event(std::uint32_t i) const noexcept {
        const auto begin = offsets_[i];
        return {payload_.subspan(events_base_ + begin, offsets_[i + 1] - begin), owner_};
    }

  private:
    friend class file;
    friend struct detail::prefetcher;

    std::vector<std::byte>     buffer_;      // decompressed payload (owned)
    std::vector<std::byte>     staging_;     // compressed staging (pread path only)
    std::span<const std::byte> payload_;     // into buffer_ or into the file mapping
    std::vector<std::uint32_t> offsets_;     // nevents+1 cumulative event offsets
    std::uint32_t              nevents_     = 0;
    std::uint32_t              events_base_ = 0; // index array + user header (+pad)
    const file*                owner_       = nullptr;
};

/// A HIPO file opened for reading.
///
/// Thread-safety: const member functions are safe to call concurrently;
/// in particular `read_record` decodes into caller-owned record_data and
/// is the building block for external parallel readers (one record_data
/// per thread over one shared file). Iteration (`begin`) and `event_at`
/// are stateful and belong to one thread at a time.
class file {
  public:
    /// Opens a file; returns an error instead of throwing.
    [[nodiscard]] static result<file> open(const std::filesystem::path& path,
                                           read_options options = {});

    /// Opens a file; throws hipo::io_error on failure.
    explicit file(const std::filesystem::path& path, read_options options = {});

    file(file&&) noexcept;
    file& operator=(file&&) noexcept;
    file(const file&)            = delete;
    file& operator=(const file&) = delete;
    ~file();

    // ----- dictionary --------------------------------------------------

    /// All bank schemas declared in the file, sorted by name.
    [[nodiscard]] std::span<const bank_schema> schemas() const noexcept { return schemas_; }

    /// Resolves a bank by name ("REC::Particle").
    [[nodiscard]] result<bank_handle> bank(std::string_view name) const;

    /// Schema lookup by name / by (group,item); nullptr when absent.
    [[nodiscard]] const bank_schema* find_schema(std::string_view name) const noexcept;
    [[nodiscard]] const bank_schema* find_schema(std::uint16_t group,
                                                 std::uint16_t item) const noexcept;

    /// Key/value pairs written into the file's configuration record
    /// (structures (32555,1)/(32555,2), e.g. by writer::addUserConfig).
    /// Empty when the file carries no configuration.
    [[nodiscard]] std::vector<std::pair<std::string, std::string>> user_config() const;

    // ----- index --------------------------------------------------------

    [[nodiscard]] std::uint64_t event_count()  const noexcept { return rec_cum_.back(); }
    [[nodiscard]] std::uint32_t record_count() const noexcept {
        return static_cast<std::uint32_t>(rec_pos_.size());
    }
    [[nodiscard]] std::uint64_t record_first_event(std::uint32_t rec) const noexcept {
        return rec_cum_[rec];
    }
    [[nodiscard]] std::uint32_t record_event_count(std::uint32_t rec) const noexcept {
        return static_cast<std::uint32_t>(rec_cum_[rec + 1] - rec_cum_[rec]);
    }

    /// True when the file ends in an incomplete record (e.g. the writer
    /// was killed); the readable prefix is still served.
    [[nodiscard]] bool truncated() const noexcept { return truncated_; }

    /// Identity key for the cached accessors (event_view::get<"...">);
    /// unique per open file, never reused, preserved across moves.
    [[nodiscard]] std::uint64_t cache_id() const noexcept { return cache_id_; }

    // ----- iteration ----------------------------------------------------

    struct sentinel {};

    class iterator {
      public:
        using value_type      = event_view;
        using difference_type = std::ptrdiff_t;

        iterator() = default;

        [[nodiscard]] event_view operator*() const noexcept { return slot_->event(ev_); }

        iterator& operator++() {
            if (++ev_ >= nev_) f_->iter_advance_record(*this);
            return *this;
        }
        void operator++(int) { ++*this; }

        [[nodiscard]] bool operator==(sentinel) const noexcept { return f_ == nullptr; }

      private:
        friend class file;
        file*              f_    = nullptr;
        const record_data* slot_ = nullptr;
        std::uint32_t      rec_  = 0;
        std::uint32_t      ev_   = 0;
        std::uint32_t      nev_  = 0;
    };

    /// (Re)starts iteration from the first event. Single-pass input range;
    /// one active iteration per file. Throws hipo::io_error if a record
    /// fails to decode mid-iteration.
    [[nodiscard]] iterator begin();
    [[nodiscard]] sentinel end() const noexcept { return {}; }

    // ----- random access -------------------------------------------------

    /// Event by global index. Decodes the containing record into an
    /// internal scratch slot; the returned view is valid until the next
    /// event_at call. Does not disturb sequential iteration.
    [[nodiscard]] result<event_view> event_at(std::uint64_t index);

    /// Decodes record `rec` into `into`. Thread-safe and const: this is
    /// the primitive for external parallel processing.
    [[nodiscard]] result<void> read_record(std::uint32_t rec, record_data& into) const;

  private:
    file() = default;

    [[nodiscard]] result<void> init(const std::filesystem::path& path, read_options options);
    [[nodiscard]] result<void> decode_record_at(std::uint64_t position, record_data& into) const;
    [[nodiscard]] result<void> build_index_from_trailer(std::uint64_t trailer_position,
                                                        const std::vector<std::int64_t>& tags);
    void build_index_sequential(std::uint64_t first_record_position, bool skip_dictionary,
                                const std::vector<std::int64_t>& tags);
    void load_dictionary(std::uint64_t header_end, std::uint64_t first_record_position);

    void ensure_prefetcher();
    void iter_start(iterator& it);
    void iter_advance_record(iterator& it);

    detail::input_source                 src_;
    std::vector<bank_schema>             schemas_;   // sorted by name
    std::vector<std::uint64_t>           rec_pos_;   // record file positions
    std::vector<std::uint64_t>           rec_cum_;   // cumulative event counts (size+1)
    std::unique_ptr<detail::prefetcher>  pf_;        // nullptr when prefetch disabled
    record_data                          iter_data_;    // synchronous iteration slot
    record_data                          scratch_;      // event_at slot
    std::uint32_t                        scratch_rec_ = UINT32_MAX;
    std::uint64_t                        header_end_  = 0; // config/dictionary records live here
    std::uint64_t                        cache_id_    = detail::next_cache_id();
    read_options                         opt_;
    bool                                 truncated_ = false;
};

inline bank_view event_view::get(std::string_view bank_name) const noexcept {
    if (owner_ == nullptr) return {};
    const auto* s = owner_->find_schema(bank_name);
    if (s == nullptr) return {};
    return find(*s);
}

template <fixed_string Name>
inline bank_view event_view::get() const {
    // Cache shared by every call site of this Name, keyed on the owning
    // file's never-reused cache id (NOT its address — a new file object
    // can reuse a destroyed one's allocation). A release-store of the id
    // publishes the schema pointer resolved just before it.
    static std::atomic<std::uint64_t>      cached{0};
    static std::atomic<const bank_schema*> schema{nullptr};
    if (owner_ == nullptr)
        throw io_error({errc::unknown_bank,
                        std::format("event_view::get<\"{}\">: event has no owning file "
                                    "(views from event_builder have no dictionary)",
                                    Name.view())});
    if (cached.load(std::memory_order_acquire) != owner_->cache_id()) [[unlikely]] {
        const auto* s = owner_->find_schema(Name.view());
        if (s == nullptr)
            throw io_error({errc::unknown_bank,
                            std::format("file dictionary has no bank \"{}\"", Name.view())});
        schema.store(s, std::memory_order_relaxed);
        cached.store(owner_->cache_id(), std::memory_order_release);
        return find(*s);
    }
    return find(*schema.load(std::memory_order_relaxed));
}

static_assert(std::input_iterator<file::iterator>);
static_assert(std::sentinel_for<file::sentinel, file::iterator>);

} // namespace hipo
