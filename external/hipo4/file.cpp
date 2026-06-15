#include "file.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <format>
#include <mutex>
#include <stop_token>
#include <thread>
#include <unordered_set>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __LZ4__
#include <lz4.h>
#endif

#include "constants.h"

namespace hipo {

namespace detail {

    // ----- input_source ---------------------------------------------------

    input_source::~input_source() {
        if (map_ != nullptr) ::munmap(const_cast<std::byte*>(map_), size_);
        if (fd_ >= 0) ::close(fd_);
    }

    input_source::input_source(input_source&& other) noexcept
        : map_(std::exchange(other.map_, nullptr)),
          size_(std::exchange(other.size_, 0)),
          fd_(std::exchange(other.fd_, -1)) {}

    input_source& input_source::operator=(input_source&& other) noexcept {
        if (this != &other) {
            if (map_ != nullptr) ::munmap(const_cast<std::byte*>(map_), size_);
            if (fd_ >= 0) ::close(fd_);
            map_  = std::exchange(other.map_, nullptr);
            size_ = std::exchange(other.size_, 0);
            fd_   = std::exchange(other.fd_, -1);
        }
        return *this;
    }

    result<input_source> input_source::open(const std::filesystem::path& path, bool prefer_mmap) {
        input_source src;
        src.fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (src.fd_ < 0)
            return std::unexpected(error{
                errc::cannot_open,
                std::format("cannot open \"{}\": {}", path.string(), std::strerror(errno))});
        struct stat st {};
        if (::fstat(src.fd_, &st) != 0 || !S_ISREG(st.st_mode))
            return std::unexpected(error{
                errc::cannot_open,
                std::format("cannot stat \"{}\" (not a regular file?)", path.string())});
        src.size_ = static_cast<std::uint64_t>(st.st_size);

        if (prefer_mmap && src.size_ > 0) {
            void* m = ::mmap(nullptr, src.size_, PROT_READ, MAP_PRIVATE, src.fd_, 0);
            if (m != MAP_FAILED) {
                src.map_ = static_cast<const std::byte*>(m);
                ::madvise(m, src.size_, MADV_SEQUENTIAL); // advisory; ignore failure
            }
            // mmap failure silently falls back to pread.
        }
        return src;
    }

    result<void> input_source::read_at(std::uint64_t offset, std::byte* dst,
                                       std::size_t n) const {
        while (n > 0) {
            const ssize_t got = ::pread(fd_, dst, n, static_cast<off_t>(offset));
            if (got < 0) {
                if (errno == EINTR) continue;
                return std::unexpected(
                    error{errc::truncated_file,
                          std::format("pread failed at offset {}: {}", offset,
                                      std::strerror(errno))});
            }
            if (got == 0)
                return std::unexpected(error{
                    errc::truncated_file,
                    std::format("unexpected end of file at offset {}", offset)});
            dst += got;
            n -= static_cast<std::size_t>(got);
            offset += static_cast<std::uint64_t>(got);
        }
        return {};
    }

    /// Background decode pipeline: a ring of `ring_size` decoded-record
    /// slots filled by `n_workers` threads. Records are independent, so
    /// they decode in parallel; the consumer still receives them strictly
    /// in sequence order. Claims are sequential, the consumer releases
    /// sequentially, and a slot is reused only after the consumer has
    /// moved past it — which is what makes the "views are valid while the
    /// iterator stays within the record" guarantee hold.
    struct prefetcher {
        struct slot {
            record_data data;
            std::size_t seq = SIZE_MAX;       // sequence index currently held
            enum class state : std::uint8_t { empty, decoding, ready, failed } st = state::empty;
            error              err;
            std::exception_ptr eptr;
        };

        prefetcher(const file& f, unsigned ring_size, unsigned n_workers)
            : file_(&f), ring_(std::max<std::size_t>(ring_size, n_workers + 1)) {
            workers_.reserve(n_workers);
            for (unsigned i = 0; i < n_workers; ++i)
                workers_.emplace_back([this](std::stop_token st) { worker_loop(st); });
        }

        ~prefetcher() {
            {
                const std::scoped_lock lk(m_);
                for (auto& w : workers_) w.request_stop();
            }
            cv_worker_.notify_all();
            workers_.clear(); // jthread dtors join
        }

        /// Resets the pipeline to decode `records` (record indices in
        /// iteration order). Waits for in-flight decodes to drain first.
        void restart(std::vector<std::uint32_t> records) {
            std::unique_lock lk(m_);
            ++generation_;
            next_claim_ = seq_records_.size(); // freeze claims from the old sequence
            cv_drained_.wait(lk, [&] { return inflight_ == 0; });
            seq_records_ = std::move(records);
            next_claim_  = 0;
            consumer_at_ = 0;
            for (auto& s : ring_) {
                s.seq = SIZE_MAX;
                s.st  = slot::state::empty;
            }
            lk.unlock();
            cv_worker_.notify_all();
        }

        [[nodiscard]] std::size_t sequence_size() const noexcept { return seq_records_.size(); }

        /// Blocks until the record for sequence index `seq` is decoded and
        /// returns it. Throws io_error / rethrows worker exceptions.
        [[nodiscard]] const record_data* acquire(std::size_t seq) {
            std::unique_lock lk(m_);
            slot& s = ring_[seq % ring_.size()];
            cv_consumer_.wait(lk, [&] {
                return s.seq == seq &&
                       (s.st == slot::state::ready || s.st == slot::state::failed);
            });
            if (s.st == slot::state::failed) {
                if (s.eptr) std::rethrow_exception(s.eptr);
                throw io_error(s.err);
            }
            return &s.data;
        }

        /// Marks sequence index `seq` consumed; its slot becomes reusable.
        void release(std::size_t seq) {
            {
                const std::scoped_lock lk(m_);
                consumer_at_ = seq + 1;
            }
            cv_worker_.notify_all();
        }

      private:
        void worker_loop(std::stop_token st) {
            std::unique_lock lk(m_);
            while (true) {
                cv_worker_.wait(lk, st, [&] {
                    return next_claim_ < seq_records_.size() &&
                           next_claim_ < consumer_at_ + ring_.size();
                });
                if (st.stop_requested()) return;

                const std::size_t   seq = next_claim_++;
                const std::uint64_t gen = generation_;
                slot&               s   = ring_[seq % ring_.size()];
                s.seq  = seq;
                s.st   = slot::state::decoding;
                s.eptr = nullptr;
                const std::uint32_t rec = seq_records_[seq];
                ++inflight_;

                lk.unlock();
                result<void>       res;
                std::exception_ptr eptr;
                try {
                    res = file_->read_record(rec, s.data);
                } catch (...) {
                    eptr = std::current_exception();
                }
                lk.lock();

                --inflight_;
                if (gen != generation_) {
                    // restart() invalidated this decode; discard.
                    s.seq = SIZE_MAX;
                    s.st  = slot::state::empty;
                    cv_drained_.notify_all();
                    continue;
                }
                if (eptr) {
                    s.st   = slot::state::failed;
                    s.eptr = eptr;
                } else if (!res) {
                    s.st  = slot::state::failed;
                    s.err = res.error();
                } else {
                    s.st = slot::state::ready;
                }
                cv_consumer_.notify_all();
            }
        }

        const file*                 file_;
        std::vector<slot>           ring_;
        std::vector<std::jthread>   workers_;
        std::mutex                  m_;
        std::condition_variable_any cv_worker_;   // waits also on stop_token
        std::condition_variable     cv_consumer_;
        std::condition_variable     cv_drained_;
        std::vector<std::uint32_t>  seq_records_;
        std::size_t                 next_claim_  = 0;
        std::size_t                 consumer_at_ = 0;
        std::size_t                 inflight_    = 0;
        std::uint64_t               generation_  = 0;
    };

    namespace {

        template <class T>
        [[nodiscard]] T load(const std::byte* p) noexcept {
            T v;
            std::memcpy(&v, p, sizeof(T));
            return v;
        }

        /// Copies `n` bytes at `offset` from the source (mapping or pread).
        [[nodiscard]] result<void> fetch(const input_source& src, std::uint64_t offset,
                                         std::byte* dst, std::size_t n) {
            if (offset + n > src.size())
                return std::unexpected(error{
                    errc::truncated_file,
                    std::format("read of {} bytes at offset {} past end of file ({} bytes)", n,
                                offset, src.size())});
            if (src.is_mapped()) {
                std::memcpy(dst, src.map_at(offset), n);
                return {};
            }
            return src.read_at(offset, dst, n);
        }

        struct record_header {
            std::uint64_t position;       // file offset of the record
            std::uint32_t total_bytes;    // recordLength * 4
            std::uint32_t header_bytes;   // headerLength * 4
            std::uint32_t nevents;
            std::uint32_t user_header_len;
            std::uint32_t data_len;       // uncompressed event data bytes
            std::uint32_t pad1;           // user header padding
            std::uint32_t pad3;           // compressed data padding
            std::uint32_t comp_type;
            std::uint32_t raw_bitinfo;
            bool          swapped;

            [[nodiscard]] std::uint32_t payload_bytes() const noexcept {
                return total_bytes - header_bytes;
            }
            [[nodiscard]] std::uint32_t decompressed_bytes() const noexcept {
                return 4 * nevents + user_header_len + pad1 + data_len;
            }
            [[nodiscard]] std::uint32_t events_base() const noexcept {
                return 4 * nevents + user_header_len + pad1;
            }
            [[nodiscard]] bool last_record() const noexcept { return (raw_bitinfo >> 8) & 1u; }
        };

        [[nodiscard]] result<record_header> parse_record_header(const input_source& src,
                                                                std::uint64_t position) {
            std::array<std::byte, RECORD_HEADER_SIZE> buf;
            if (auto r = fetch(src, position, buf.data(), buf.size()); !r)
                return std::unexpected(r.error());

            const auto magic = load<std::uint32_t>(buf.data() + RH_MAGIC_NUMBER_OFFSET);
            const bool swapped = magic == HEADER_MAGIC_BE;
            if (!swapped && magic != HEADER_MAGIC)
                return std::unexpected(error{
                    errc::bad_record_header,
                    std::format("bad record magic {:#010x} at offset {}", magic, position)});

            const auto word = [&](int off) {
                const auto v = load<std::uint32_t>(buf.data() + off);
                return swapped ? std::byteswap(v) : v;
            };

            record_header h{};
            h.position        = position;
            h.total_bytes     = word(RH_RECORD_LENGTH_OFFSET) * 4;
            h.header_bytes    = word(RH_HEADER_LENGTH_OFFSET) * 4;
            h.nevents         = word(RH_EVENT_COUNT_OFFSET);
            h.user_header_len = word(RH_USER_HEADER_LEN_OFFSET);
            h.data_len        = word(RH_DATA_LENGTH_OFFSET);
            h.raw_bitinfo     = word(RH_BIT_INFO_OFFSET);
            h.pad1            = (h.raw_bitinfo >> BITINFO_PAD1_SHIFT) & BITINFO_PAD_MASK;
            h.pad3            = (h.raw_bitinfo >> BITINFO_PAD3_SHIFT) & BITINFO_PAD_MASK;
            h.comp_type       = (word(RH_COMP_WORD_OFFSET) >> COMP_TYPE_SHIFT) & COMP_TYPE_BYTE;
            h.swapped         = swapped;

            // Structural sanity: header within record, sizes consistent.
            if (h.header_bytes < RECORD_HEADER_SIZE || h.total_bytes < h.header_bytes ||
                position + h.total_bytes > src.size() ||
                4ull * h.nevents > h.decompressed_bytes())
                return std::unexpected(error{
                    errc::bad_record_header,
                    std::format("inconsistent record header at offset {} (total {} bytes, "
                                "header {} bytes, {} events, file size {})",
                                position, h.total_bytes, h.header_bytes, h.nevents,
                                src.size())});
            return h;
        }

    } // namespace

} // namespace detail

// ----- record decode ------------------------------------------------------

result<void> file::decode_record_at(std::uint64_t position, record_data& into) const {
    const auto header = detail::parse_record_header(src_, position);
    if (!header) return std::unexpected(header.error());
    const auto& h = *header;

    const std::uint64_t data_position = position + h.header_bytes;
    const std::uint32_t decomp_len    = h.decompressed_bytes();

    switch (h.comp_type) {
        case 0: { // uncompressed
            if (src_.is_mapped()) {
                // Zero-copy: the payload is served straight from the mapping.
                into.payload_ = {src_.map_at(data_position), decomp_len};
            } else {
                if (into.buffer_.size() < decomp_len) into.buffer_.resize(decomp_len);
                if (auto r = src_.read_at(data_position, into.buffer_.data(), decomp_len); !r)
                    return r;
                into.payload_ = {into.buffer_.data(), decomp_len};
            }
            break;
        }
        case 1:
        case 2: { // LZ4
#ifdef __LZ4__
            const std::uint32_t comp_len = h.payload_bytes() - h.pad3;
            if (into.buffer_.size() < decomp_len) into.buffer_.resize(decomp_len);
            const std::byte* src_bytes = nullptr;
            if (src_.is_mapped()) {
                src_bytes = src_.map_at(data_position);
            } else {
                if (into.staging_.size() < comp_len) into.staging_.resize(comp_len);
                if (auto r = src_.read_at(data_position, into.staging_.data(), comp_len); !r)
                    return r;
                src_bytes = into.staging_.data();
            }
            const int produced = LZ4_decompress_safe(
                reinterpret_cast<const char*>(src_bytes),
                reinterpret_cast<char*>(into.buffer_.data()), static_cast<int>(comp_len),
                static_cast<int>(decomp_len));
            if (produced < 0 || static_cast<std::uint32_t>(produced) != decomp_len)
                return std::unexpected(error{
                    errc::decompress_failed,
                    std::format("LZ4 decompression of record at offset {} failed "
                                "(returned {}, expected {} bytes)",
                                position, produced, decomp_len)});
            into.payload_ = {into.buffer_.data(), decomp_len};
            break;
#else
            return std::unexpected(error{errc::unsupported_compression,
                                         "this build has no LZ4 support (__LZ4__ not set)"});
#endif
        }
        default:
            return std::unexpected(error{
                errc::unsupported_compression,
                std::format("record at offset {} uses unsupported compression type {}",
                            position, h.comp_type)});
    }

    // Convert the leading per-event size table into cumulative offsets.
    into.offsets_.resize(h.nevents + 1);
    into.offsets_[0]  = 0;
    std::uint32_t cum = 0;
    for (std::uint32_t i = 0; i < h.nevents; ++i) {
        auto size = detail::load<std::uint32_t>(into.payload_.data() + 4ull * i);
        if (h.swapped) size = std::byteswap(size);
        cum += size;
        into.offsets_[i + 1] = cum;
    }
    if (std::uint64_t{h.events_base()} + cum > decomp_len)
        return std::unexpected(
            error{errc::bad_record_header,
                  std::format("record at offset {}: event index overruns payload "
                              "({}+{} > {})",
                              position, h.events_base(), cum, decomp_len)});

    into.nevents_     = h.nevents;
    into.events_base_ = h.events_base();
    into.owner_       = this;
    return {};
}

result<void> file::read_record(std::uint32_t rec, record_data& into) const {
    if (rec >= record_count())
        return std::unexpected(
            error{errc::event_out_of_range,
                  std::format("record {} out of range (file has {})", rec, record_count())});
    return decode_record_at(rec_pos_[rec], into);
}

// ----- open / init ----------------------------------------------------------

result<file> file::open(const std::filesystem::path& path, read_options options) {
    file f;
    if (auto r = f.init(path, std::move(options)); !r) return std::unexpected(r.error());
    return f;
}

file::file(const std::filesystem::path& path, read_options options) {
    if (auto r = init(path, std::move(options)); !r) throw io_error(r.error());
}

result<void> file::init(const std::filesystem::path& path, read_options options) {
    opt_ = std::move(options);

    auto src = detail::input_source::open(path, opt_.use_mmap);
    if (!src) return std::unexpected(src.error());
    src_ = std::move(*src);

    // --- file header (56 bytes) ---
    std::array<std::byte, FILE_HEADER_SIZE> hdr;
    if (auto r = detail::fetch(src_, 0, hdr.data(), hdr.size()); !r)
        return std::unexpected(
            error{errc::not_a_hipo_file,
                  std::format("\"{}\" is too small to be a HIPO file", path.string())});

    const auto magic   = detail::load<std::uint32_t>(hdr.data() + FH_MAGIC_NUMBER_OFFSET);
    const bool swapped = magic == HEADER_MAGIC_BE;
    if (!swapped && magic != HEADER_MAGIC)
        return std::unexpected(
            error{errc::not_a_hipo_file,
                  std::format("\"{}\" is not a HIPO file (magic {:#010x})", path.string(),
                              magic)});
    const auto word32 = [&](int off) {
        const auto v = detail::load<std::uint32_t>(hdr.data() + off);
        return swapped ? std::byteswap(v) : v;
    };
    const auto word64 = [&](int off) {
        const auto v = detail::load<std::uint64_t>(hdr.data() + off);
        return swapped ? std::byteswap(v) : v;
    };

    const std::uint64_t header_bytes     = std::uint64_t{word32(FH_HEADER_LENGTH_OFFSET)} * 4;
    const std::uint32_t user_header_len  = word32(FH_USER_HEADER_LEN_OFFSET);
    const std::uint32_t bitinfo          = word32(FH_BIT_INFO_OFFSET);
    const std::uint64_t trailer_position = word64(FH_TRAILER_POS_OFFSET);
    const bool          has_dictionary   = (bitinfo >> BITINFO_HAS_DICTIONARY_BIT) & 1u;
    const std::uint64_t first_record_pos = header_bytes + user_header_len;
    header_end_                          = header_bytes;

    // --- dictionary ---
    load_dictionary(header_bytes, first_record_pos);

    // --- record index ---
    rec_pos_.clear();
    rec_cum_.assign(1, 0);
    bool indexed = false;
    if (trailer_position >= first_record_pos &&
        trailer_position + RECORD_HEADER_SIZE <= src_.size()) {
        indexed = static_cast<bool>(build_index_from_trailer(trailer_position, opt_.tags));
    }
    if (!indexed) {
        rec_pos_.clear();
        rec_cum_.assign(1, 0);
        build_index_sequential(first_record_pos, has_dictionary, opt_.tags);
    }
    return {};
}

void file::load_dictionary(std::uint64_t header_end, std::uint64_t first_record_position) {
    schemas_.clear();
    record_data tmp;
    // The dictionary record normally sits right after the file header
    // (legacy readDictionary); files with a user-header record may have it
    // at the first record position instead — try both.
    const std::array<std::uint64_t, 2> candidates{header_end, first_record_position};
    for (std::size_t k = 0; k < candidates.size(); ++k) {
        if (k > 0 && candidates[k] == candidates[k - 1]) break;
        const std::uint64_t candidate = candidates[k];
        if (!decode_record_at(candidate, tmp)) continue;
        for (std::uint32_t i = 0; i < tmp.event_count(); ++i) {
            for (const auto& s : tmp.event(i).structures()) {
                if (s.group != DICT_GROUP || s.item != DICT_ITEM) continue;
                const std::string_view entry(reinterpret_cast<const char*>(s.payload.data()),
                                             s.payload.size());
                if (auto schema = bank_schema::parse(entry)) schemas_.push_back(std::move(*schema));
            }
        }
        if (!schemas_.empty()) break;
    }
    std::ranges::stable_sort(schemas_, std::less{},
                             [](const bank_schema& s) { return s.name(); });
    const auto dup = std::ranges::unique(schemas_, std::ranges::equal_to{},
                                         [](const bank_schema& s) { return s.name(); });
    schemas_.erase(dup.begin(), dup.end());
}

result<void> file::build_index_from_trailer(std::uint64_t trailer_position,
                                            const std::vector<std::int64_t>& tags) {
    record_data trailer;
    if (auto r = decode_record_at(trailer_position, trailer); !r) return r;
    if (trailer.event_count() == 0)
        return std::unexpected(error{errc::bad_record_header, "empty trailer record"});

    std::span<const std::byte> index_payload;
    for (const auto& s : trailer.event(0).structures()) {
        if (s.group == FILE_INDEX_GROUP && s.item == FILE_INDEX_ITEM) {
            index_payload = s.payload;
            break;
        }
    }
    if (index_payload.empty())
        return std::unexpected(
            error{errc::bad_record_header, "trailer record carries no file index"});

    const std::unordered_set<std::int64_t> wanted(tags.begin(), tags.end());

    // Column-major index bank: position/L, length/I, entries/I, uid1/L, uid2/L.
    const std::size_t rows = index_payload.size() / 32;
    const auto*       base = index_payload.data();
    for (std::size_t i = 0; i < rows; ++i) {
        const auto position = detail::load<std::int64_t>(base + i * 8);
        const auto entries  = detail::load<std::int32_t>(base + rows * 12 + i * 4);
        const auto uid1     = detail::load<std::int64_t>(base + rows * 16 + i * 8);
        if (!wanted.empty() && !wanted.contains(uid1)) continue;
        if (position < 0 || std::uint64_t(position) + RECORD_HEADER_SIZE > src_.size() ||
            entries < 0)
            return std::unexpected(
                error{errc::bad_record_header,
                      std::format("file index row {} is invalid (position {}, {} events)", i,
                                  position, entries)});
        rec_pos_.push_back(static_cast<std::uint64_t>(position));
        rec_cum_.push_back(rec_cum_.back() + static_cast<std::uint64_t>(entries));
    }
    return {};
}

void file::build_index_sequential(std::uint64_t first_record_position, bool skip_dictionary,
                                  const std::vector<std::int64_t>& /*tags*/) {
    // Fallback for files without a readable trailer (e.g. the writer was
    // killed). Walks the record header chain; tag filtering is not
    // available here (tags live in the trailer index).
    std::uint64_t pos   = first_record_position;
    bool          first = true;
    while (pos + RECORD_HEADER_SIZE <= src_.size()) {
        const auto header = detail::parse_record_header(src_, pos);
        if (!header) {
            truncated_ = true;
            return;
        }
        const auto& h = *header;
        if (pos + h.total_bytes > src_.size()) {
            truncated_ = true;
            return;
        }
        const bool is_dictionary = first && skip_dictionary;
        if (!is_dictionary && !h.last_record() && h.nevents > 0) {
            rec_pos_.push_back(pos);
            rec_cum_.push_back(rec_cum_.back() + h.nevents);
        }
        first = false;
        pos += h.total_bytes;
    }
    if (pos != src_.size()) truncated_ = true;
}

// ----- moves / destructor ---------------------------------------------------

// Moving a file resets any iteration state: the source's background
// pipeline (whose workers hold a pointer to it) is stopped before its
// members are stolen, and the next begin() on the destination recreates it.
file::file(file&& other) noexcept : file() { *this = std::move(other); }

file& file::operator=(file&& other) noexcept {
    if (this != &other) {
        pf_.reset();       // stop our workers (if any) before replacing state
        other.pf_.reset(); // stop theirs before stealing the members they use
        src_         = std::move(other.src_);
        schemas_     = std::move(other.schemas_);
        rec_pos_     = std::move(other.rec_pos_);
        rec_cum_     = std::move(other.rec_cum_);
        iter_data_   = std::move(other.iter_data_);
        scratch_     = std::move(other.scratch_);
        scratch_rec_ = other.scratch_rec_;
        header_end_  = other.header_end_;
        // the moved-to object IS the same open file (schemas_ moved with
        // it), so it inherits the cache identity; the husk gets a fresh
        // one so stale views can never hit the caches through it
        cache_id_       = other.cache_id_;
        other.cache_id_ = detail::next_cache_id();
        opt_         = std::move(other.opt_);
        truncated_   = other.truncated_;
        iter_data_.owner_ = this;
        scratch_.owner_   = this;
    }
    return *this;
}

file::~file() = default;

// ----- dictionary lookups ----------------------------------------------------

const bank_schema* file::find_schema(std::string_view name) const noexcept {
    const auto it = std::ranges::lower_bound(schemas_, name, std::less{},
                                             [](const bank_schema& s) { return s.name(); });
    if (it == schemas_.end() || it->name() != name) return nullptr;
    return &*it;
}

const bank_schema* file::find_schema(std::uint16_t group, std::uint16_t item) const noexcept {
    for (const auto& s : schemas_)
        if (s.group() == group && s.item() == item) return &s;
    return nullptr;
}

std::vector<std::pair<std::string, std::string>> file::user_config() const {
    // The configuration record sits right after the file header (same place
    // the dictionary lives); each of its events carries a key string
    // (32555,1) and a value string (32555,2).
    std::vector<std::pair<std::string, std::string>> config;
    record_data tmp;
    if (!decode_record_at(header_end_, tmp)) return config;
    for (std::uint32_t i = 0; i < tmp.event_count(); ++i) {
        const auto ev    = tmp.event(i);
        const auto key   = ev.string(CONFIG_GROUP, CONFIG_KEY_ITEM);
        const auto value = ev.string(CONFIG_GROUP, CONFIG_STRING_ITEM);
        if (key && !key->empty()) config.emplace_back(*key, value.value_or(""));
    }
    return config;
}

result<bank_handle> file::bank(std::string_view name) const {
    const auto* s = find_schema(name);
    if (s == nullptr)
        return std::unexpected(error{
            errc::unknown_bank,
            std::format("file dictionary has no bank \"{}\" ({} schemas available)", name,
                        schemas_.size())});
    return bank_handle(*s);
}

// ----- iteration --------------------------------------------------------------

file::iterator file::begin() {
    iterator it;
    iter_start(it);
    return it;
}

void file::ensure_prefetcher() {
    if (pf_) return;
    unsigned workers = opt_.decode_threads;
    if (workers == 0) {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        workers = std::clamp(hw / 2, 1u, 4u);
    }
    pf_ = std::make_unique<detail::prefetcher>(*this, opt_.prefetch_records, workers);
}

void file::iter_start(iterator& it) {
    it.f_ = this;
    if (opt_.prefetch_records > 0 && record_count() > 0) {
        ensure_prefetcher();
        std::vector<std::uint32_t> records;
        records.reserve(record_count());
        for (std::uint32_t r = 0; r < record_count(); ++r)
            if (record_event_count(r) > 0) records.push_back(r);
        pf_->restart(std::move(records));

        for (std::size_t seq = 0; seq < pf_->sequence_size(); ++seq) {
            const record_data* slot = pf_->acquire(seq); // throws on decode failure
            if (slot->event_count() == 0) {
                pf_->release(seq);
                continue;
            }
            it.rec_  = static_cast<std::uint32_t>(seq); // sequence index in prefetch mode
            it.ev_   = 0;
            it.nev_  = slot->event_count();
            it.slot_ = slot;
            return;
        }
        it.f_ = nullptr; // no events
        return;
    }

    for (std::uint32_t r = 0; r < record_count(); ++r) {
        if (record_event_count(r) == 0) continue;
        if (auto res = read_record(r, iter_data_); !res) throw io_error(res.error());
        if (iter_data_.event_count() == 0) continue;
        it.rec_  = r;
        it.ev_   = 0;
        it.nev_  = iter_data_.event_count();
        it.slot_ = &iter_data_;
        return;
    }
    it.f_ = nullptr; // no events
}

void file::iter_advance_record(iterator& it) {
    if (pf_) {
        for (std::size_t seq = it.rec_;;) {
            pf_->release(seq);
            if (++seq >= pf_->sequence_size()) break;
            const record_data* slot = pf_->acquire(seq); // throws on decode failure
            if (slot->event_count() == 0) continue;
            it.rec_  = static_cast<std::uint32_t>(seq);
            it.ev_   = 0;
            it.nev_  = slot->event_count();
            it.slot_ = slot;
            return;
        }
        it.f_ = nullptr; // end of file
        return;
    }

    for (std::uint32_t r = it.rec_ + 1; r < record_count(); ++r) {
        if (record_event_count(r) == 0) continue;
        if (auto res = read_record(r, iter_data_); !res) throw io_error(res.error());
        if (iter_data_.event_count() == 0) continue;
        it.rec_ = r;
        it.ev_  = 0;
        it.nev_ = iter_data_.event_count();
        return;
    }
    it.f_ = nullptr; // end of file
}

// ----- random access ------------------------------------------------------------

result<event_view> file::event_at(std::uint64_t index) {
    if (index >= event_count())
        return std::unexpected(
            error{errc::event_out_of_range,
                  std::format("event {} out of range (file has {})", index, event_count())});
    const auto it  = std::ranges::upper_bound(rec_cum_, index);
    const auto rec = static_cast<std::uint32_t>(std::distance(rec_cum_.begin(), it) - 1);
    if (rec != scratch_rec_) {
        if (auto r = read_record(rec, scratch_); !r) return std::unexpected(r.error());
        scratch_rec_ = rec;
    }
    return scratch_.event(static_cast<std::uint32_t>(index - rec_cum_[rec]));
}

} // namespace hipo
