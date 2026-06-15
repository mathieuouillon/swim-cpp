#include "file_writer.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>

#ifdef __LZ4__
#include <lz4.h>
#endif

namespace hipo {

namespace detail {

    namespace {
        template <class T>
        void store(std::byte* base, int offset, T v) noexcept {
            std::memcpy(base + offset, &v, sizeof(T));
        }
    } // namespace

    /// One record's worth of events, sealed and ready for compression.
    struct sealed_record {
        std::vector<std::byte> index;  // 4 bytes per event (sizes)
        std::vector<std::byte> events; // event bytes
        std::uint32_t          count = 0;
        std::int64_t           user1 = 0;
        std::int64_t           user2 = 0;

        void clear() noexcept {
            index.clear();
            events.clear();
            count = 0;
            user1 = 0;
            user2 = 0;
        }
    };

    struct record_info {
        std::int64_t  position;
        std::int32_t  length;
        std::int32_t  entries;
        std::int64_t  user1;
        std::int64_t  user2;
    };

    /// Owns the output stream and (optionally) the background thread that
    /// compresses and writes sealed records in submission order.
    struct write_pipeline {
        static constexpr std::size_t max_queue = 4;

        std::ofstream            out;
        write_options            opt;
        std::vector<record_info> records; // worker-owned while running

        std::mutex                  m;
        std::condition_variable_any cv_submit; // worker waits (stop_token aware)
        std::condition_variable     cv_space;  // producer waits for queue space / drain
        std::deque<sealed_record>   queue;
        std::vector<sealed_record>  free_list;
        std::optional<error>        err;
        bool                        busy = false; // worker is processing a record
        std::jthread                worker;       // last member: joins first

        // worker-local scratch (single worker; also used inline in sync mode)
        std::vector<std::byte> src_buf;
        std::vector<std::byte> record_buf;

        explicit write_pipeline(write_options options) : opt(options) {}

        ~write_pipeline() {
            if (worker.joinable()) {
                worker.request_stop();
                cv_submit.notify_all();
            }
        }

        result<void> open(const std::filesystem::path& path) {
            out.open(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open())
                return std::unexpected(error{
                    errc::cannot_open,
                    std::format("cannot create \"{}\" for writing", path.string())});
            if (opt.background)
                worker = std::jthread([this](std::stop_token st) { worker_loop(st); });
            return {};
        }

        /// Builds the complete on-disk record (56-byte header + compressed
        /// payload + padding) for `rec` into `record_buf`.
        result<void> seal_to_bytes(const sealed_record& rec) {
            const std::size_t index_size  = rec.index.size();
            const std::size_t events_size = rec.events.size();
            const std::size_t uncompressed = index_size + events_size;

            src_buf.resize(uncompressed);
            if (index_size) std::memcpy(src_buf.data(), rec.index.data(), index_size);
            if (events_size)
                std::memcpy(src_buf.data() + index_size, rec.events.data(), events_size);

            std::size_t payload_size = 0;
            std::uint32_t comp_type  = 0;
            if (opt.compression == write_options::codec::lz4) {
#ifdef __LZ4__
                record_buf.resize(RECORD_HEADER_SIZE +
                                  static_cast<std::size_t>(LZ4_compressBound(
                                      static_cast<int>(uncompressed))) +
                                  4);
                const int n = LZ4_compress_fast(
                    reinterpret_cast<const char*>(src_buf.data()),
                    reinterpret_cast<char*>(record_buf.data() + RECORD_HEADER_SIZE),
                    static_cast<int>(uncompressed),
                    static_cast<int>(record_buf.size() - RECORD_HEADER_SIZE - 4), 1);
                if (n <= 0)
                    return std::unexpected(
                        error{errc::write_failed,
                              std::format("LZ4 compression failed for a {}-byte record",
                                          uncompressed)});
                payload_size = static_cast<std::size_t>(n);
                comp_type    = 1;
#else
                return std::unexpected(error{errc::unsupported_compression,
                                             "this build has no LZ4 support (__LZ4__ not set)"});
#endif
            } else {
                record_buf.resize(RECORD_HEADER_SIZE + uncompressed + 4);
                std::memcpy(record_buf.data() + RECORD_HEADER_SIZE, src_buf.data(),
                            uncompressed);
                payload_size = uncompressed;
            }

            const std::uint32_t pad3  = (4 - payload_size % 4) % 4;
            const std::size_t   total = payload_size + pad3;
            std::memset(record_buf.data() + RECORD_HEADER_SIZE + payload_size, 0, pad3);
            record_buf.resize(RECORD_HEADER_SIZE + total);

            const auto payload_words = static_cast<std::uint32_t>(total / 4);
            std::byte* h             = record_buf.data();
            store<std::uint32_t>(h, RH_RECORD_LENGTH_OFFSET, payload_words + RECORD_HEADER_WORDS);
            store<std::uint32_t>(h, RH_RECORD_NUMBER_OFFSET, 0);
            store<std::uint32_t>(h, RH_HEADER_LENGTH_OFFSET, RECORD_HEADER_WORDS);
            store<std::uint32_t>(h, RH_EVENT_COUNT_OFFSET, rec.count);
            store<std::uint32_t>(h, RH_INDEX_ARRAY_LEN_OFFSET,
                                 static_cast<std::uint32_t>(index_size));
            store<std::uint32_t>(h, RH_BIT_INFO_OFFSET,
                                 (pad3 << BITINFO_PAD3_SHIFT) | HIPO_VERSION);
            store<std::uint32_t>(h, RH_USER_HEADER_LEN_OFFSET, 0);
            store<std::uint32_t>(h, RH_MAGIC_NUMBER_OFFSET, HEADER_MAGIC);
            store<std::uint32_t>(h, RH_DATA_LENGTH_OFFSET,
                                 static_cast<std::uint32_t>(events_size));
            store<std::uint32_t>(h, RH_COMP_WORD_OFFSET,
                                 (comp_type << COMP_TYPE_SHIFT) |
                                     (COMP_LENGTH_MASK & payload_words));
            store<std::int64_t>(h, RH_USER_WORD1_OFFSET, rec.user1);
            store<std::int64_t>(h, RH_USER_WORD2_OFFSET, rec.user2);
            return {};
        }

        /// Seals, writes and registers one record (caller context: worker
        /// thread, or inline when there is no worker).
        result<void> process(const sealed_record& rec) {
            if (rec.count == 0) return {};
            if (auto r = seal_to_bytes(rec); !r) return r;
            const auto position = static_cast<std::int64_t>(out.tellp());
            out.write(reinterpret_cast<const char*>(record_buf.data()),
                      static_cast<std::streamsize>(record_buf.size()));
            if (out.fail())
                return std::unexpected(
                    error{errc::write_failed,
                          std::format("write of a {}-byte record at offset {} failed",
                                      record_buf.size(), position)});
            records.push_back({position, static_cast<std::int32_t>(record_buf.size()),
                               static_cast<std::int32_t>(rec.count), rec.user1, rec.user2});
            return {};
        }

        /// Hands a sealed record to the pipeline (bounded queue when a
        /// worker runs; processed inline otherwise).
        result<void> submit(sealed_record&& rec) {
            if (!worker.joinable()) {
                auto r = process(rec);
                recycle(std::move(rec));
                return r;
            }
            std::unique_lock lk(m);
            if (err) return std::unexpected(*err);
            cv_space.wait(lk, [&] { return queue.size() < max_queue; });
            queue.push_back(std::move(rec));
            lk.unlock();
            cv_submit.notify_one();
            return {};
        }

        /// Reuses buffers of an already-written record.
        [[nodiscard]] sealed_record take_buffers() {
            const std::scoped_lock lk(m);
            if (free_list.empty()) return {};
            sealed_record rec = std::move(free_list.back());
            free_list.pop_back();
            rec.clear();
            return rec;
        }

        /// Waits until every submitted record reached the disk and stops
        /// the worker; afterwards process() may be called inline.
        result<void> drain_and_stop() {
            if (worker.joinable()) {
                std::unique_lock lk(m);
                cv_space.wait(lk, [&] { return (queue.empty() && !busy) || err.has_value(); });
                lk.unlock();
                worker.request_stop();
                cv_submit.notify_all();
                worker = {}; // join
            }
            const std::scoped_lock lk(m);
            if (err) return std::unexpected(*err);
            return {};
        }

      private:
        void recycle(sealed_record&& rec) {
            const std::scoped_lock lk(m);
            if (free_list.size() < max_queue) free_list.push_back(std::move(rec));
        }

        void worker_loop(std::stop_token st) {
            std::unique_lock lk(m);
            while (true) {
                cv_submit.wait(lk, st, [&] { return !queue.empty(); });
                if (queue.empty()) {
                    if (st.stop_requested()) return;
                    continue;
                }
                sealed_record rec = std::move(queue.front());
                queue.pop_front();
                busy = true;
                lk.unlock();

                auto res = process(rec);

                lk.lock();
                busy = false;
                if (!res && !err) err = res.error();
                rec.clear();
                if (free_list.size() < max_queue) free_list.push_back(std::move(rec));
                cv_space.notify_all();
            }
        }
    };

} // namespace detail

// ----- open / construction --------------------------------------------------

result<file_writer> file_writer::open(const std::filesystem::path& path,
                                      write_options options) {
    file_writer w;
    w.opt_  = options;
    w.path_ = path;
    w.pipe_ = std::make_unique<detail::write_pipeline>(options);
    if (auto r = w.pipe_->open(path); !r) return std::unexpected(r.error());
    return w;
}

file_writer::file_writer(const std::filesystem::path& path, write_options options) {
    auto w = open(path, options);
    if (!w) throw io_error(w.error());
    *this = std::move(*w);
}

file_writer::file_writer(file_writer&&) noexcept            = default;
file_writer& file_writer::operator=(file_writer&&) noexcept = default;
file_writer::file_writer()                                  = default;

file_writer::~file_writer() {
    if (pipe_ && !closed_) (void)close(); // best effort; close() reports errors
}

// ----- configuration ---------------------------------------------------------

result<bank_handle> file_writer::add_schema(bank_schema schema) {
    if (started_)
        return std::unexpected(error{
            errc::bad_usage, "file_writer::add_schema: dictionary already written "
                             "(add all schemas before the first write)"});
    for (const auto& s : schemas_)
        if (s.name() == schema.name())
            return std::unexpected(
                error{errc::bad_usage, std::format("file_writer::add_schema: duplicate "
                                                   "schema \"{}\"",
                                                   schema.name())});
    schemas_.push_back(std::move(schema));
    return bank_handle(schemas_.back());
}

result<bank_handle> file_writer::bank(std::string_view name) const {
    for (const auto& s : schemas_)
        if (s.name() == name) return bank_handle(s);
    return std::unexpected(error{
        errc::unknown_bank,
        std::format("file_writer has no schema \"{}\" ({} registered)", name, schemas_.size())});
}

result<void> file_writer::config(std::string_view key, std::string_view value) {
    if (started_)
        return std::unexpected(error{
            errc::bad_usage, "file_writer::config: dictionary already written "
                             "(set all configuration before the first write)"});
    configs_.emplace_back(key, value);
    return {};
}

// ----- writing ----------------------------------------------------------------

result<void> file_writer::start() {
    started_ = true;

    // Dictionary record: one event per schema (JSON structure first, then
    // the parseable schema string — the classic writer's layout), followed
    // by one event per configuration pair.
    detail::sealed_record dict;
    event_builder         ev;
    const auto append = [&] {
        const auto bytes = ev.bytes();
        const auto size  = static_cast<std::uint32_t>(bytes.size());
        dict.index.resize(dict.index.size() + 4);
        std::memcpy(dict.index.data() + dict.index.size() - 4, &size, 4);
        dict.events.insert(dict.events.end(), bytes.begin(), bytes.end());
        ++dict.count;
    };
    for (const auto& schema : schemas_) {
        ev.clear();
        ev.add_string(DICT_GROUP, DICT_JSON_ITEM, schema.schema_string_json());
        ev.add_string(DICT_GROUP, DICT_ITEM, schema.schema_string());
        append();
    }
    for (const auto& [key, value] : configs_) {
        ev.clear();
        ev.add_string(CONFIG_GROUP, CONFIG_KEY_ITEM, key);
        ev.add_string(CONFIG_GROUP, CONFIG_STRING_ITEM, value);
        append();
    }

    // The dictionary record's byte size goes into the file header as the
    // "user header length", so seal it before writing the header.
    if (auto r = pipe_->seal_to_bytes(dict); !r) return r;
    const auto dict_bytes = pipe_->record_buf; // copy: header write reuses the pipeline

    std::array<std::byte, FILE_HEADER_SIZE> header{};
    std::byte* h = header.data();
    detail::store<std::uint32_t>(h, FH_UNIQUE_WORD_OFFSET, HIPO_FILE_UNIQUE_WORD);
    detail::store<std::uint32_t>(h, FH_FILE_NUMBER_OFFSET, 1);
    detail::store<std::uint32_t>(h, FH_HEADER_LENGTH_OFFSET, FILE_HEADER_WORDS);
    detail::store<std::uint32_t>(h, FH_RECORD_COUNT_OFFSET, 0);
    detail::store<std::uint32_t>(h, FH_INDEX_ARRAY_LEN_OFFSET, 0);
    detail::store<std::uint32_t>(h, FH_BIT_INFO_OFFSET, BITINFO_VERSION_MASK & HIPO_VERSION);
    detail::store<std::uint32_t>(h, FH_USER_HEADER_LEN_OFFSET,
                                 static_cast<std::uint32_t>(dict_bytes.size()));
    detail::store<std::uint32_t>(h, FH_MAGIC_NUMBER_OFFSET, HEADER_MAGIC);
    detail::store<std::int64_t>(h, FH_USER_REGISTER_OFFSET, 0);
    detail::store<std::int64_t>(h, FH_TRAILER_POS_OFFSET, 0); // patched in close()
    detail::store<std::uint32_t>(h, FH_USER_INT1_OFFSET, 0);
    detail::store<std::uint32_t>(h, FH_USER_INT2_OFFSET, 0);

    auto& out = pipe_->out;
    out.write(reinterpret_cast<const char*>(header.data()), header.size());
    out.write(reinterpret_cast<const char*>(dict_bytes.data()),
              static_cast<std::streamsize>(dict_bytes.size()));
    if (out.fail())
        return std::unexpected(error{
            errc::write_failed,
            std::format("writing the file header of \"{}\" failed", path_.string())});
    return {};
}

result<void> file_writer::write_bytes(std::span<const std::byte> event_bytes,
                                      std::uint32_t tag) {
    if (closed_)
        return std::unexpected(error{errc::bad_usage, "file_writer: write after close"});
    if (event_bytes.size() < EVENT_HEADER_SIZE)
        return std::unexpected(error{errc::bad_usage, "file_writer: malformed event"});
    if (!started_)
        if (auto r = start(); !r) return r;

    auto& acc = accums_[tag];
    if (acc.count == 0 && acc.events.empty()) {
        // fresh accumulator (possibly recycled buffers)
        auto recycled = pipe_->take_buffers();
        acc.index     = std::move(recycled.index);
        acc.events    = std::move(recycled.events);
        acc.events.reserve(opt_.record_bytes);
        acc.user1 = static_cast<std::int64_t>(tag);
    }

    const auto size = static_cast<std::uint32_t>(event_bytes.size());
    acc.index.resize(acc.index.size() + 4);
    std::memcpy(acc.index.data() + acc.index.size() - 4, &size, 4);
    acc.events.insert(acc.events.end(), event_bytes.begin(), event_bytes.end());
    ++acc.count;
    ++events_written_;

    if (acc.events.size() >= opt_.record_bytes) {
        auto sealed = std::move(acc);
        acc.clear();
        if (auto r = pipe_->submit(std::move(sealed)); !r) return r;
    }
    return {};
}

// ----- close -------------------------------------------------------------------

result<void> file_writer::close() {
    if (closed_) return {};
    closed_ = true;
    if (!pipe_) return {};

    if (!started_)
        if (auto r = start(); !r) return r;

    // flush remaining accumulators: tag 0 first, then ascending tags
    for (auto& [tag, acc] : accums_)
        if (acc.count > 0)
            if (auto r = pipe_->submit(std::move(acc)); !r) return r;
    accums_.clear();

    if (auto r = pipe_->drain_and_stop(); !r) return r;

    // trailer: the file index bank (32111,1), column-major
    const auto& records = pipe_->records;
    const auto  rows    = records.size();
    std::vector<std::byte> payload(rows * 32);
    for (std::size_t i = 0; i < rows; ++i) {
        detail::store(payload.data(), static_cast<int>(i * 8), records[i].position);
        detail::store(payload.data(), static_cast<int>(rows * 8 + i * 4), records[i].length);
        detail::store(payload.data(), static_cast<int>(rows * 12 + i * 4), records[i].entries);
        detail::store(payload.data(), static_cast<int>(rows * 16 + i * 8), records[i].user1);
        detail::store(payload.data(), static_cast<int>(rows * 24 + i * 8), records[i].user2);
    }
    event_builder trailer_event;
    trailer_event.add_raw(FILE_INDEX_GROUP, FILE_INDEX_ITEM, 11, payload);

    detail::sealed_record trailer;
    const auto bytes = trailer_event.bytes();
    const auto size  = static_cast<std::uint32_t>(bytes.size());
    trailer.index.resize(4);
    std::memcpy(trailer.index.data(), &size, 4);
    trailer.events.assign(bytes.begin(), bytes.end());
    trailer.count = 1;

    auto& out = pipe_->out;
    const auto trailer_position = static_cast<std::int64_t>(out.tellp());
    if (auto r = pipe_->process(trailer); !r) return r;

    out.seekp(FH_TRAILER_POS_OFFSET);
    out.write(reinterpret_cast<const char*>(&trailer_position), 8);
    out.flush();
    if (out.fail())
        return std::unexpected(
            error{errc::write_failed,
                  std::format("finalizing \"{}\" failed", path_.string())});
    out.close();
    return {};
}

} // namespace hipo
