// Multi-file, record-parallel event loop over the modern hipo4 reading API.
//
// The new hipo4 library (hipo4/hipo.hpp) dropped the bespoke hipo::chain that
// this project was built around. This header re-creates the small slice we
// used — open N files, fold every event through a callback on a thread pool,
// with a progress bar — on top of the new library's thread-safe parallel
// primitive, hipo::file::read_record (one record_data per worker over one
// shared, const file).
//
//   vz::Chain ch(threads, /*progress=*/true, /*verbose=*/false);
//   for (auto& p : paths) ch.add(p);
//   ch.open();
//   vz::BankList banks = ch.getBanks({"REC::Particle", "REC::Traj"});
//   const long i_part = vz::getBanklistIndex(banks, "REC::Particle");
//   ch.process(banks, [&](vz::BankList& bl, int file_idx, long event_idx) {
//       const hipo::bank_view& part = bl[i_part];
//       for (auto r : part.row_ids()) { /* ... */ }
//   });
//
// Each worker owns its own record_data and BankList view-slots, so no hipo
// object is shared mutably across threads; the callback, however, IS shared and
// must be safe to call concurrently (use a per-thread accumulator or a mutex,
// as ParticleDumper / AccumulatorRegistry do).
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include "hipo4/hipo.hpp"
#include "progresstracker.hpp"

namespace vz {

/// Ordered set of bank views for one event, addressed by the position of each
/// wanted bank name — the new-API analog of the old hipo::banklist. The
/// template returned by Chain::getBanks carries only `names`; Chain::process
/// fills `views` (in the same order) per event before invoking the callback.
struct BankList {
    std::vector<std::string>     names;
    std::vector<hipo::bank_view> views;

    [[nodiscard]] hipo::bank_view operator[](long i) const noexcept {
        return views[static_cast<std::size_t>(i)];
    }
    [[nodiscard]] long index_of(std::string_view name) const noexcept {
        for (std::size_t i = 0; i < names.size(); ++i)
            if (names[i] == name) return static_cast<long>(i);
        return -1;
    }
};

/// Position of `name` in the banklist; throws when absent (like the old
/// hipo::getBanklistIndex) so existing `try { ... } catch { -1 }` fallbacks for
/// optional banks keep working unchanged.
[[nodiscard]] inline long getBanklistIndex(const BankList& bl, std::string_view name) {
    const long i = bl.index_of(name);
    if (i < 0) throw std::out_of_range(fmt::format("bank '{}' not in banklist", name));
    return i;
}

/// A set of HIPO files processed as one logical stream.
class Chain {
  public:
    /// threads: worker threads per file (0 = hardware_concurrency). progress:
    /// show a live progress bar. verbose: print a one-line summary per run.
    Chain(int threads, bool progress, bool verbose) noexcept
        : threads_(threads), progress_(progress), verbose_(verbose) {}

    void add(const std::string& path) { paths_.push_back(path); }

    /// Opens every added file (building each one's record/event index). Throws
    /// std::runtime_error on the first file that fails to open.
    void open(bool validate_all = true) {
        (void)validate_all;  // hipo::file::open already validates the header/index
        files_.clear();
        file_event_offset_.clear();
        total_events_ = 0;
        files_.reserve(paths_.size());
        for (const auto& p : paths_) {
            // prefetch_records = 0: no background decode pipeline — we drive
            // records ourselves across the worker threads via read_record.
            hipo::read_options opt;
            opt.prefetch_records = 0;
            auto f = hipo::file::open(p, opt);
            if (!f)
                throw std::runtime_error(fmt::format("cannot open '{}': {}", p, f.error().message));
            file_event_offset_.push_back(total_events_);
            total_events_ += f->event_count();
            files_.push_back(std::move(*f));
        }
        if (files_.empty()) throw std::runtime_error("chain: no files to open");
    }

    [[nodiscard]] std::size_t size()         const noexcept { return files_.size(); }
    [[nodiscard]] long long   total_events() const noexcept {
        return static_cast<long long>(total_events_);
    }

    /// Builds the banklist template for `names`. Every name must exist in the
    /// first file's dictionary (callers pre-filter with present-bank lists);
    /// throws std::runtime_error otherwise, matching the old getBanks contract.
    [[nodiscard]] BankList getBanks(const std::vector<std::string>& names) const {
        if (files_.empty()) throw std::runtime_error("chain: getBanks() before open()");
        for (const auto& n : names)
            if (files_.front().find_schema(n) == nullptr)
                throw std::runtime_error(fmt::format("bank '{}' not in dictionary", n));
        BankList bl;
        bl.names = names;
        bl.views.assign(names.size(), hipo::bank_view{});
        return bl;
    }

    /// Folds every event (or the first `percentage`% of them) through `func`,
    /// called once per event as `func(BankList& bl, int file_idx, long
    /// event_idx)`, where event_idx is the stable global event index. A fixed
    /// pool of worker threads claims (file, record) chunks across ALL files via
    /// one atomic cursor, so there are exactly `nthreads` thread-local
    /// accumulators regardless of file count. `func` runs concurrently across
    /// workers and must be safe to call so.
    template <class F>
    void process(const BankList& banks, F&& func, double percentage = 100.0) {
        if (files_.empty()) return;
        percentage = std::clamp(percentage, 0.0, 100.0);
        const std::uint64_t target =
            percentage >= 100.0 ? total_events_
                                : static_cast<std::uint64_t>(
                                      static_cast<double>(total_events_) * percentage / 100.0);
        const bool capped = target < total_events_;

        const unsigned nthreads =
            threads_ > 0 ? static_cast<unsigned>(threads_)
                         : std::max(1u, std::thread::hardware_concurrency());

        if (verbose_)
            fmt::print("[chain] processing {} of {} events from {} file(s), {} thread(s)\n", target,
                       total_events_, files_.size(), nthreads);

        // Resolve the wanted banks against each file's dictionary once; an empty
        // handle (bank absent in that file) yields an empty view per event,
        // exactly as a -1 banklist index did before.
        std::vector<std::vector<hipo::bank_handle>> file_handles(files_.size());
        for (std::size_t fi = 0; fi < files_.size(); ++fi) {
            file_handles[fi].resize(banks.names.size());
            for (std::size_t i = 0; i < banks.names.size(); ++i)
                if (auto h = files_[fi].bank(banks.names[i])) file_handles[fi][i] = *h;
        }

        // Flat (file, record) work list claimed by one global atomic cursor.
        struct Work {
            std::uint32_t file;
            std::uint32_t rec;
        };
        std::vector<Work> work;
        for (std::size_t fi = 0; fi < files_.size(); ++fi) {
            const std::uint32_t nrec = files_[fi].record_count();
            for (std::uint32_t r = 0; r < nrec; ++r)
                work.push_back({static_cast<std::uint32_t>(fi), r});
        }

        std::unique_ptr<ProgressTracker> bar;
        if (progress_ && target > 0) {
            ProgressTracker::Config cfg;
            cfg.label = "Processing";
            bar = std::make_unique<ProgressTracker>(target, cfg);
            bar->start();
        }

        std::atomic<std::size_t>   next{0};
        std::atomic<std::uint64_t> emitted{0};  // only consulted when `capped`
        std::mutex                 err_mutex;

        auto worker = [&] {
            hipo::record_data rd;
            BankList          local;
            local.names = banks.names;
            local.views.assign(banks.names.size(), hipo::bank_view{});
            try {
                for (;;) {
                    if (capped && emitted.load(std::memory_order_relaxed) >= target) break;
                    const std::size_t w = next.fetch_add(1, std::memory_order_relaxed);
                    if (w >= work.size()) break;
                    const std::uint32_t fi  = work[w].file;
                    const std::uint32_t rec = work[w].rec;
                    const hipo::file&   f   = files_[fi];
                    if (auto r = f.read_record(rec, rd); !r) continue;  // skip bad record
                    const std::vector<hipo::bank_handle>& handles = file_handles[fi];
                    const std::uint64_t event_base =
                        file_event_offset_[fi] + f.record_first_event(rec);
                    const std::uint32_t nev = rd.event_count();
                    for (std::uint32_t e = 0; e < nev; ++e) {
                        if (capped) {
                            const std::uint64_t g = emitted.fetch_add(1, std::memory_order_relaxed);
                            if (g >= target) {
                                emitted.fetch_sub(1, std::memory_order_relaxed);
                                break;
                            }
                        }
                        const hipo::event_view ev = rd.event(e);
                        for (std::size_t i = 0; i < handles.size(); ++i)
                            local.views[i] = ev.get(handles[i]);
                        func(local, static_cast<int>(fi), static_cast<long>(event_base + e));
                        if (bar) bar->increment();
                    }
                }
            } catch (const std::exception& ex) {
                std::lock_guard lk(err_mutex);
                fmt::print(stderr, "[chain] worker error: {}\n", ex.what());
            }
        };

        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (unsigned t = 0; t < nthreads; ++t) pool.emplace_back(worker);
        for (auto& th : pool) th.join();

        if (bar) bar->finish();
    }

  private:
    int  threads_;
    bool progress_;
    bool verbose_;

    std::vector<std::string>   paths_;
    std::vector<hipo::file>    files_;
    std::vector<std::uint64_t> file_event_offset_;  // cumulative global event base per file
    std::uint64_t              total_events_ = 0;
};

}  // namespace vz
