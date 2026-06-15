// Multi-file, serial event loop over the modern hipo4 reading API.
//
// The new hipo4 library (hipo4/hipo.hpp) dropped the bespoke hipo::chain that
// this project was built around. This header re-creates the small slice we
// used — open N files and fold every event through a callback, with a progress
// bar — on top of the new library's record reader, hipo::file::read_record.
//
//   vz::chain ch(/*progress=*/true, /*verbose=*/false);
//   for (auto& p : paths) ch.add(p);
//   ch.open();
//   vz::bank_list banks = ch.get_banks({"REC::Particle", "REC::Traj"});
//   const long i_part = vz::get_banklist_index(banks, "REC::Particle");
//   ch.process(banks, [&](vz::bank_list& bl, int file_idx, long event_idx) {
//       const hipo::bank_view& part = bl[i_part];
//       for (auto r : part.row_ids()) { /* ... */ }
//   });
//
// The loop is single-threaded: parallelism is across PROCESSES, each handling a
// disjoint shard of the input files (see process_pool.hpp). The callback is
// invoked once per event in file/record order.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "hipo4/hipo.hpp"
#include "progresstracker.hpp"

namespace vz {

/// Ordered set of bank views for one event, addressed by the position of each
/// wanted bank name — the new-API analog of the old hipo::banklist. The
/// template returned by chain::get_banks carries only `names`; chain::process
/// fills `views` (in the same order) per event before invoking the callback.
struct bank_list {
    std::vector<std::string>     names;
    std::vector<hipo::bank_view> views;

    [[nodiscard]] auto operator[](long i) const noexcept -> hipo::bank_view {
        return views[static_cast<std::size_t>(i)];
    }
    [[nodiscard]] auto index_of(std::string_view name) const noexcept -> long {
        for (std::size_t i = 0; i < names.size(); ++i)
            if (names[i] == name) return static_cast<long>(i);
        return -1;
    }
};

/// Position of `name` in the banklist; throws when absent (like the old
/// hipo::get_banklist_index) so existing `try { ... } catch { -1 }` fallbacks for
/// optional banks keep working unchanged.
[[nodiscard]] inline auto get_banklist_index(const bank_list& bl, std::string_view name) -> long {
    const long i = bl.index_of(name);
    if (i < 0) throw std::out_of_range(fmt::format("bank '{}' not in banklist", name));
    return i;
}

/// A set of HIPO files processed as one logical stream.
class chain {
  public:
    /// progress: show a live progress bar. verbose: print a one-line summary
    /// per run. The event loop is single-threaded — parallelism is by process
    /// (see process_pool.hpp), one shard of files per worker.
    chain(bool progress, bool verbose) noexcept : progress_(progress), verbose_(verbose) {}

    auto add(const std::string& path) -> void { paths_.push_back(path); }

    /// Opens every added file (building each one's record/event index). Throws
    /// std::runtime_error on the first file that fails to open.
    auto open(bool validate_all = true) -> void {
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

    [[nodiscard]] auto size()         const noexcept -> std::size_t { return files_.size(); }
    [[nodiscard]] auto total_events() const noexcept -> long long {
        return static_cast<long long>(total_events_);
    }

    /// Builds the banklist template for `names`. Every name must exist in the
    /// first file's dictionary (callers pre-filter with present-bank lists);
    /// throws std::runtime_error otherwise, matching the old get_banks contract.
    [[nodiscard]] auto get_banks(const std::vector<std::string>& names) const -> bank_list {
        if (files_.empty()) throw std::runtime_error("chain: get_banks() before open()");
        for (const auto& n : names)
            if (files_.front().find_schema(n) == nullptr)
                throw std::runtime_error(fmt::format("bank '{}' not in dictionary", n));
        bank_list bl;
        bl.names = names;
        bl.views.assign(names.size(), hipo::bank_view{});
        return bl;
    }

    /// Folds every event (or the first `percentage`% of them) through `func`,
    /// called once per event as `func(bank_list& bl, int file_idx, long
    /// event_idx)`, where event_idx is the stable global event index. The loop
    /// is serial (one record_data, one bank_list) — parallelism is across
    /// processes, each chain holding a disjoint shard of the files.
    template <class F>
    auto process(const bank_list& banks, F&& func, double percentage = 100.0) -> void {
        if (files_.empty()) return;
        percentage = std::clamp(percentage, 0.0, 100.0);
        const std::uint64_t target =
            percentage >= 100.0 ? total_events_
                                : static_cast<std::uint64_t>(
                                      static_cast<double>(total_events_) * percentage / 100.0);
        const bool capped = target < total_events_;

        if (verbose_)
            fmt::print("[chain] processing {} of {} events from {} file(s), serial\n", target,
                       total_events_, files_.size());

        // Resolve the wanted banks against each file's dictionary once; an empty
        // handle (bank absent in that file) yields an empty view per event,
        // exactly as a -1 banklist index did before.
        std::vector<std::vector<hipo::bank_handle>> file_handles(files_.size());
        for (std::size_t fi = 0; fi < files_.size(); ++fi) {
            file_handles[fi].resize(banks.names.size());
            for (std::size_t i = 0; i < banks.names.size(); ++i)
                if (auto h = files_[fi].bank(banks.names[i])) file_handles[fi][i] = *h;
        }

        std::unique_ptr<progress_tracker> bar;
        if (progress_ && target > 0) {
            progress_tracker::config cfg;
            cfg.label = "Processing";
            bar = std::make_unique<progress_tracker>(target, cfg);
            bar->start();
        }

        hipo::record_data rd;
        bank_list          local;
        local.names = banks.names;
        local.views.assign(banks.names.size(), hipo::bank_view{});

        std::uint64_t emitted = 0;
        try {
            for (std::size_t fi = 0; fi < files_.size() && (!capped || emitted < target); ++fi) {
                const hipo::file&                     f       = files_[fi];
                const std::vector<hipo::bank_handle>& handles = file_handles[fi];
                const std::uint32_t                   nrec    = f.record_count();
                for (std::uint32_t rec = 0; rec < nrec && (!capped || emitted < target); ++rec) {
                    if (auto r = f.read_record(rec, rd); !r) continue;  // skip bad record
                    const std::uint64_t event_base =
                        file_event_offset_[fi] + f.record_first_event(rec);
                    const std::uint32_t nev = rd.event_count();
                    for (std::uint32_t e = 0; e < nev; ++e) {
                        if (capped && emitted >= target) break;
                        const hipo::event_view ev = rd.event(e);
                        for (std::size_t i = 0; i < handles.size(); ++i)
                            local.views[i] = ev.get(handles[i]);
                        func(local, static_cast<int>(fi), static_cast<long>(event_base + e));
                        ++emitted;
                        if (bar) bar->increment();
                    }
                }
            }
        } catch (const std::exception& ex) {
            fmt::print(stderr, "[chain] error: {}\n", ex.what());
        }

        if (bar) bar->finish();
    }

  private:
    bool progress_;
    bool verbose_;

    std::vector<std::string>   paths_;
    std::vector<hipo::file>    files_;
    std::vector<std::uint64_t> file_event_offset_;  // cumulative global event base per file
    std::uint64_t              total_events_ = 0;
};

}  // namespace vz
