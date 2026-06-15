// Self-re-exec process pool for the swim executables.
//
// Instead of one process with a thread pool, the work is sharded across N
// independent, single-threaded worker processes — each a re-exec of this very
// binary — whose partial ROOT outputs are then merged. On the JLab farm this
// scales cleanly across the cores a batch scheduler hands out, and sidesteps
// ROOT's thread-safety locking and shared global state entirely.
//
// Supervisor sketch (see main.cpp / vz_swim_hist.cpp for the real callers):
//   const std::string exe = self_exe_path(argv[0]);
//   const std::string dir = make_temp_dir("swim-analysis");
//   ... write one shard list per worker, build a full argv per worker ...
//   if (!run_workers(argvs)) { /* a worker failed -> abort */ }
//   merge_root_files(parts, output);          // sums same-named histograms
//   run_counts total; for (c : count_files) total.add_from_file(c);  // summary
//
// The merged ROOT file is byte-equivalent to the old single-process output:
// TFileMerger sums same-named histograms bin-by-bin, exactly like the previous
// analysis::merge_from. Counters that never lived in the ROOT file (event total,
// truth x reco fill grid) travel in a tiny text sidecar per worker instead.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include <fmt/format.h>

#include "Compression.h"
#include "TFileMerger.h"

#include "constants.hpp"  // N_SPECIES

namespace vz {

/// Absolute path to the running executable, for self re-exec: `/proc/self/exe`
/// on Linux (the farm), `_NSGetExecutablePath` on macOS, else the `argv0`
/// fallback. `execv` resolves the result, so a relative macOS path is fine.
[[nodiscard]] inline auto self_exe_path(const char* argv0) -> std::string {
#if defined(__linux__)
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        return std::string(buf);
    }
#elif defined(__APPLE__)
    char buf[4096];
    std::uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) == 0) return std::string(buf);
#endif
    return std::string(argv0 != nullptr ? argv0 : "");
}

/// Create a unique temp directory `<tmpdir>/<prefix>.<pid>` for the worker
/// shards/outputs and return its path. The caller removes it
/// (std::filesystem::remove_all) once the merge is done.
[[nodiscard]] inline auto make_temp_dir(const std::string& prefix) -> std::string {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / fmt::format("{}.{}", prefix, ::getpid());
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec)
        throw std::runtime_error(
            fmt::format("cannot create temp dir '{}': {}", dir.string(), ec.message()));
    return dir.string();
}

/// Create the parent directory of `path` (and any ancestors) if missing — call
/// before writing an output file so e.g. `output/cpp/vz.root` just works. No-op
/// when the path has no directory part or it already exists.
inline auto ensure_parent_dir(const std::string& path) -> void {
    namespace fs = std::filesystem;
    const fs::path parent = fs::path(path).parent_path();
    if (parent.empty()) return;
    std::error_code ec;
    fs::create_directories(parent, ec);
}

/// A process-shared atomic counter backed by a small temp file, so re-exec'd
/// workers can report progress to the supervisor (which renders one aggregate
/// bar). The supervisor map()s it with create=true; each worker map()s the same
/// path with create=false. uint64 atomics are lock-free and address-free, so a
/// single shared mapping is safe across processes; an unopenable path yields a
/// no-op counter.
struct shared_counter {
    std::atomic<std::uint64_t>* cell = nullptr;
    int fd = -1;

    shared_counter() = default;
    shared_counter(const shared_counter&) = delete;
    auto operator=(const shared_counter&) -> shared_counter& = delete;
    shared_counter(shared_counter&& o) noexcept : cell(o.cell), fd(o.fd) {
        o.cell = nullptr;
        o.fd = -1;
    }
    ~shared_counter() {
        if (cell != nullptr) ::munmap(cell, sizeof(std::atomic<std::uint64_t>));
        if (fd >= 0) ::close(fd);
    }

    [[nodiscard]] static auto map(const std::string& path, bool create) -> shared_counter {
        shared_counter s;
        s.fd = ::open(path.c_str(), create ? (O_RDWR | O_CREAT | O_TRUNC) : O_RDWR, 0600);
        if (s.fd < 0) return s;  // null cell -> a silent no-op
        if (create && ::ftruncate(s.fd, sizeof(std::atomic<std::uint64_t>)) != 0) {
            ::close(s.fd);
            s.fd = -1;
            return s;
        }
        void* p = ::mmap(nullptr, sizeof(std::atomic<std::uint64_t>), PROT_READ | PROT_WRITE,
                         MAP_SHARED, s.fd, 0);
        if (p == MAP_FAILED) {
            ::close(s.fd);
            s.fd = -1;
            return s;
        }
        s.cell = static_cast<std::atomic<std::uint64_t>*>(p);
        if (create) s.cell->store(0, std::memory_order_relaxed);
        return s;
    }
    [[nodiscard]] auto valid() const -> bool { return cell != nullptr; }
    auto add(std::uint64_t n) -> void {
        if (cell != nullptr) cell->fetch_add(n, std::memory_order_relaxed);
    }
    [[nodiscard]] auto load() const -> std::uint64_t {
        return cell != nullptr ? cell->load(std::memory_order_relaxed) : 0;
    }
};

/// Fork+exec one worker per argv vector (argv[i][0] is the exe path); returns the
/// child pids, or an empty vector (after reaping any already-started children) if
/// a fork fails. Call from a single-threaded supervisor before starting any
/// progress thread — children re-exec immediately, so the address space is never
/// copied wholesale.
[[nodiscard]] inline auto spawn_workers(const std::vector<std::vector<std::string>>& argvs)
    -> std::vector<::pid_t> {
    std::vector<::pid_t> pids;
    pids.reserve(argvs.size());
    for (const auto& argv : argvs) {
        std::vector<char*> cargv;
        cargv.reserve(argv.size() + 1);
        for (const auto& s : argv) cargv.push_back(const_cast<char*>(s.c_str()));
        cargv.push_back(nullptr);

        const ::pid_t pid = ::fork();
        if (pid < 0) {
            std::perror("fork");
            for (::pid_t p : pids) {
                int st = 0;
                ::waitpid(p, &st, 0);
            }
            return {};
        }
        if (pid == 0) {
            ::execv(cargv[0], cargv.data());
            std::perror("execv");  // only reached if execv failed
            std::_Exit(127);
        }
        pids.push_back(pid);
    }
    return pids;
}

/// Poll-wait for all `pids`, calling `tick()` ~10x/s while any are still running
/// (used to refresh the aggregate progress bar). Returns true iff every worker
/// exited 0; all are reaped even when one fails.
template <class Tick>
[[nodiscard]] inline auto reap_workers(const std::vector<::pid_t>& pids, Tick&& tick) -> bool {
    if (pids.empty()) return false;
    std::vector<bool> done(pids.size(), false);
    std::size_t remaining = pids.size();
    bool ok = true;
    while (remaining > 0) {
        for (std::size_t i = 0; i < pids.size(); ++i) {
            if (done[i]) continue;
            int status = 0;
            const ::pid_t r = ::waitpid(pids[i], &status, WNOHANG);
            if (r == 0) continue;  // still running
            done[i] = true;
            --remaining;
            if (r < 0) {
                ok = false;
            } else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
                fmt::print(stderr, "[pool] worker {} exited with status {}\n", i,
                           WEXITSTATUS(status));
                ok = false;
            } else if (WIFSIGNALED(status)) {
                fmt::print(stderr, "[pool] worker {} killed by signal {}\n", i, WTERMSIG(status));
                ok = false;
            }
        }
        tick();
        if (remaining > 0) {
            const timespec ts{.tv_sec = 0, .tv_nsec = 100'000'000};  // 100 ms
            ::nanosleep(&ts, nullptr);
        }
    }
    return ok;
}

/// Fork+exec all workers and wait for them (no progress callback).
[[nodiscard]] inline auto run_workers(const std::vector<std::vector<std::string>>& argvs) -> bool {
    const std::vector<::pid_t> pids = spawn_workers(argvs);
    if (pids.empty()) return false;
    return reap_workers(pids, [] {});
}

/// The compression for every partial file AND the merged output. They MUST
/// match: only then can TFileMerger fast-clone TTree baskets (a raw compressed
/// copy) instead of decompressing and recompressing every basket — the
/// difference between a few seconds and minutes when merging a multi-GB tree.
/// Writers of partial files (e.g. hipo2root's TFile) must use this too.
[[nodiscard]] inline auto merge_compression() -> int {
    return ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kZSTD, 5);
}

/// Merge partial ROOT files into `out_path`. TTrees are concatenated (baskets
/// fast-cloned when the partials share `merge_compression()`); same-named
/// histograms are summed bin-by-bin — TFileMerger is the engine behind `hadd`.
/// Throws on failure.
inline auto merge_root_files(const std::vector<std::string>& parts, const std::string& out_path,
                             int compression = merge_compression()) -> void {
    TFileMerger merger(/*isLocal=*/kFALSE, /*histoOneGo=*/kFALSE);
    if (!merger.OutputFile(out_path.c_str(), "RECREATE", compression))
        throw std::runtime_error(fmt::format("cannot open merge output '{}'", out_path));
    for (const auto& p : parts)
        if (!merger.AddFile(p.c_str()))
            throw std::runtime_error(fmt::format("cannot add '{}' to the merge", p));
    if (!merger.Merge()) throw std::runtime_error("TFileMerger::Merge() failed");
}

/// Run bookkeeping that the ROOT histogram file does not hold: the total event
/// count and the truth x reco fill grid (for the final summary). Each worker
/// writes one with write(); the supervisor folds them with add_from_file().
/// Plain text, one unsigned integer per line: events, then the N_SPECIES^2
/// fills in row-major order.
struct run_counts {
    std::uint64_t events = 0;
    std::array<std::array<std::uint64_t, N_SPECIES>, N_SPECIES> fills{};

    auto write(const std::string& path) const -> void {
        std::ofstream out(path);
        if (!out) throw std::runtime_error(fmt::format("cannot write counts '{}'", path));
        out << events << '\n';
        for (const auto& row : fills)
            for (std::uint64_t v : row) out << v << '\n';
    }

    auto add_from_file(const std::string& path) -> void {
        std::ifstream in(path);
        if (!in) throw std::runtime_error(fmt::format("cannot read counts '{}'", path));
        std::uint64_t e = 0;
        in >> e;
        events += e;
        for (auto& row : fills)
            for (std::uint64_t& v : row) {
                std::uint64_t x = 0;
                in >> x;
                v += x;
            }
    }
};

}  // namespace vz
