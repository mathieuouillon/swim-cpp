// Per-worker Analysis accumulators for vz::Chain::process, merged on the main
// thread after processing. This is the C++ analog of the Rust par_reduce_records
// (per-worker accumulator merged bin-by-bin), and matches the original C++
// "thread_local histogram set + registry" pattern.
#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "analysis.hpp"

namespace vz {

class AccumulatorRegistry {
public:
    /// Returns the calling thread's Analysis, creating & registering it on first
    /// use. Construction happens under the lock (serializing the one-time TH1D
    /// creation per thread); the cached raw pointer stays valid because the
    /// registry owns each Analysis via unique_ptr (vector reallocation moves only
    /// the pointers, never the Analysis objects). The hot path is lock-free.
    Analysis& local() {
        thread_local Analysis* mine = nullptr;
        if (mine) return *mine;
        std::lock_guard<std::mutex> lk(m_mutex);
        m_accs.push_back(std::make_unique<Analysis>());
        mine = m_accs.back().get();
        return *mine;
    }

    /// Fold all per-thread accumulators into one. Call only after the parallel
    /// process() has fully returned (all workers joined -> happens-before).
    Analysis merge_all() {
        Analysis total;
        for (const auto& a : m_accs) total.merge_from(*a);
        return total;
    }

private:
    std::mutex m_mutex;
    std::vector<std::unique_ptr<Analysis>> m_accs;
};

}  // namespace vz
