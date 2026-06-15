// `analysis` is the per-worker accumulator for the ~1722 ROOT histograms of the
// v_z / PID diagnostic. Each worker thread folds events into its own copy
// (fill_event) and the partials are summed bin-by-bin (merge_from), then written
// to a single ROOT file (write). Port of analysis.rs.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "TH1.h"
#include "TH2.h"
#include "constants.hpp"
#include "field.hpp"
#include "hipo_chain.hpp"

namespace vz {

/// Resolves the wanted banks to their banklist indices once, up front. A value
/// of -1 means the bank is absent from this run's files (the C++ analog of the
/// Rust `Option<&Bank>` being None) — `fill_event` then sees an empty bank_view.
struct bank_index {
    long rec_particle = -1, mc_particle = -1, mc_recmatch = -1, mc_true = -1;
    long rec_track = -1, rec_traj = -1, rec_covmat = -1;
    long rec_scint = -1, rec_scintx = -1, rec_cher = -1, rec_calo = -1;
    explicit bank_index(bank_list& bl);
};

class analysis {
public:
    analysis();

    /// Fold one event into the accumulator.
    auto fill_event(const bank_list& banks, const bank_index& idx, const magnetic_field& field) -> void;

    /// Merge another partial accumulator into this one (bin-by-bin sum).
    auto merge_from(const analysis& other) -> void;

    /// Write every histogram into a single ROOT file (zstd-compressed).
    auto write(const std::string& path) const -> void;

    auto events() const -> std::uint64_t { return events_; }
    auto fills() const -> const std::array<std::array<std::uint64_t, N_SPECIES>, N_SPECIES>& {
        return fills_;
    }

private:
    std::vector<std::unique_ptr<TH1D>> vz_, vz_true_, dvz_;
    std::vector<std::unique_ptr<TH2D>> vz_theta_;
    std::vector<std::unique_ptr<TH1D>> vars_;
    std::vector<std::unique_ptr<TH1D>> res_;
    std::vector<std::unique_ptr<TH2D>> ptrue_;
    std::vector<std::unique_ptr<TH1D>> dp_;
    std::vector<std::unique_ptr<TH2D>> dpvsp_;
    std::vector<std::unique_ptr<TH1D>> dpf_;
    std::vector<std::unique_ptr<TH2D>> dpfvsp_;
    std::vector<std::unique_ptr<TH1D>> genp_;
    std::vector<std::unique_ptr<TH1D>> swum_vz_, swum_minus_rec_;
    std::vector<std::unique_ptr<TH2D>> swum_rec_vs_p_;
    std::vector<std::unique_ptr<TH1D>> swum_minus_true_;
    std::vector<std::unique_ptr<TH2D>> rec_vs_swum_;
    std::vector<std::unique_ptr<TH1D>> recvz_p_, swumvz_p_;
    std::vector<std::unique_ptr<TH1D>> recoswum_vz_, recoswum_minus_rec_, recoswum_minus_true_;
    std::vector<std::unique_ptr<TH2D>> rec_vs_recoswum_;
    std::vector<std::unique_ptr<TH1D>> recoswumvz_p_;
    std::vector<std::unique_ptr<TH2D>> pr1_reco_vs_true_;
    std::vector<std::unique_ptr<TH1D>> pr1_reco_minus_true_;

    // Counters (not written; for the run summary).
    std::uint64_t events_ = 0;
    std::array<std::array<std::uint64_t, N_SPECIES>, N_SPECIES> fills_{};
};

}  // namespace vz
