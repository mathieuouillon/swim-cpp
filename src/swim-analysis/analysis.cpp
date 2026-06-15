#include "analysis.hpp"

#include <algorithm>
#include <cmath>

#include <fmt/format.h>

#include "Compression.h"
#include "TFile.h"
#include "bank_access.hpp"
#include "swim.hpp"

namespace vz {

namespace {

// Flat-index helpers into the histogram vectors (must match the construction
// loop order in the constructor).
constexpr auto idx_trp(std::size_t t, std::size_t r, std::size_t b) -> std::size_t {
    return (t * N_SPECIES + r) * N_MOM_BINS + b;
}
constexpr auto idx_tr(std::size_t t, std::size_t r) -> std::size_t { return t * N_SPECIES + r; }
constexpr auto idx_var(std::size_t t, std::size_t r, std::size_t band, std::size_t v) -> std::size_t {
    return ((t * N_PION_SPECIES + r) * N_PT_BANDS + band) * N_TRACK_VARS + v;
}
constexpr auto idx_res(std::size_t rv, std::size_t t, std::size_t r, std::size_t b) -> std::size_t {
    return ((rv * N_SPECIES + t) * N_SPECIES + r) * N_MOM_BINS + b;
}

// Bank index -> view (empty view if absent), reproducing Rust's Option<&Bank>.
auto bp(const bank_list& bl, long i) -> hipo::bank_view {
    return i < 0 ? hipo::bank_view{} : bl[i];
}

auto mk1(std::vector<std::unique_ptr<TH1D>>& v, const std::string& name, const std::string& title,
         int nb, double lo, double hi) -> void {
    v.push_back(std::make_unique<TH1D>(name.c_str(), title.c_str(), nb, lo, hi));
}
auto mk2(std::vector<std::unique_ptr<TH2D>>& v, const std::string& name, const std::string& title,
         int nx, double xlo, double xhi, int ny, double ylo, double yhi) -> void {
    v.push_back(std::make_unique<TH2D>(name.c_str(), title.c_str(), nx, xlo, xhi, ny, ylo, yhi));
}

}  // namespace

bank_index::bank_index(bank_list& bl) {
    auto g = [&](const char* n) -> long {
        try {
            return get_banklist_index(bl, n);
        } catch (const std::exception&) {
            return -1;
        }
    };
    rec_particle = g("REC::Particle");
    mc_particle = g("MC::Particle");
    mc_recmatch = g("MC::RecMatch");
    mc_true = g("MC::True");
    rec_track = g("REC::Track");
    rec_traj = g("REC::Traj");
    rec_covmat = g("REC::CovMat");
    rec_scint = g("REC::Scintillator");
    rec_scintx = g("REC::ScintExtras");
    rec_cher = g("REC::Cherenkov");
    rec_calo = g("REC::Calorimeter");
}

analysis::analysis() {
    // reco v_z / MC-truth v_z / Delta v_z, indexed idx_trp(truth, reco, mom).
    for (std::size_t t = 0; t < N_SPECIES; ++t) {
        for (std::size_t r = 0; r < N_SPECIES; ++r) {
            for (std::size_t b = 0; b < N_MOM_BINS; ++b) {
                p_bin pb = pbin(b);
                mk1(vz_,
                    fmt::format("vz_true{}_reco{}_p{:02}_{:02}", SPECIES_TAG[t], SPECIES_TAG[r],
                                pb.lo10, pb.hi10),
                    fmt::format("v_{{z}} truth {} / reco {}  {:.1f}<p_{{rec}}<{:.1f} "
                                "GeV;v_{{z}} [cm];counts",
                                SPECIES_TEX[t], SPECIES_TEX[r], pb.lo, pb.hi),
                    VZ_BINS, VZ_MIN, VZ_MAX);
                mk1(vz_true_,
                    fmt::format("vztrue_true{}_reco{}_p{:02}_{:02}", SPECIES_TAG[t], SPECIES_TAG[r],
                                pb.lo10, pb.hi10),
                    fmt::format("MC-truth v_{{z}} truth {} / reco {}  {:.1f}<p_{{rec}}<{:.1f} "
                                "GeV;v_{{z}} [cm];counts",
                                SPECIES_TEX[t], SPECIES_TEX[r], pb.lo, pb.hi),
                    VZ_BINS, VZ_MIN, VZ_MAX);
                mk1(dvz_,
                    fmt::format("dvz_true{}_reco{}_p{:02}_{:02}", SPECIES_TAG[t], SPECIES_TAG[r],
                                pb.lo10, pb.hi10),
                    fmt::format("#Deltav_{{z}}=reco-truth, truth {} / reco {}  "
                                "{:.1f}<p_{{rec}}<{:.1f} GeV;#Deltav_{{z}} [cm];counts",
                                SPECIES_TEX[t], SPECIES_TEX[r], pb.lo, pb.hi),
                    DVZ_BINS, DVZ_MIN, DVZ_MAX);
            }
        }
    }

    // v_z vs theta, indexed idx_tr(truth, reco).
    for (std::size_t t = 0; t < N_SPECIES; ++t) {
        for (std::size_t r = 0; r < N_SPECIES; ++r) {
            mk2(vz_theta_,
                fmt::format("vztheta_true{}_reco{}", SPECIES_TAG[t], SPECIES_TAG[r]),
                fmt::format("v_{{z}} vs #theta, truth {} / reco {};v_{{z}} [cm];#theta [deg]",
                            SPECIES_TEX[t], SPECIES_TEX[r]),
                VZ_BINS, VZ_MIN, VZ_MAX, THETA_BINS, THETA_MIN, THETA_MAX);
        }
    }

    // 1-D track-var distributions, indexed idx_var(truth, reco_pion, band, var).
    for (std::size_t t = 0; t < N_SPECIES; ++t) {
        for (std::size_t r = 0; r < N_PION_SPECIES; ++r) {
            for (const auto& band : PT_BANDS) {
                for (const auto& cv : TRACK_VARS) {
                    mk1(vars_,
                        fmt::format("var_true{}_reco{}_{}_{}", SPECIES_TAG[t], SPECIES_TAG[r],
                                    band.tag, cv.tag),
                        fmt::format("{} (truth {} / reco {}, {:.0f}<p<{:.0f} GeV);{};counts",
                                    cv.tag, SPECIES_TEX[t], SPECIES_TEX[r], band.lo, band.hi,
                                    cv.tag),
                        cv.nbins, cv.lo, cv.hi);
                }
            }
        }
    }

    // Resolution variables per fine momentum bin, full truth x reco grid.
    for (const auto& rvv : RES_VARS) {
        for (std::size_t t = 0; t < N_SPECIES; ++t) {
            for (std::size_t r = 0; r < N_SPECIES; ++r) {
                for (std::size_t b = 0; b < N_MOM_BINS; ++b) {
                    p_bin pb = pbin(b);
                    mk1(res_,
                        fmt::format("{}_true{}_reco{}_p{:02}_{:02}", rvv.tag, SPECIES_TAG[t],
                                    SPECIES_TAG[r], pb.lo10, pb.hi10),
                        fmt::format("{} truth {} / reco {}  {:.1f}<p<{:.1f} GeV;{};counts", rvv.tag,
                                    SPECIES_TEX[t], SPECIES_TEX[r], pb.lo, pb.hi, rvv.tag),
                        rvv.nbins, rvv.lo, rvv.hi);
                }
            }
        }
    }

    // Per reco pion: energy-loss / generated / swim / p@R1 histograms.
    for (std::size_t r = 0; r < N_PION_SPECIES; ++r) {
        const auto tag = SPECIES_TAG[r];
        const auto tex = SPECIES_TEX[r];
        mk2(ptrue_, fmt::format("ptrue_vtx_dcr1_reco{}", tag),
            fmt::format("true p: vertex vs DC R1, reco {};p_{{vtx}} [GeV];p_{{DC1}} [GeV]", tex),
            120, 0.0, 6.0, 120, 0.0, 6.0);
        mk1(dp_, fmt::format("dp_vtx_dcr1_reco{}", tag),
            fmt::format("#Deltap = p_{{vtx}} - p_{{DC1}} (true), reco {};#Deltap [GeV];counts", tex),
            120, -0.01, 0.15);
        mk2(dpvsp_, fmt::format("dp_vs_p_reco{}", tag),
            fmt::format("#Deltap vs p_{{vtx}} (true), reco {};p_{{vtx}} [GeV];#Deltap [GeV]", tex),
            120, 0.0, 6.0, 120, -0.01, 0.15);
        mk1(dpf_, fmt::format("dpf_vtx_dcr1_reco{}", tag),
            fmt::format("#Deltap/p_{{vtx}} (true), reco {};#Deltap/p_{{vtx}};counts", tex), 120,
            -0.005, 0.1);
        mk2(dpfvsp_, fmt::format("dpf_vs_p_reco{}", tag),
            fmt::format("#Deltap/p_{{vtx}} vs p_{{vtx}} (true), reco {};p_{{vtx}} "
                        "[GeV];#Deltap/p_{{vtx}}",
                        tex),
            120, 0.0, 6.0, 120, -0.005, 0.1);
        mk1(genp_, fmt::format("genp_{}", tag),
            fmt::format("generated momentum, {} (all MC::Particle primaries);p_{{gen}} "
                        "[GeV];counts",
                        tex),
            120, 0.0, 6.0);
        mk1(swum_vz_, fmt::format("swum_vz_reco{}", tag),
            fmt::format("swum v_{{z}} (DC R1 #rightarrow beamline), reco {};v_{{z}} [cm];counts",
                        tex),
            VZ_BINS, VZ_MIN, VZ_MAX);
        mk1(swum_minus_rec_, fmt::format("swum_minus_rec_reco{}", tag),
            fmt::format("v_{{z}}^{{swum}} - v_{{z}}^{{rec}}, reco {};#Deltav_{{z}} [cm];counts",
                        tex),
            SWUM_DVZ_BINS, SWUM_DVZ_MIN, SWUM_DVZ_MAX);
        mk2(swum_rec_vs_p_, fmt::format("swum_minus_rec_vs_p_reco{}", tag),
            fmt::format("v_{{z}}^{{swum}}-v_{{z}}^{{rec}} vs p, reco {};p [GeV];#Deltav_{{z}} [cm]",
                        tex),
            120, 0.0, 6.0, SWUM_DVZ_BINS, SWUM_DVZ_MIN, SWUM_DVZ_MAX);
        mk1(swum_minus_true_, fmt::format("swum_minus_true_reco{}", tag),
            fmt::format("v_{{z}}^{{swum}} - v_{{z}}^{{true}}, reco {};#Deltav_{{z}} [cm];counts",
                        tex),
            SWUM_DVZ_BINS, SWUM_DVZ_MIN, SWUM_DVZ_MAX);
        mk2(rec_vs_swum_, fmt::format("rec_vs_swum_reco{}", tag),
            fmt::format("v_{{z}}^{{rec}} vs v_{{z}}^{{swum}}, reco {};v_{{z}}^{{rec}} "
                        "[cm];v_{{z}}^{{swum}} [cm]",
                        tex),
            VZ_BINS, VZ_MIN, VZ_MAX, VZ_BINS, VZ_MIN, VZ_MAX);
        mk1(recoswum_vz_, fmt::format("recoswum_vz_reco{}", tag),
            fmt::format("reco-swum v_{{z}} (REC::Traj DC R1 #rightarrow beamline), reco "
                        "{};v_{{z}} [cm];counts",
                        tex),
            VZ_BINS, VZ_MIN, VZ_MAX);
        mk1(recoswum_minus_rec_, fmt::format("recoswum_minus_rec_reco{}", tag),
            fmt::format("v_{{z}}^{{reco-swum}} - v_{{z}}^{{rec}}, reco {};#Deltav_{{z}} "
                        "[cm];counts",
                        tex),
            SWUM_DVZ_BINS, SWUM_DVZ_MIN, SWUM_DVZ_MAX);
        mk1(recoswum_minus_true_, fmt::format("recoswum_minus_true_reco{}", tag),
            fmt::format("v_{{z}}^{{reco-swum}} - v_{{z}}^{{true}}, reco {};#Deltav_{{z}} "
                        "[cm];counts",
                        tex),
            SWUM_DVZ_BINS, SWUM_DVZ_MIN, SWUM_DVZ_MAX);
        mk2(rec_vs_recoswum_, fmt::format("rec_vs_recoswum_reco{}", tag),
            fmt::format("v_{{z}}^{{rec}} vs v_{{z}}^{{reco-swum}}, reco {};v_{{z}}^{{rec}} "
                        "[cm];v_{{z}}^{{reco-swum}} [cm]",
                        tex),
            VZ_BINS, VZ_MIN, VZ_MAX, VZ_BINS, VZ_MIN, VZ_MAX);
        mk2(pr1_reco_vs_true_, fmt::format("pr1_reco_vs_true_reco{}", tag),
            fmt::format("reco |p| vs true p at DC R1, reco {};p_{{DC1}}^{{true}} "
                        "[GeV];p^{{rec}} [GeV]",
                        tex),
            120, 0.0, 6.0, 120, 0.0, 6.0);
        mk1(pr1_reco_minus_true_, fmt::format("pr1_reco_minus_true_reco{}", tag),
            fmt::format("p^{{rec}} - p_{{DC1}}^{{true}}, reco {};#Deltap [GeV];counts", tex), 120,
            -0.3, 0.3);
        for (std::size_t b = 0; b < N_MOM_BINS; ++b) {
            p_bin pb = pbin(b);
            mk1(recvz_p_, fmt::format("recvz_reco{}_p{:02}_{:02}", tag, pb.lo10, pb.hi10),
                fmt::format("reco v_{{z}}, reco {}  {:.1f}<p<{:.1f} GeV;v_{{z}} [cm];counts", tex,
                            pb.lo, pb.hi),
                VZ_BINS, VZ_MIN, VZ_MAX);
            mk1(swumvz_p_, fmt::format("swumvz_reco{}_p{:02}_{:02}", tag, pb.lo10, pb.hi10),
                fmt::format("swum v_{{z}}, reco {}  {:.1f}<p<{:.1f} GeV;v_{{z}} [cm];counts", tex,
                            pb.lo, pb.hi),
                VZ_BINS, VZ_MIN, VZ_MAX);
            mk1(recoswumvz_p_, fmt::format("recoswumvz_reco{}_p{:02}_{:02}", tag, pb.lo10, pb.hi10),
                fmt::format("reco-swum v_{{z}}, reco {}  {:.1f}<p<{:.1f} GeV;v_{{z}} [cm];counts",
                            tex, pb.lo, pb.hi),
                VZ_BINS, VZ_MIN, VZ_MAX);
        }
    }
}

auto analysis::fill_event(const bank_list& bl, const bank_index& bi, const magnetic_field& field) -> void {
    events_ += 1;

    hipo::bank_view rec = bp(bl, bi.rec_particle);
    hipo::bank_view mc = bp(bl, bi.mc_particle);
    hipo::bank_view mtch = bp(bl, bi.mc_recmatch);
    if (!rec || !mc || !mtch) return;

    hipo::bank_view trk = bp(bl, bi.rec_track);
    hipo::bank_view traj = bp(bl, bi.rec_traj);
    hipo::bank_view cov = bp(bl, bi.rec_covmat);
    hipo::bank_view scint = bp(bl, bi.rec_scint);
    hipo::bank_view scintx = bp(bl, bi.rec_scintx);
    hipo::bank_view cher = bp(bl, bi.rec_cher);
    hipo::bank_view calo = bp(bl, bi.rec_calo);
    hipo::bank_view mctrue = bp(bl, bi.mc_true);

    const int rec_rows = static_cast<int>(rec.rows());
    const int mc_rows = static_cast<int>(mc.rows());

    // Generated-momentum spectrum: every primary pion in MC::Particle, with no
    // reconstruction / FD / DC requirement (acceptance cross-check).
    for (int i = 0; i < mc_rows; ++i) {
        auto s = species_index(mc.get<int>("pid", i));
        if (!s) continue;
        if (*s < N_PION_SPECIES) {
            double gx = mc.get<double>("px", i);
            double gy = mc.get<double>("py", i);
            double gz = mc.get<double>("pz", i);
            genp_[*s]->Fill(std::sqrt(gx * gx + gy * gy + gz * gz));
        }
    }

    const int mtch_rows = static_cast<int>(mtch.rows());
    for (int m = 0; m < mtch_rows; ++m) {
        const int pindex = mtch.get<int>("pindex", m);
        const int mcindex = mtch.get<int>("mcindex", m);
        if (pindex < 0 || pindex >= rec_rows) continue;
        if (mcindex < 0 || mcindex >= mc_rows) continue;

        const int truth_pid = mc.get<int>("pid", mcindex);
        auto truth_o = species_index(truth_pid);
        if (!truth_o) continue;
        auto reco_o = species_index(rec.get<int>("pid", pindex));
        if (!reco_o) continue;
        const std::size_t truth = *truth_o;
        const std::size_t reco = *reco_o;

        // Forward Detector only (reconstructed-track region).
        if (!is_forward(rec.get<int>("status", pindex))) continue;

        const double px = rec.get<double>("px", pindex);
        const double py = rec.get<double>("py", pindex);
        const double pz = rec.get<double>("pz", pindex);
        const double p = std::sqrt(px * px + py * py + pz * pz);
        if (p < MOM_MIN || p >= MOM_MAX) continue;
        const std::size_t mom_bin = static_cast<std::size_t>((p - MOM_MIN) / MOM_WIDTH);
        if (mom_bin >= N_MOM_BINS) continue;

        const double vz_rec = rec.get<double>("vz", pindex);
        const double vz_true = mc.get<double>("vz", mcindex);

        vz_[idx_trp(truth, reco, mom_bin)]->Fill(vz_rec);
        vz_true_[idx_trp(truth, reco, mom_bin)]->Fill(vz_true);
        dvz_[idx_trp(truth, reco, mom_bin)]->Fill(vz_rec - vz_true);
        fills_[truth][reco] += 1;

        const double theta =
            p > 0.0 ? std::acos(std::clamp(pz / p, -1.0, 1.0)) * RAD2DEG : 0.0;
        vz_theta_[idx_tr(truth, reco)]->Fill(vz_rec, theta);

        const cov_res cr = covariance_resolutions(cov, pindex, p, px, py, pz);
        const double res_vals[3] = {cr.sigtx, cr.sigty, cr.sigtheta};
        for (std::size_t rv = 0; rv < 3; ++rv) {
            if (!std::isnan(res_vals[rv])) res_[idx_res(rv, truth, reco, mom_bin)]->Fill(res_vals[rv]);
        }

        // True DC-region-1 state — energy loss + true-state swim.
        if (reco < N_PION_SPECIES) {
            if (auto st = mc_true_dc_state(mctrue, truth_pid)) {
                const auto& pos_cm = st->pos;
                const auto& mom_gev = st->mom;
                const double p_dcr1 = std::sqrt(mom_gev[0] * mom_gev[0] + mom_gev[1] * mom_gev[1] +
                                                mom_gev[2] * mom_gev[2]);
                const double pxm = mc.get<double>("px", mcindex);
                const double pym = mc.get<double>("py", mcindex);
                const double pzm = mc.get<double>("pz", mcindex);
                const double p_vtx = std::sqrt(pxm * pxm + pym * pym + pzm * pzm);
                const double dp = p_vtx - p_dcr1;
                ptrue_[reco]->Fill(p_vtx, p_dcr1);
                dp_[reco]->Fill(dp);
                dpvsp_[reco]->Fill(p_vtx, dp);
                pr1_reco_vs_true_[reco]->Fill(p_dcr1, p);
                pr1_reco_minus_true_[reco]->Fill(p - p_dcr1);
                if (p_vtx > 0.0) {
                    const double dpf = dp / p_vtx;
                    dpf_[reco]->Fill(dpf);
                    dpfvsp_[reco]->Fill(p_vtx, dpf);
                }

                const double q = (truth_pid == 211) ? 1.0 : -1.0;
                swim_result sw = swim_back_to_beamline(field, pos_cm, mom_gev, q);
                if (sw.status == swim_status::converged && sw.doca_rho < SWUM_MAX_DOCA_RHO) {
                    swum_vz_[reco]->Fill(sw.vz);
                    swum_minus_rec_[reco]->Fill(sw.vz - vz_rec);
                    swum_rec_vs_p_[reco]->Fill(p, sw.vz - vz_rec);
                    swum_minus_true_[reco]->Fill(sw.vz - vz_true);
                    rec_vs_swum_[reco]->Fill(vz_rec, sw.vz);
                    const std::size_t pidx = reco * N_MOM_BINS + mom_bin;
                    recvz_p_[pidx]->Fill(vz_rec);
                    swumvz_p_[pidx]->Fill(sw.vz);
                }
            }
        }

        // Reco-state swim: swim the reconstructed DC-R1 trajectory back.
        if (reco < N_PION_SPECIES) {
            if (auto rst = rec_traj_dc_state(traj, pindex, p)) {
                const double q = (reco == 0) ? 1.0 : -1.0;  // reco pi+ / pi-
                swim_result sw = swim_back_to_beamline(field, rst->pos, rst->mom, q);
                if (sw.status == swim_status::converged && sw.doca_rho < SWUM_MAX_DOCA_RHO) {
                    recoswum_vz_[reco]->Fill(sw.vz);
                    recoswum_minus_rec_[reco]->Fill(sw.vz - vz_rec);
                    recoswum_minus_true_[reco]->Fill(sw.vz - vz_true);
                    rec_vs_recoswum_[reco]->Fill(vz_rec, sw.vz);
                    recoswumvz_p_[reco * N_MOM_BINS + mom_bin]->Fill(sw.vz);
                }
            }
        }

        // 1-D track-variable distributions: reco pions in a momentum band only.
        if (reco < N_PION_SPECIES) {
            auto band_o = pt_band(p);
            if (band_o) {
                const std::size_t band = *band_o;
                double trkchi2 = DNAN;
                double trkndf = DNAN;
                if (trk) {
                    const int tn = static_cast<int>(trk.rows());
                    for (int k = 0; k < tn; ++k) {
                        if (trk.get<int>("pindex", k) == pindex) {
                            trkchi2 = trk.get<double>("chi2", k);
                            trkndf = static_cast<double>(trk.get<int>("NDF", k));
                            break;
                        }
                    }
                }

                const double pt = std::sqrt(px * px + py * py);
                const double phi = std::atan2(py, px) * RAD2DEG;
                const double trkredchi2 = (trkndf > 0.0) ? trkchi2 / trkndf : DNAN;
                const double beta_meas = rec.get<double>("beta", pindex);
                const double beta_exp = p / std::sqrt(p * p + PION_MASS * PION_MASS);
                const double dbeta = beta_meas - beta_exp;

                const double edge_dc1 =
                    layer_f64(traj, pindex, DET_DC, DC_LAYERS[0], "edge").value_or(DNAN);
                const double edge_dc2 =
                    layer_f64(traj, pindex, DET_DC, DC_LAYERS[1], "edge").value_or(DNAN);
                const double edge_dc3 =
                    layer_f64(traj, pindex, DET_DC, DC_LAYERS[2], "edge").value_or(DNAN);
                const double dc_path =
                    layer_f64(traj, pindex, DET_DC, DC_LAYERS[2], "path").value_or(DNAN);

                // REC::Scintillator (FTOF) matching + REC::ScintExtras dE/dx.
                double ftof_chi2 = DNAN;
                double ftof_dt = DNAN;
                double ftof_dedx = DNAN;
                auto ftof_k = match_row(scint, pindex, DET_FTOF, FTOF_PREF_LAYER);
                if (!ftof_k) ftof_k = match_row(scint, pindex, DET_FTOF, std::nullopt);
                if (ftof_k) {
                    if (scint) {
                        ftof_chi2 = scint.get<double>("chi2", *ftof_k);
                        const double time = scint.get<double>("time", *ftof_k);
                        const double path = scint.get<double>("path", *ftof_k);
                        const double vt = rec.get<double>("vt", pindex);
                        const double beta_pi = p / std::sqrt(p * p + PION_MASS * PION_MASS);
                        ftof_dt = time - vt - path / (beta_pi * C_CM_PER_NS);
                    }
                    if (scintx) {
                        ftof_dedx = scintx.get<double>("dedx", *ftof_k);
                    }
                }

                const double htcc_nphe = first_f64(cher, pindex, DET_HTCC, "nphe").value_or(DNAN);
                auto ecal_e_o = sum_f64(calo, pindex, DET_ECAL, "energy");
                const double ecal_e = ecal_e_o.value_or(DNAN);
                const double ecal_sf = std::isnan(ecal_e) ? DNAN : ecal_e / p;

                // Order MUST match TRACK_VARS in constants.hpp.
                const double vals[N_TRACK_VARS] = {
                    p,
                    pt,
                    theta,
                    phi,
                    rec.get<double>("vx", pindex),
                    rec.get<double>("vy", pindex),
                    vz_rec,
                    dbeta,
                    rec.get<double>("chi2pid", pindex),
                    rec.get<double>("vt", pindex),
                    trkchi2,
                    trkndf,
                    trkredchi2,
                    edge_dc1,
                    edge_dc2,
                    edge_dc3,
                    dc_path,
                    cr.sigp,
                    cr.sigtx,
                    cr.sigty,
                    cr.sigtheta,
                    cr.sigphi,
                    ftof_chi2,
                    ftof_dt,
                    ftof_dedx,
                    htcc_nphe,
                    ecal_e,
                    ecal_sf,
                };
                for (std::size_t v = 0; v < N_TRACK_VARS; ++v) {
                    if (!std::isnan(vals[v])) vars_[idx_var(truth, reco, band, v)]->Fill(vals[v]);
                }
            }
        }
    }
}

auto analysis::merge_from(const analysis& o) -> void {
    auto add1 = [](std::vector<std::unique_ptr<TH1D>>& a,
                   const std::vector<std::unique_ptr<TH1D>>& b) {
        for (std::size_t i = 0; i < a.size(); ++i) a[i]->Add(b[i].get(), 1.0);
    };
    auto add2 = [](std::vector<std::unique_ptr<TH2D>>& a,
                   const std::vector<std::unique_ptr<TH2D>>& b) {
        for (std::size_t i = 0; i < a.size(); ++i) a[i]->Add(b[i].get(), 1.0);
    };
    add1(vz_, o.vz_);
    add1(vz_true_, o.vz_true_);
    add1(dvz_, o.dvz_);
    add2(vz_theta_, o.vz_theta_);
    add1(vars_, o.vars_);
    add1(res_, o.res_);
    add2(ptrue_, o.ptrue_);
    add1(dp_, o.dp_);
    add2(dpvsp_, o.dpvsp_);
    add1(dpf_, o.dpf_);
    add2(dpfvsp_, o.dpfvsp_);
    add1(genp_, o.genp_);
    add1(swum_vz_, o.swum_vz_);
    add1(swum_minus_rec_, o.swum_minus_rec_);
    add2(swum_rec_vs_p_, o.swum_rec_vs_p_);
    add1(swum_minus_true_, o.swum_minus_true_);
    add2(rec_vs_swum_, o.rec_vs_swum_);
    add1(recvz_p_, o.recvz_p_);
    add1(swumvz_p_, o.swumvz_p_);
    add1(recoswum_vz_, o.recoswum_vz_);
    add1(recoswum_minus_rec_, o.recoswum_minus_rec_);
    add1(recoswum_minus_true_, o.recoswum_minus_true_);
    add2(rec_vs_recoswum_, o.rec_vs_recoswum_);
    add1(recoswumvz_p_, o.recoswumvz_p_);
    add2(pr1_reco_vs_true_, o.pr1_reco_vs_true_);
    add1(pr1_reco_minus_true_, o.pr1_reco_minus_true_);

    events_ += o.events_;
    for (std::size_t t = 0; t < N_SPECIES; ++t)
        for (std::size_t r = 0; r < N_SPECIES; ++r) fills_[t][r] += o.fills_[t][r];
}

auto analysis::write(const std::string& path) const -> void {
    TFile f(path.c_str(), "RECREATE");
    f.SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kZSTD);
    f.SetCompressionLevel(5);
    f.cd();
    auto w1 = [](const std::vector<std::unique_ptr<TH1D>>& v) {
        for (const auto& h : v) h->Write();
    };
    auto w2 = [](const std::vector<std::unique_ptr<TH2D>>& v) {
        for (const auto& h : v) h->Write();
    };
    w1(vz_);
    w1(vz_true_);
    w1(dvz_);
    w2(vz_theta_);
    w1(vars_);
    w1(res_);
    w2(ptrue_);
    w1(dp_);
    w2(dpvsp_);
    w1(dpf_);
    w2(dpfvsp_);
    w1(genp_);
    w1(swum_vz_);
    w1(swum_minus_rec_);
    w2(swum_rec_vs_p_);
    w1(swum_minus_true_);
    w2(rec_vs_swum_);
    w1(recvz_p_);
    w1(swumvz_p_);
    w1(recoswum_vz_);
    w1(recoswum_minus_rec_);
    w1(recoswum_minus_true_);
    w2(rec_vs_recoswum_);
    w1(recoswumvz_p_);
    w2(pr1_reco_vs_true_);
    w1(pr1_reco_minus_true_);
    f.Close();
}

}  // namespace vz
