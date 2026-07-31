#!/usr/bin/env python3
"""Presentation-quality figures for the (p, theta) vz correction study.

Reads two vz_bins.root files from the SAME nominal run config -- one WITHOUT and
one WITH the empirical correction vz += (c0 + c1/p)*cot(theta) (vz-swim-hist
--vz-cot-coeff/-p-coeff) -- and produces four slide-ready figures (PNG + a
combined PDF):

  fig1_peaks_aligned   swum-vz distributions vs theta, without | with correction
                       (a few momentum rows): the target peaks line up.
  fig2_peak_walk       peak2 mean vs theta per momentum bin, before vs after,
                       with the per-bin theta-spread annotated.
  fig3_model           the fitted walk a(p) = a0 + a1/p (geometric + magnetic),
                       the basis of the correction.
  fig4_spread_summary  theta-spread per momentum bin, before vs after (bars).

Usage (from the repo root, on ifarm):
  python plot_correction_presentation.py \
      output/python/electron_018354/torus-scale_scan/torus-scale_-1.00/vz_bins.root \
      output/python/electron_018354/corrfit/vz_bins.root \
      --outdir output/python/electron_018354/presentation \
      --run 18354
"""
import argparse
from pathlib import Path

import numpy as np
import uproot
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe

from scan_field import (
    P_BINS, TH_BINS, PEAK_WINDOWS, MIN_ENTRIES, _key, fit_peak2_gauss,
    momentum_edges, theta_edges,
)

# ── House style ──────────────────────────────────────────────────────────────
BLUE, RED = "#2166AC", "#B2182B"          # colourblind-safe before/after pair
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 220,
    "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
    "axes.titleweight": "medium",
    "legend.fontsize": 11, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
})
THETA_CMAP = plt.cm.viridis


def _save(fig, outdir: Path, name: str, pdf: PdfPages):
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  {outdir/name}.png")


def load_cells(root_file: Path, prefix: str = "vz_swum"):
    """{(ip, it): (counts, centers)} for the `prefix` vz histograms (vz_swum or
    vz_rec), plus peak2 fits."""
    _, lo2, hi2 = PEAK_WINDOWS[1]
    n_p, n_th = len(P_BINS), len(TH_BINS)
    hists: dict = {}
    mean = np.full((n_p, n_th), np.nan)
    mean_e = np.full((n_p, n_th), np.nan)
    with uproot.open(root_file) as f:
        for ip, (p_lo, p_hi) in enumerate(P_BINS):
            for it, (t_lo, t_hi) in enumerate(TH_BINS):
                key = _key(prefix, p_lo, p_hi, t_lo, t_hi)
                if key not in f:
                    continue
                hv, edges = f[key].to_numpy()
                centers = 0.5 * (edges[:-1] + edges[1:])
                hists[(ip, it)] = (hv, centers)
                if hv.sum() >= MIN_ENTRIES:
                    ft = fit_peak2_gauss(hv, centers, lo2, hi2)
                    if ft.ok:
                        mean[ip, it] = ft.mean
                        mean_e[ip, it] = ft.mean_err
    return hists, mean, mean_e


def theta_spread(mat):
    with np.errstate(invalid="ignore"):
        return np.array([np.nanstd(r) if np.isfinite(r).sum() >= 2 else np.nan for r in mat])


def fit_a_of_p(mean):
    """a(p) walk coefficient per p (slope of mean vs cot theta) and the
    a(p)=a0+a1/p fit. Returns (p_mid, a, a0, a1)."""
    th_mid = np.array([0.5 * (a + b) for a, b in TH_BINS])
    cot = 1.0 / np.tan(np.radians(th_mid))
    p_mid = np.array([0.5 * (a + b) for a, b in P_BINS])
    a = np.full(len(P_BINS), np.nan)
    for ip in range(len(P_BINS)):
        m = np.isfinite(mean[ip])
        if m.sum() >= 3:
            a[ip], _ = np.polyfit(cot[m], mean[ip][m], 1)
    ok = np.isfinite(a)
    a1, a0 = np.polyfit(1.0 / p_mid[ok], a[ok], 1)
    return p_mid, a, a0, a1


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_peaks_aligned(before_h, after_h, outdir, pdf, run, rows=(0, 3, 6)):
    """swum-vz distributions overlaid over theta, without | with correction, for
    a few momentum rows. The peaks aligning after correction is the headline."""
    rows = [ip for ip in rows if ip < len(P_BINS)]
    nrow = len(rows)
    fig, axs = plt.subplots(nrow, 2, figsize=(11, 2.6 * nrow + 0.6),
                            squeeze=False, sharex=True)
    norm = Normalize(vmin=TH_BINS[0][0], vmax=TH_BINS[-1][1])
    for r, ip in enumerate(rows):
        p_lo, p_hi = P_BINS[ip]
        for col, hs in enumerate((before_h, after_h)):
            ax = axs[r][col]
            for it, (t_lo, t_hi) in enumerate(TH_BINS):
                cell = hs.get((ip, it))
                if cell is None:
                    continue
                hv, centers = cell
                if hv.sum() < MIN_ENTRIES:
                    continue
                area = hv.sum() * (centers[1] - centers[0])
                ax.plot(centers, hv / area, lw=1.3,
                        color=THETA_CMAP(norm(0.5 * (t_lo + t_hi))))
            ax.set_xlim(-9, -0.5)
            ax.margins(y=0.02)
            if r == 0:
                ax.set_title(["without correction", "with correction"][col],
                             color=[BLUE, RED][col], fontweight="bold")
            if col == 0:
                ax.set_ylabel(rf"${p_lo:g}\!<\!p\!<\!{p_hi:g}$ GeV/$c$" "\nnormalized",
                              fontsize=11)
            if r == nrow - 1:
                ax.set_xlabel(r"swum $v_z$ [cm]")
    sm = ScalarMappable(norm=norm, cmap=THETA_CMAP); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, fraction=0.025, pad=0.02)
    cbar.set_label(r"$\theta$ [deg]")
    fig.suptitle(f"Electron run {run}: the correction aligns the target-foil "
                 r"$v_z$ peaks across $\theta$", fontsize=14, fontweight="bold")
    _save(fig, outdir, "fig1_peaks_aligned", pdf)


def _vz_theta_map(hists):
    """Build a (theta x vz) map by summing the per-(p, theta) swum-vz histograms
    over momentum. Returns (M, vz_centers, th_edges) with M[it] = sum_ip counts."""
    n_th = len(TH_BINS)
    centers = None
    rows = [None] * n_th
    for (ip, it), (hv, c) in hists.items():
        centers = c
        rows[it] = hv if rows[it] is None else rows[it] + hv
    n_vz = len(centers)
    M = np.full((n_th, n_vz), np.nan)
    for it in range(n_th):
        if rows[it] is not None and rows[it].sum() > 0:
            M[it] = rows[it]
    th_edges = np.array([TH_BINS[0][0]] + [b for _, b in TH_BINS])
    return M, centers, th_edges


def fit_peak2_per_theta(hists):
    """Per theta slice (summed over p): downstream-peak mean(theta) and
    sigma(theta) from a Gaussian-on-linear fit. Arrays over TH_BINS (NaN when a
    slice is under MIN_ENTRIES or the fit fails)."""
    _, lo2, hi2 = PEAK_WINDOWS[1]
    n_th = len(TH_BINS)
    rows = [None] * n_th
    centers = None
    for (ip, it), (hv, c) in hists.items():
        centers = c
        rows[it] = hv if rows[it] is None else rows[it] + hv
    mean_th = np.full(n_th, np.nan)
    sig_th = np.full(n_th, np.nan)
    for it in range(n_th):
        if rows[it] is None or rows[it].sum() < MIN_ENTRIES:
            continue
        ft = fit_peak2_gauss(rows[it], centers, lo2, hi2)
        if ft.ok:
            mean_th[it], sig_th[it] = ft.mean, ft.sigma
    return mean_th, sig_th


def fig_vz_theta_2d(before_h, after_h, before_m, after_m, outdir, pdf, run):
    """Two 2-D v_z-vs-theta maps (without | with correction). Each theta row is
    area-normalized so the peak *positions* drive the colour. Per theta slice the
    downstream peak is fit for mean(theta) & sigma(theta); mean and mean +/- 3 sigma
    points are overlaid and each is described by a 3rd-order polynomial in theta
    (mu(theta) and sigma(theta) fit with cubics, band = mu_fit +/- 3 sigma_fit)."""
    th_mid = np.array([0.5 * (a + b) for a, b in TH_BINS])
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    _, centers, th_edges = _vz_theta_map(before_h)
    vz_edges = np.concatenate(([centers[0] - 0.5 * (centers[1] - centers[0])],
                               0.5 * (centers[:-1] + centers[1:]),
                               [centers[-1] + 0.5 * (centers[1] - centers[0])]))
    stroke = [pe.Stroke(linewidth=3.0, foreground="black"), pe.Normal()]
    stroke_thin = [pe.Stroke(linewidth=2.6, foreground="black"), pe.Normal()]
    im = None
    coeffs = {}
    for ax, hists, ttl, col in ((axs[0], before_h, "without correction", BLUE),
                                (axs[1], after_h, "with correction", RED)):
        M, _, _ = _vz_theta_map(hists)
        with np.errstate(invalid="ignore"):
            row_area = np.nansum(M, axis=1, keepdims=True) * (centers[1] - centers[0])
            Mn = M / np.where(row_area > 0, row_area, np.nan)
        im = ax.pcolormesh(vz_edges, th_edges, Mn, cmap="viridis", shading="flat")

        # Per-theta downstream-peak mean & sigma, then cubic fits in theta.
        mth, sth = fit_peak2_per_theta(hists)
        ok = np.isfinite(mth) & np.isfinite(sth)
        pm = np.polyfit(th_mid[ok], mth[ok], 3)
        ps = np.polyfit(th_mid[ok], sth[ok], 3)
        coeffs[ttl] = (pm, ps)
        tf = np.linspace(th_mid[ok].min(), th_mid[ok].max(), 200)
        mfit, sfit = np.polyval(pm, tf), np.polyval(ps, tf)

        # Measured points: mean (filled) and mean +/- 3 sigma (open).
        ax.plot(mth[ok], th_mid[ok], "o", color="white", mec="black", mew=0.8,
                ms=4.5, zorder=6, path_effects=stroke_thin)
        ax.plot(mth[ok] - 3 * sth[ok], th_mid[ok], "o", ms=3, mfc="none",
                mec="white", mew=1.0, zorder=6, path_effects=stroke_thin)
        ax.plot(mth[ok] + 3 * sth[ok], th_mid[ok], "o", ms=3, mfc="none",
                mec="white", mew=1.0, zorder=6, path_effects=stroke_thin)
        # Cubic curves: mean (solid) and the +/- 3 sigma cut band (dashed).
        ax.plot(mfit, tf, "-", color="white", lw=2.0, zorder=5, path_effects=stroke)
        ax.plot(mfit - 3 * sfit, tf, "--", color="white", lw=1.6, zorder=5, path_effects=stroke_thin)
        ax.plot(mfit + 3 * sfit, tf, "--", color="white", lw=1.6, zorder=5, path_effects=stroke_thin)

        ax.set_xlim(-9, -0.5)
        ax.set_xlabel(r"reconstructed $v_z$ [cm]")
        ax.set_title(ttl, color=col, fontweight="bold")
    axs[0].set_ylabel(r"$\theta$ [deg]")
    # Legend for the overlay (proxy handles).
    axs[1].plot([], [], "-", color="white", lw=2.0, path_effects=stroke, label=r"$\mu(\theta)$ cubic")
    axs[1].plot([], [], "--", color="white", lw=1.6, path_effects=stroke_thin,
                label=r"$\mu\pm3\sigma$ cubic")
    axs[1].legend(loc="lower left", fontsize=9, labelcolor="white",
                  facecolor="0.2", framealpha=0.6)
    cbar = fig.colorbar(im, ax=axs, fraction=0.04, pad=0.02)
    cbar.set_label(r"normalized / $\theta$-row")
    fig.suptitle(rf"Electron run {run}: reconstructed $v_z$ vs $\theta$ with "
                 r"$\mu(\theta)\pm3\sigma$ vertex band (cubic fit)",
                 fontsize=14, fontweight="bold")
    _save(fig, outdir, "fig5_vz_theta_2d", pdf)
    # Report the cubic coefficients (lowest order last, np.polyfit order).
    for ttl, (pm, ps) in coeffs.items():
        print(f"  [{ttl}] mu(theta)  cubic (hi->lo): {np.array2string(pm, precision=5)}")
        print(f"  [{ttl}] sigma(theta) cubic (hi->lo): {np.array2string(ps, precision=5)}")


def fig_peak_walk(before_m, before_e, after_m, after_e, outdir, pdf, run):
    """peak2 mean vs theta per momentum bin, before vs after, spread annotated."""
    th_mid = np.array([0.5 * (a + b) for a, b in TH_BINS])
    sb, sa = theta_spread(before_m), theta_spread(after_m)
    n_p = len(P_BINS); ncol = 4; nrow = (n_p + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow),
                            squeeze=False, sharex=True)
    for ip, (p_lo, p_hi) in enumerate(P_BINS):
        ax = axs[ip // ncol][ip % ncol]
        for m, e, col, lab, mk, ls in (
                (before_m, before_e, BLUE, "without", "o", "-"),
                (after_m, after_e, RED, "with", "D", "--")):
            y, ye = m[ip], e[ip]
            ok = np.isfinite(y)
            if ok.any():
                ax.errorbar(th_mid[ok], y[ok], yerr=ye[ok], color=col, marker=mk,
                            ms=4.5, lw=1.4, ls=ls, capsize=2, elinewidth=0.7,
                            label=lab if ip == 0 else None)
        ax.set_title(rf"${p_lo:g}\!<\!p\!<\!{p_hi:g}$ GeV/$c$")
        if np.isfinite(sb[ip]) and np.isfinite(sa[ip]):
            ax.text(0.96, 0.94,
                    rf"$\sigma_\theta:\ {sb[ip]:.2f}\to{sa[ip]:.2f}$ cm",
                    transform=ax.transAxes, fontsize=9.5, color="0.25",
                    ha="right", va="top")
        if ip // ncol == nrow - 1:
            ax.set_xlabel(r"$\theta$ [deg]")
        if ip % ncol == 0:
            ax.set_ylabel(r"peak $v_z$ [cm]")
    for j in range(n_p, nrow * ncol):
        axs[j // ncol][j % ncol].set_visible(False)
    handles = [Line2D([], [], color=BLUE, marker="o", lw=1.6, label="without correction"),
               Line2D([], [], color=RED, marker="D", ls="--", lw=1.6, label="with correction")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(rf"Electron run {run}: downstream-foil $v_z$ vs $\theta$ "
                 r"flattens after correction (all $p$)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(fig, outdir, "fig2_peak_walk", pdf)
    return sb, sa


def fig_model(before_m, outdir, pdf, run):
    """a(p) = a0 + a1/p: the fitted cot-theta walk coefficient vs momentum."""
    p_mid, a, a0, a1 = fit_a_of_p(before_m)
    ok = np.isfinite(a)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    pp = np.linspace(p_mid[ok].min(), p_mid[ok].max(), 200)
    ax.axhline(a0, color="0.6", ls=":", lw=1.3)
    ax.plot(pp, a0 + a1 / pp, color=RED, lw=2.2, zorder=2,
            label=rf"$a(p)=a_0+a_1/p$")
    ax.plot(p_mid[ok], a[ok], "o", color=BLUE, ms=8, zorder=3, label="measured")
    ax.text(0.30, 0.20, r"$a_0$: geometric floor (DC alignment)",
            transform=ax.transAxes, fontsize=10.5, color="0.45", va="bottom")
    ax.set_xlabel(r"$p$ [GeV/$c$]")
    ax.set_ylabel(r"walk coefficient $a$  ($\Delta v_z = a\,\cot\theta$)  [cm]")
    ax.legend(loc="upper right")
    txt = (rf"$a_0={a0:+.3f}$ cm,   $a_1={a1:+.3f}$ cm$\cdot$GeV" "\n"
           rf"correction: $v_z \to v_z-({a0:+.3f}{a1:+.3f}/p)\cot\theta$")
    ax.text(0.03, 0.05, txt, transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", fc="#f2f2f2", ec="0.7"))
    fig.suptitle(rf"Electron run {run}: the $v_z$ walk is $\propto\cot\theta$ "
                 r"with $a(p)=a_0+a_1/p$", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir, "fig3_model", pdf)


def fig_spread_summary(sb, sa, outdir, pdf, run):
    """theta-spread per momentum bin, before vs after (grouped bars)."""
    labels = [rf"{p_lo:g}-{p_hi:g}" for p_lo, p_hi in P_BINS]
    x = np.arange(len(P_BINS)); w = 0.38
    ok = np.isfinite(sb) & np.isfinite(sa)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x[ok] - w / 2, sb[ok], w, color=BLUE, label="without correction")
    ax.bar(x[ok] + w / 2, sa[ok], w, color=RED, label="with correction")
    for xi, v in zip(x[ok], sa[ok]):
        ax.text(xi + w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=RED)
    mb, ma = np.nanmean(sb), np.nanmean(sa)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel(r"momentum bin [GeV/$c$]")
    ax.set_ylabel(r"$v_z$ peak spread over $\theta$  (std) [cm]")
    ax.legend(loc="upper right")
    ax.set_title(rf"Electron run {run}: $\theta$-spread reduced "
                 rf"{mb:.2f}$\to${ma:.2f} cm (mean over $p$, {mb/ma:.1f}$\times$)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir, "fig4_spread_summary", pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before_root", type=Path)
    ap.add_argument("after_root", type=Path)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--run", default="")
    ap.add_argument("--pid", type=int, default=11)
    ap.add_argument("--rows", type=int, nargs="+", default=[0, 3, 6],
                    help="momentum-bin indices to show in fig1 (default 0 3 6)")
    args = ap.parse_args()

    import scan_field
    scan_field.P_EDGES = momentum_edges(args.pid)
    scan_field.P_BINS[:] = list(zip(scan_field.P_EDGES[:-1], scan_field.P_EDGES[1:]))
    scan_field.TH_EDGES = theta_edges(args.pid)
    scan_field.TH_BINS[:] = list(zip(scan_field.TH_EDGES[:-1], scan_field.TH_EDGES[1:]))

    outdir = args.outdir or (args.before_root.parent / "presentation")
    outdir.mkdir(parents=True, exist_ok=True)

    before_h, before_m, before_e = load_cells(args.before_root)
    after_h, after_m, after_e = load_cells(args.after_root)
    # Reconstructed-vz cells for the 2-D v_z-vs-theta maps (the correction is
    # applied to rec as well as swum in vz-swim-hist).
    before_hr, before_mr, _ = load_cells(args.before_root, "vz_rec")
    after_hr, after_mr, _ = load_cells(args.after_root, "vz_rec")

    print("Saved:")
    with PdfPages(outdir / "vz_correction_presentation.pdf") as pdf:
        fig_peaks_aligned(before_h, after_h, outdir, pdf, args.run, tuple(args.rows))
        fig_vz_theta_2d(before_hr, after_hr, before_mr, after_mr, outdir, pdf, args.run)
        sb, sa = fig_peak_walk(before_m, before_e, after_m, after_e, outdir, pdf, args.run)
        fig_model(before_m, outdir, pdf, args.run)
        fig_spread_summary(sb, sa, outdir, pdf, args.run)
    print(f"  {outdir/'vz_correction_presentation.pdf'}  (5 pages)")


if __name__ == "__main__":
    main()
