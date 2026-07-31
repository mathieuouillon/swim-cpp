#!/usr/bin/env python3
"""Compare the swum-vz peak position vs theta, per momentum bin, WITH and WITHOUT
the empirical (p, theta) vz correction -- at a single (nominal) field setting.

The reconstructed/swum electron vz of the fixed RG-D target foils drifts with
polar angle theta (a cot(theta) miss-distance walk, larger at low p). The
correction vz += (c0 + c1/p)*cot(theta) (vz-swim-hist --vz-cot-coeff/-p-coeff) is
meant to flatten that walk. This script reads two vz_bins.root files from the
SAME run config -- one uncorrected, one corrected -- fits the 2nd (downstream,
~ -3 cm) swum-vz peak in every (p, theta) cell (scan_field.fit_peak2_gauss), and
plots its mean (page 1) and width (page 2) versus theta, one panel per momentum
bin, uncorrected as solid circles and corrected as dashed crosses. A flat
corrected curve across theta = the correction worked in that (p) bin; the per-p
theta-spread (std over theta) before/after is printed and annotated.

Usage (from the repo root):
  python plot_vz_theta_compare.py \
      output/python/electron_018354/torus-scale_scan/torus-scale_-1.00/vz_bins.root \
      output/python/electron_018354/torus-scale_scan_corr/torus-scale_-1.00/vz_bins.root \
      -o output/python/electron_018354/vz_theta_correction_compare.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import uproot
import matplotlib as mpl
mpl.use("Agg")  # headless: write a PDF only
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

# Reuse the exact grid, key format and peak2 fitter the scan uses, so the fits
# here match torus-scale_peak2_fit.* bin-for-bin.
from scan_field import (
    P_BINS, TH_BINS, PEAK_WINDOWS, MIN_ENTRIES, _key,
    fit_peak2_gauss, momentum_edges, theta_edges,
)


def fit_file(root_file: Path):
    """peak2 (mean, sigma, mean_err, sigma_err) per (ip, it) cell; NaN when the
    cell is empty / below MIN_ENTRIES / the fit fails."""
    _, lo2, hi2 = PEAK_WINDOWS[1]  # ("peak2", -5.5, -0.5)
    n_p, n_th = len(P_BINS), len(TH_BINS)
    mean = np.full((n_p, n_th), np.nan)
    sigma = np.full((n_p, n_th), np.nan)
    mean_e = np.full((n_p, n_th), np.nan)
    sigma_e = np.full((n_p, n_th), np.nan)
    with uproot.open(root_file) as f:
        for ip, (p_lo, p_hi) in enumerate(P_BINS):
            for it, (t_lo, t_hi) in enumerate(TH_BINS):
                key = _key("vz_swum", p_lo, p_hi, t_lo, t_hi)
                if key not in f:
                    continue
                hv, edges = f[key].to_numpy()
                if hv.sum() < MIN_ENTRIES:
                    continue
                centers = 0.5 * (edges[:-1] + edges[1:])
                ft = fit_peak2_gauss(hv, centers, lo2, hi2)
                if ft.ok:
                    mean[ip, it] = ft.mean
                    sigma[ip, it] = ft.sigma
                    mean_e[ip, it] = ft.mean_err
                    sigma_e[ip, it] = ft.sigma_err
    return dict(mean=mean, sigma=sigma, mean_err=mean_e, sigma_err=sigma_e)


def theta_spread(mat: np.ndarray) -> np.ndarray:
    """Per-momentum-bin std of the peak position across theta (ignoring NaNs)."""
    with np.errstate(invalid="ignore"):
        return np.array([np.nanstd(row) if np.isfinite(row).sum() >= 2 else np.nan
                         for row in mat])


def draw_page(pdf, before, after, *, value_key, err_key, ylabel, title,
              before_label, after_label, annotate_spread):
    """One page: peak2 <value> vs theta, one panel per momentum bin, before
    solid/o and after dashed/x."""
    th_mid = np.array([0.5 * (t_lo + t_hi) for t_lo, t_hi in TH_BINS])
    n_p = len(P_BINS)
    pncol = 4
    pnrow = (n_p + pncol - 1) // pncol
    fig, axs = plt.subplots(pnrow, pncol, figsize=(3.3 * pncol, 2.9 * pnrow),
                            squeeze=False, sharex=True)
    sb = theta_spread(before[value_key])
    sa = theta_spread(after[value_key])
    for ip, (p_lo, p_hi) in enumerate(P_BINS):
        ax = axs[ip // pncol][ip % pncol]
        for src, style, col in ((before, dict(ls="-", marker="o"), "tab:blue"),
                                (after, dict(ls="--", marker="x"), "tab:red")):
            y = src[value_key][ip]
            ye = src[err_key][ip]
            m = np.isfinite(y)
            if m.any():
                ax.errorbar(th_mid[m], y[m], yerr=ye[m], ms=4, lw=1.0, capsize=1.5,
                            elinewidth=0.6, color=col, **style)
        ax.set_title(rf"${p_lo:g} < p < {p_hi:g}$ GeV/$c$", fontsize=9)
        ax.tick_params(labelsize=7)
        if annotate_spread and np.isfinite(sb[ip]) and np.isfinite(sa[ip]):
            ax.text(0.03, 0.05, rf"std$_\theta$: {sb[ip]:.2f}$\to${sa[ip]:.2f} cm",
                    transform=ax.transAxes, fontsize=7, color="0.25",
                    va="bottom", ha="left")
        if ip // pncol == pnrow - 1:
            ax.set_xlabel(r"$\theta$ [deg]", fontsize=8)
        if ip % pncol == 0:
            ax.set_ylabel(ylabel, fontsize=8)
    for j in range(n_p, pnrow * pncol):
        axs[j // pncol][j % pncol].set_visible(False)

    style_handles = [
        Line2D([], [], color="tab:blue", ls="-", marker="o", ms=5, label=before_label),
        Line2D([], [], color="tab:red", ls="--", marker="x", ms=5, label=after_label),
    ]
    fig.legend(handles=style_handles, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return sb, sa


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before_root", type=Path, help="uncorrected vz_bins.root")
    ap.add_argument("after_root", type=Path, help="corrected vz_bins.root (same run config)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output PDF (default: <before dir>/vz_theta_correction_compare.pdf)")
    ap.add_argument("--before-label", default="without correction")
    ap.add_argument("--after-label", default="with correction")
    ap.add_argument("--species", default=None, help="title prefix (default: inferred from path)")
    ap.add_argument("--pid", type=int, default=11,
                    help="species pid, to pick the (p, theta) grid (default 11 = e-)")
    args = ap.parse_args()

    # Adopt the species (p, theta) grid so the keys/fits match vz-swim-hist.
    import scan_field
    scan_field.P_EDGES = momentum_edges(args.pid)
    scan_field.P_BINS[:] = list(zip(scan_field.P_EDGES[:-1], scan_field.P_EDGES[1:]))
    scan_field.TH_EDGES = theta_edges(args.pid)
    scan_field.TH_BINS[:] = list(zip(scan_field.TH_EDGES[:-1], scan_field.TH_EDGES[1:]))

    before = fit_file(args.before_root)
    after = fit_file(args.after_root)

    species = args.species
    if species is None:
        parts = args.before_root.resolve().parts
        species = parts[parts.index("python") + 1] if "python" in parts else ""

    out = args.output or (args.before_root.parent / "vz_theta_correction_compare.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        sb, sa = draw_page(
            pdf, before, after, value_key="mean", err_key="mean_err",
            ylabel=r"peak2 $\mu$ [cm]",
            title=rf"{species} — 2nd-peak mean vs $\theta$, with/without vz correction",
            before_label=args.before_label, after_label=args.after_label,
            annotate_spread=True)
        draw_page(
            pdf, before, after, value_key="sigma", err_key="sigma_err",
            ylabel=r"peak2 $\sigma$ [cm]",
            title=rf"{species} — 2nd-peak width vs $\theta$, with/without vz correction",
            before_label=args.before_label, after_label=args.after_label,
            annotate_spread=False)
    print(f"Saved {out}")
    # Quantify: mean over p of the theta-spread, before vs after.
    mb, ma = np.nanmean(sb), np.nanmean(sa)
    print(f"peak2 mean theta-spread (std over theta, averaged over p): "
          f"{mb:.3f} cm  ->  {ma:.3f} cm")
    for ip, (p_lo, p_hi) in enumerate(P_BINS):
        if np.isfinite(sb[ip]) and np.isfinite(sa[ip]):
            print(f"  {p_lo:g}-{p_hi:g} GeV: {sb[ip]:.3f} -> {sa[ip]:.3f} cm")


if __name__ == "__main__":
    main()
