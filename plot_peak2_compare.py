#!/usr/bin/env python3
"""Overlay the 2nd-peak (downstream, ~ -3 cm) swum-vz fit vs a scanned field
parameter BEFORE and AFTER the empirical (p, theta) vz correction.

Reads two `<param>_peak2_fit.csv` files written by scan_field.py -- one from the
uncorrected scan, one from the same scan re-run with vz-swim-hist's
`--vz-cot-coeff / --vz-cot-p-coeff` (vz += (c0 + c1/p)*cot(theta)). Draws the
same layout as scan_field.py's peak2_fit summary (one panel per momentum bin,
one colour per theta bin, x-axis = the scanned value) with the BEFORE points as
solid circles and the AFTER points as dashed crosses, so the theta lines
collapsing after correction is visible at a glance. One page for peak2 mu, one
for sigma.

Usage (from the repo root):
  python plot_peak2_compare.py \
      output/python/electron_018354/torus-scale_scan/torus-scale_peak2_fit.csv \
      output/python/electron_018354/torus-scale_scan_corr/torus-scale_peak2_fit.csv \
      -o output/python/electron_018354/torus-scale_peak2_compare.pdf

  # labels default to "before" / "after"; override with --before-label/--after-label.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")  # headless: write a PDF only
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


def load_peak2_csv(path: Path):
    """Parse a scan_field.py `<param>_peak2_fit.csv`.

    Returns (param_name, data) where data[(p_lo, p_hi)][(t_lo, t_hi)] is a dict
    of parallel arrays {val, mean, mean_err, sigma, sigma_err} sorted by the
    scanned value. The first CSV column header is the scanned-parameter name."""
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        param = r.fieldnames[0]
        for row in r:
            rows.append(row)
    # data[(p_lo,p_hi)][(t_lo,t_hi)] -> list of (val, mean, mean_err, sigma, sigma_err)
    data: dict = defaultdict(lambda: defaultdict(list))
    for row in rows:
        p_key = (float(row["p_lo"]), float(row["p_hi"]))
        t_key = (float(row["theta_lo"]), float(row["theta_hi"]))
        data[p_key][t_key].append((
            float(row[param]), float(row["peak2_mean_cm"]),
            float(row["peak2_mean_err_cm"]), float(row["peak2_sigma_cm"]),
            float(row["peak2_sigma_err_cm"]),
        ))
    # Sort each series by the scanned value and repack into arrays.
    packed: dict = defaultdict(dict)
    for p_key, tmap in data.items():
        for t_key, recs in tmap.items():
            recs.sort(key=lambda x: x[0])
            arr = np.array(recs, dtype=float)
            packed[p_key][t_key] = dict(
                val=arr[:, 0], mean=arr[:, 1], mean_err=arr[:, 2],
                sigma=arr[:, 3], sigma_err=arr[:, 4])
    return param, packed


def sorted_bins(*datasets):
    """Union of momentum bins and theta bins across the datasets, each sorted."""
    p_bins, t_bins = set(), set()
    for d in datasets:
        for p_key, tmap in d.items():
            p_bins.add(p_key)
            t_bins.update(tmap.keys())
    return sorted(p_bins), sorted(t_bins)


def draw_page(pdf, before, after, p_bins, t_bins, param, *,
              value_key, err_key, ylabel, title, before_label, after_label):
    """One summary page: per-p panels, one colour per theta, before solid/o and
    after dashed/x."""
    th_colors = [plt.cm.tab10(i % 10) for i in range(len(t_bins))]
    th_labels = [rf"${int(t_lo)}^\circ \leq \theta < {int(t_hi)}^\circ$" for t_lo, t_hi in t_bins]
    n_p = len(p_bins)
    pncol = 4
    pnrow = (n_p + pncol - 1) // pncol
    fig, axs = plt.subplots(pnrow, pncol, figsize=(3.3 * pncol, 2.9 * pnrow),
                            squeeze=False, sharex=True)
    for ip, p_key in enumerate(p_bins):
        ax = axs[ip // pncol][ip % pncol]
        for it, t_key in enumerate(t_bins):
            for src, style in ((before, dict(ls="-", marker="o")),
                               (after, dict(ls="--", marker="x"))):
                series = src.get(p_key, {}).get(t_key)
                if series is None:
                    continue
                ax.errorbar(series["val"], series[value_key], yerr=series[err_key],
                            ms=3, lw=0.8, capsize=1.5, elinewidth=0.5,
                            color=th_colors[it], **style,
                            label=th_labels[it] if (ip == 0 and style["ls"] == "-") else None)
        p_lo, p_hi = p_key
        ax.set_title(rf"${p_lo:g} < p < {p_hi:g}$ GeV/$c$", fontsize=9)
        ax.tick_params(labelsize=6)
        if ip // pncol == pnrow - 1:
            ax.set_xlabel(param, fontsize=8)
        if ip % pncol == 0:
            ax.set_ylabel(ylabel, fontsize=8)
    for j in range(n_p, pnrow * pncol):  # hide unused panels
        axs[j // pncol][j % pncol].set_visible(False)

    # Two legends: theta colours (top row handles) + before/after line style.
    h, l = axs[0][0].get_legend_handles_labels()
    style_handles = [
        Line2D([], [], color="0.3", ls="-", marker="o", ms=4, label=before_label),
        Line2D([], [], color="0.3", ls="--", marker="x", ms=4, label=after_label),
    ]
    leg1 = fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, frameon=False,
                      title=r"$\theta$ bin", bbox_to_anchor=(0.5, -0.02))
    fig.add_artist(leg1)
    fig.legend(handles=style_handles, loc="lower center", ncol=2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before_csv", type=Path,
                    help="<param>_peak2_fit.csv from the uncorrected scan")
    ap.add_argument("after_csv", type=Path,
                    help="<param>_peak2_fit.csv from the corrected scan")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output PDF (default: <before_csv dir>/<param>_peak2_compare.pdf)")
    ap.add_argument("--before-label", default="before correction")
    ap.add_argument("--after-label", default="after correction")
    ap.add_argument("--species", default=None,
                    help="title prefix (default: inferred from the before-CSV path)")
    args = ap.parse_args()

    param_b, before = load_peak2_csv(args.before_csv)
    param_a, after = load_peak2_csv(args.after_csv)
    if param_a != param_b:
        raise SystemExit(f"error: scanned parameter differs: '{param_b}' vs '{param_a}'")
    param = param_b

    p_bins, t_bins = sorted_bins(before, after)
    # Infer a species label from .../output/python/<species>/<param>_scan/...
    species = args.species
    if species is None:
        parts = args.before_csv.resolve().parts
        species = parts[parts.index("python") + 1] if "python" in parts else ""

    out = args.output or (args.before_csv.parent / f"{param}_peak2_compare.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        draw_page(pdf, before, after, p_bins, t_bins, param,
                  value_key="mean", err_key="mean_err",
                  ylabel=r"peak2 $\mu$ [cm]",
                  title=rf"{species} — 2nd-peak mean vs {param}, before/after vz correction",
                  before_label=args.before_label, after_label=args.after_label)
        draw_page(pdf, before, after, p_bins, t_bins, param,
                  value_key="sigma", err_key="sigma_err",
                  ylabel=r"peak2 $\sigma$ [cm]",
                  title=rf"{species} — 2nd-peak width vs {param}, before/after vz correction",
                  before_label=args.before_label, after_label=args.after_label)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
