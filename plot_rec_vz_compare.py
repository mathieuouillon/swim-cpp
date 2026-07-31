"""Overlay reconstructed-vz (p, theta) grids of N passes, plus an optional swim.

Reads two-or-more ROOT files written by rec-vz-hist (one per pass, e.g. torus
scale 0.998 vs 1.000, plus real data reconstructed with 0.998), each holding
`vz_rec_<sp>_pAA_BB_thCC_DD` histograms for sp in {ele, pip, pim}, and overlays
the matching cells. Passes A and B are positional; any further passes are added
with repeated --rec FILE LABEL.

Optionally overlays a third curve: the swum-back vz from vz-swim-hist (run with
--torus-scale, one file per species), i.e. taking one pass's reconstructed
tracks and re-swimming them through a different torus scale. Its histograms are
named `vz_swum_pAA_BB_thCC_DD` (no species in the name -> one file per species);
pass them with --swim-ele/--swim-pip/--swim-pim. Each curve is area-normalized
within the rec window so the swum range (-13..0) compares fairly with rec.

Figures (one multi-page PDF per species present in both files):
  rec_vz_<sp>_compare{suffix}.pdf
    page 1     : integrated vz (rec A, rec B, and summed swim), normalized
    then       : page per p-bin, theta panels overlaying each series
    last page  : <vz> vs theta per p-bin per series, plus the per-cell mean
                 shift of B (and swim) relative to A

Usage:
  python plot_rec_vz_compare.py A.root B.root [--labels "scale 0.998" "scale 1.0"]
      [--rec data.root "data 0.998"]   # extra pass(es), repeatable
      [--swim-ele swim_ele.root --swim-pip swim_pip.root --swim-pim swim_pim.root]
      [--swim-label "swim t=0.998"] [--suffix _tag] [--species ele pip pim]
"""
import argparse
import re
from collections import defaultdict

import uproot
import numpy as np
import matplotlib as mpl
mpl.use("Agg")  # non-interactive backend: this script only writes PDFs. Forcing Agg
                # avoids hanging on a dead X11/$DISPLAY forward on a headless farm node.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

# ── Style ──────────────────────────────────────────────────────────────────
# Same fast non-TeX style block as the other plotters: mplhep + scienceplots
# "no-latex" render `$...$` via matplotlib's built-in mathtext.
import os as _os

try:
    import mplhep as hep
    import scienceplots  # noqa: F401  (registers the "science" style)

    if _os.environ.get("MPL_USETEX") == "1":
        hep.style.use("CMSTex")
        plt.style.use(["science"])
        mpl.rc("text", usetex=True)
        mpl.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{lmodern}")
    else:
        hep.style.use("CMS")
        plt.style.use(["science", "no-latex"])
        mpl.rc("text", usetex=False)
except Exception as exc:  # pragma: no cover - environment dependent
    print(f"[warn] fancy style unavailable ({exc}); using matplotlib defaults")
    mpl.rc("text", usetex=False)

SPECIES_TEX = {"ele": r"$e^-$", "pip": r"$\pi^+$", "pim": r"$\pi^-$"}
# Handed out to series in draw order (A, B, any --rec extras, then the swim
# overlay): blue, red, green, purple, orange, brown.
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

# vz_rec_<sp>_p<lo>_<hi>_th<lo>_<hi>; edges are floats (one decimal for the
# fractional pion p-edges, e.g. p0.3_1), so accept an optional decimal part.
NAME_RE = re.compile(
    r"^vz_rec_(?P<sp>[a-z]+)_p(?P<plo>[0-9.]+)_(?P<phi>[0-9.]+)"
    r"_th(?P<tlo>[0-9]+)_(?P<thi>[0-9]+)$"
)
# vz_rec_<sp>_all: the integrated (whole-selection) distribution.
ALL_RE = re.compile(r"^vz_rec_(?P<sp>[a-z]+)_all$")
# vz_swum_p<lo>_<hi>_th<lo>_<hi>: vz-swim-hist output (one species per file).
SWUM_RE = re.compile(
    r"^vz_swum_p(?P<plo>[0-9.]+)_(?P<phi>[0-9.]+)_th(?P<tlo>[0-9]+)_(?P<thi>[0-9]+)$"
)


def histstep(values, edges, *, ax=None, color=None, label=None, linestyle="-",
             linewidth=1.2, density=False, norm_range=None):
    """Step histogram from (values, edges) — a mplhep-free stand-in for
    hep.histplot(histtype='step'). When density and norm_range=(lo,hi) are given
    the area is normalized over bins whose centers fall in [lo,hi], so series on
    different x-ranges (rec -5..5 vs swum -13..0) compare fairly in the window."""
    ax = ax or plt.gca()
    v = np.asarray(values, dtype=float)
    if density:
        centers = 0.5 * (edges[:-1] + edges[1:])
        mask = np.ones_like(v, dtype=bool) if norm_range is None else \
            (centers >= norm_range[0]) & (centers <= norm_range[1])
        area = v[mask].sum() * (edges[1] - edges[0])
        if area > 0:
            v = v / area
    ax.stairs(v, edges, color=color, label=label, linestyle=linestyle, linewidth=linewidth)


def load_cells(path):
    """Read one rec-vz file: returns (cells, integrated) where
    cells     = sp -> {(plo, phi, tlo, thi): (counts, edges)}  (per (p,theta) cell)
    integrated = sp -> (counts, edges)                         (vz_rec_<sp>_all)."""
    cells = defaultdict(dict)
    integrated = {}
    with uproot.open(path) as f:
        for key in f.keys(cycle=False):
            if m := NAME_RE.match(key):
                counts, edges = f[key].to_numpy()
                cell = (float(m["plo"]), float(m["phi"]), int(m["tlo"]), int(m["thi"]))
                cells[m["sp"]][cell] = (counts, edges)
            elif m := ALL_RE.match(key):
                integrated[m["sp"]] = f[key].to_numpy()
    return cells, integrated


def load_swim(path):
    """Read one vz-swim-hist file (single species): {cell: (counts, edges)} from
    its vz_swum_<cell> histograms."""
    cells = {}
    with uproot.open(path) as f:
        for key in f.keys(cycle=False):
            if m := SWUM_RE.match(key):
                cell = (float(m["plo"]), float(m["phi"]), int(m["tlo"]), int(m["thi"]))
                cells[cell] = f[key].to_numpy()
    return cells


def sum_cells(cells):
    """Sum every cell histogram of one species into one (counts, edges); the
    per-cell vz axis is identical within a species, so they add bin-by-bin.
    Returns None if empty."""
    if not cells:
        return None
    edges = next(iter(cells.values()))[1]
    total = np.zeros(len(edges) - 1, dtype=float)
    for counts, _ in cells.values():
        total += counts
    return total, edges


def mean_rms(counts, edges):
    """Histogram mean and RMS over bin centers; (nan, nan, 0) if empty."""
    n = counts.sum()
    if n <= 0:
        return float("nan"), float("nan"), 0.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.sum(counts * centers) / n
    var = np.sum(counts * (centers - mean) ** 2) / n
    return mean, np.sqrt(max(var, 0.0)), n


def grid_layout(t_bins):
    """Theta panels for one p-bin page, laid out ~square."""
    ncol = int(np.ceil(np.sqrt(len(t_bins))))
    nrow = int(np.ceil(len(t_bins) / ncol))
    return nrow, ncol


def window_of(series):
    """(lo, hi) x-window from the first series' cells (the rec range, -5..5)."""
    for s in series:
        for counts, edges in s["cells"].values():
            return float(edges[0]), float(edges[-1])
    return None


def plot_species(sp, series, pdf):
    """One page per p-bin: theta panels overlaying every series (normalized)."""
    base = set(series[0]["cells"]) & set(series[1]["cells"])
    if not base:
        print(f"[warn] no shared {sp} cells; skipping")
        return
    win = window_of(series)
    p_bins = sorted({(plo, phi) for plo, phi, _, _ in base})
    for plo, phi in p_bins:
        t_bins = sorted({(tlo, thi) for (a, b, tlo, thi) in base if (a, b) == (plo, phi)})
        nrow, ncol = grid_layout(t_bins)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for k, (tlo, thi) in enumerate(t_bins):
            ax = axes.flat[k]
            ax.set_visible(True)
            cell = (plo, phi, tlo, thi)
            for s in series:
                if cell not in s["cells"]:
                    continue
                counts, edges = s["cells"][cell]
                histstep(counts, edges, ax=ax, color=s["color"], label=s["label"],
                         density=True, norm_range=win)
                mean, _, _ = mean_rms(counts, edges)
                ax.axvline(mean, color=s["color"], ls="--", lw=0.8, alpha=0.7)
            if win:
                ax.set_xlim(*win)
            ax.set_title(rf"${tlo}<\theta<{thi}^\circ$", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlabel(r"$v_z$ [cm]", fontsize=7)
        labs = " vs ".join(s["label"] for s in series)
        fig.suptitle(rf"{SPECIES_TEX.get(sp, sp)}  ${plo}<p<{phi}$ GeV  —  $v_z$ ({labs})",
                     fontsize=11)
        handles = [Line2D([0], [0], color=s["color"], label=s["label"]) for s in series]
        fig.legend(handles=handles, loc="upper right", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig)
        plt.close(fig)


def plot_integrated(sp, integrated, pdf):
    """Overlay the integrated vz of each series (list of (counts, edges, label,
    color)), as raw counts and area-normalized within the rec window."""
    win = (float(integrated[0][1][0]), float(integrated[0][1][-1]))
    fig, (axc, axn) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, density in ((axc, False), (axn, True)):
        for counts, edges, lab, col in integrated:
            histstep(counts, edges, ax=ax, color=col, label=lab, density=density,
                     norm_range=win if density else None)
            mean, _, _ = mean_rms(counts, edges)
            ax.axvline(mean, color=col, ls="--", lw=0.8, alpha=0.7)
        ax.set_xlabel(r"$v_z$ [cm]")
        ax.set_xlim(*win)
        ax.legend(fontsize=9)
    axc.set_ylabel("counts")
    axn.set_ylabel("normalized")
    info = "   |   ".join(
        rf"{lab}: $\langle v_z\rangle$={mean_rms(c, e)[0]:.2f}, RMS={mean_rms(c, e)[1]:.2f}, "
        rf"N={int(mean_rms(c, e)[2])}" for c, e, lab, _ in integrated)
    fig.suptitle(rf"{SPECIES_TEX.get(sp, sp)}  integrated $v_z$   —   {info}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def plot_meanshift(sp, series, pdf):
    """<vz> vs theta per p-bin for every series, and the per-cell mean shift of
    each non-reference series relative to series[0] (= pass A)."""
    base = set(series[0]["cells"]) & set(series[1]["cells"])
    if not base:
        return
    p_bins = sorted({(plo, phi) for plo, phi, _, _ in base})
    cmap = plt.get_cmap("viridis")
    ref = series[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    linestyles = ["-", "--", ":"]
    for ip, (plo, phi) in enumerate(p_bins):
        col = cmap(ip / max(len(p_bins) - 1, 1))
        t_bins = sorted({(tlo, thi) for (a, b, tlo, thi) in base if (a, b) == (plo, phi)})
        tc = []
        means = {i: [] for i in range(len(series))}
        for tlo, thi in t_bins:
            cell = (plo, phi, tlo, thi)
            if cell not in ref["cells"]:
                continue
            tc.append(0.5 * (tlo + thi))
            for i, s in enumerate(series):
                means[i].append(mean_rms(*s["cells"][cell])[0] if cell in s["cells"] else np.nan)
        if not tc:
            continue
        tc = np.array(tc)
        for i, s in enumerate(series):
            ax1.plot(tc + 0.1 * i, means[i], color=col, ls=linestyles[i % 3], marker="o", ms=3,
                     label=f"{plo}-{phi} GeV" if i == 0 else None)
        m_ref = np.array(means[0])
        for i in range(1, len(series)):
            ax2.plot(tc, np.array(means[i]) - m_ref, color=col, ls=linestyles[i % 3],
                     marker="o", ms=3, label=f"{plo}-{phi} GeV" if i == 1 else None)
    style_key = ", ".join(f"{ls}={s['label']}" for ls, s in zip(linestyles, series))
    ax1.set_xlabel(r"$\theta$ [deg]"); ax1.set_ylabel(r"$\langle v_z\rangle$ [cm]")
    ax1.set_title(rf"{SPECIES_TEX.get(sp, sp)}  $\langle v_z\rangle$  ({style_key})")
    ax1.legend(fontsize=7, ncol=2)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel(r"$\theta$ [deg]")
    ax2.set_ylabel(rf"$\langle v_z\rangle - \langle v_z\rangle_{{\rm {series[0]['label']}}}$ [cm]")
    ax2.set_title(f"{SPECIES_TEX.get(sp, sp)}  mean shift vs {series[0]['label']}")
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file_a", help="rec-vz ROOT file, pass A")
    ap.add_argument("file_b", help="rec-vz ROOT file, pass B")
    ap.add_argument("--labels", nargs=2, default=["A", "B"],
                    help="legend labels for the two positional passes")
    ap.add_argument("--rec", action="append", nargs=2, default=[],
                    metavar=("FILE", "LABEL"),
                    help="additional rec-vz ROOT file + legend label "
                         "(repeatable; overlaid after passes A and B)")
    ap.add_argument("--swim-ele", help="vz-swim-hist file for e- (vz_swum_<cell>)")
    ap.add_argument("--swim-pip", help="vz-swim-hist file for pi+")
    ap.add_argument("--swim-pim", help="vz-swim-hist file for pi-")
    ap.add_argument("--swim-label", default="swim", help="legend label for the swim overlay")
    ap.add_argument("--species", nargs="+", default=["ele", "pip", "pim"],
                    choices=["ele", "pip", "pim"], help="species to plot")
    ap.add_argument("--suffix", default="", help="suffix added to output PDF names")
    ap.add_argument("--outdir", default=".", help="directory for the output PDFs")
    args = ap.parse_args()

    cells_a, intg_a = load_cells(args.file_a)
    cells_b, intg_b = load_cells(args.file_b)
    # Extra rec passes (e.g. real data reconstructed with a given torus); each is
    # a (cells, integrated) pair drawn after A and B, in the order given.
    extra = [(load_cells(path), label) for path, label in args.rec]
    swim_path = {"ele": args.swim_ele, "pip": args.swim_pip, "pim": args.swim_pim}
    swim_cells = {sp: load_swim(p) for sp, p in swim_path.items() if p}
    _os.makedirs(args.outdir, exist_ok=True)

    for sp in args.species:
        if sp not in cells_a or sp not in cells_b:
            print(f"[warn] {sp} absent in one of the rec files; skipping")
            continue
        # Rec passes in draw order: A, B, then any --rec extras. Colors are
        # handed out as series are built, so the swim (last) takes the next free.
        rec_inputs = [(cells_a, intg_a, args.labels[0]),
                      (cells_b, intg_b, args.labels[1])]
        rec_inputs += [(c, i, lab) for (c, i), lab in extra]

        series, integrated = [], []
        for cells, intg, lab in rec_inputs:
            if sp not in cells:
                print(f"[warn] {sp} absent in '{lab}'; skipping that series")
                continue
            col = COLORS[len(series) % len(COLORS)]
            series.append({"cells": cells[sp], "label": lab, "color": col})
            if sp in intg:
                integrated.append((*intg[sp], lab, col))
        if swim_cells.get(sp):
            col = COLORS[len(series) % len(COLORS)]
            series.append({"cells": swim_cells[sp], "label": args.swim_label, "color": col})
            if s := sum_cells(swim_cells[sp]):
                integrated.append((*s, args.swim_label, col))

        out = _os.path.join(args.outdir, f"rec_vz_{sp}_compare{args.suffix}.pdf")
        with PdfPages(out) as pdf:
            if len(integrated) >= 2:
                plot_integrated(sp, integrated, pdf)
            plot_species(sp, series, pdf)
            plot_meanshift(sp, series, pdf)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
