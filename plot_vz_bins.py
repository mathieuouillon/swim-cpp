"""Plot the electron-vz (p, theta) grid produced by vz-swim-hist (vz_bins.root).

Figures:
  vz_theta_overlay{suffix}.pdf  reference-style summary: one panel per p-bin
                                (2x4), all theta bins overlaid (normalized);
                                page 1 = reconstructed vz, page 2 = swum vz
  vz_rec_swum{suffix}.pdf       one page per p-bin: 2x5 theta panels, EB-
                                reconstructed vz overlaid with the swum vz
  vz_dvz{suffix}.pdf            same layout for dvz = vz(swum) - vz(rec)
  vz_summary{suffix}.pdf        <dvz> vs theta, one curve per p-bin (bars = RMS)

Usage:
  python plot_vz_bins.py [vz_bins.root] [--suffix _tag]
"""
import re

import uproot
import numpy as np
import matplotlib as mpl
mpl.use("Agg")  # non-interactive backend: this script only writes PDFs. Forcing Agg
                # avoids hanging on a dead X11/$DISPLAY forward on a headless farm node.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

# ── Style ──────────────────────────────────────────────────────────────────
# Same fast non-TeX style block as plot_vz.py: mplhep + scienceplots "no-latex"
# render `$...$` via matplotlib's built-in mathtext. Set MPL_USETEX=1 to opt in.
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


def histstep(values, edges, *, ax=None, color=None, label=None, linestyle="-",
             linewidth=1.0, density=False, **_):
    """Step histogram drawn from (values, edges) — a mplhep-free stand-in for
    hep.histplot(histtype='step'), so the plotter has no hard mplhep dependency."""
    ax = ax or plt.gca()
    v = np.asarray(values, dtype=float)
    if density:
        area = v.sum() * (edges[1] - edges[0])
        if area > 0:
            v = v / area
    ax.stairs(v, edges, color=color, label=label, linestyle=linestyle, linewidth=linewidth)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": False, "ytick.right": False,
})

# ── Layout of the vz-swim-hist output ───────────────────────────────────────
# Histograms are named {vz_rec,vz_swum,dvz}_p{AA}_{BB}_th{CC}_{DD}, with the
# momentum-bin edges in GeV and the theta-bin edges in degrees, zero-padded to
# two digits (the cell_tag() encoding in src/vz_swim_hist.cpp).
P_EDGES = np.arange(2, 10 + 1, 1)        # 2,3,...,10 GeV -> 8 bins
TH_EDGES = np.arange(6, 26 + 1, 2)       # 6..26 deg      -> 10 bins
P_BINS = list(zip(P_EDGES[:-1], P_EDGES[1:]))
TH_BINS = list(zip(TH_EDGES[:-1], TH_EDGES[1:]))
NCOL = 5                                 # theta panels per row; rows follow the bin count


def _theta_rows() -> int:
    return (len(TH_BINS) + NCOL - 1) // NCOL

# One color per theta bin. tab10 for <=10 bins (the electron grid, as in the
# reference figure); a theta-ordered turbo ramp when there are more (the 12-bin
# pion grid) so every bin stays distinct.
def _theta_colors(n: int):
    if n <= 10:
        return [plt.cm.tab10(i) for i in range(n)]
    return [plt.cm.turbo(x) for x in np.linspace(0.05, 0.95, n)]


TH_COLORS = _theta_colors(len(TH_BINS))


def _ptag(e: float) -> str:
    """Momentum-edge tag matching vz-swim-hist's p_tag: integer GeV -> 2-digit
    ('02'); fractional -> one decimal ('0.3')."""
    r = round(e)
    return f"{int(r):02d}" if abs(e - r) < 1e-6 else f"{e:.1f}"


def _key(prefix: str, p_lo: float, p_hi: float, t_lo: int, t_hi: int) -> str:
    return f"{prefix}_p{_ptag(p_lo)}_{_ptag(p_hi)}_th{int(t_lo):02d}_{int(t_hi):02d}"


def _set_grid_from(f) -> None:
    """Adopt the (p, theta) grid present in an open vz_bins.root so the plotter
    follows whatever momentum binning vz-swim-hist used (pion 0.3-6 GeV vs the
    electron 2-10). Updates the module P_BINS/TH_BINS/TH_COLORS in place."""
    global P_BINS, TH_BINS, TH_COLORS
    pat = re.compile(r"vz_swum_p([0-9.]+)_([0-9.]+)_th(\d+)_(\d+)")
    ps, ths = set(), set()
    for k in f.keys():
        m = pat.match(k.split(";")[0])
        if m:
            ps.add((float(m.group(1)), float(m.group(2))))
            ths.add((int(m.group(3)), int(m.group(4))))
    if ps:
        P_BINS = sorted(ps)
        TH_BINS = sorted(ths)
        TH_COLORS = _theta_colors(len(TH_BINS))


def _mean_rms(vals: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    """Weighted mean and RMS of a binned histogram (NaN if empty)."""
    tot = vals.sum()
    if tot <= 0:
        return float("nan"), float("nan")
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = float(np.sum(centers * vals) / tot)
    rms = float(np.sqrt(max(np.sum((centers - mean) ** 2 * vals) / tot, 0.0)))
    return mean, rms


MIN_ENTRIES = 500  # skip curves with fewer entries (spiky normalized shapes)


def _theta_overlay_page(f, prefix: str, label: str, tag: str = ""):
    """One reference-style page: one panel per p-bin (2x4 grid), all theta bins
    overlaid as normalized curves of histogram `prefix` (vz_rec or vz_swum).
    Each panel autoscales its own y axis; curves below MIN_ENTRIES are skipped."""
    ncol = 4
    nrow = (len(P_BINS) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.0 * nrow),
                            squeeze=False, sharex=True)
    for j in range(len(P_BINS), nrow * ncol):  # hide unused panels (e.g. the pion 6-bin grid)
        axs[j // ncol][j % ncol].set_visible(False)
    for ip, (p_lo, p_hi) in enumerate(P_BINS):
        ax = axs[ip // ncol][ip % ncol]
        for (t_lo, t_hi), color in zip(TH_BINS, TH_COLORS):
            vals, edges = f[_key(prefix, p_lo, p_hi, t_lo, t_hi)].to_numpy()
            if vals.sum() >= MIN_ENTRIES:
                histstep(vals, edges, ax=ax, color=color, density=True, linewidth=1.0)
        ax.set_title(rf"${p_lo:.1f} < p < {p_hi:.1f}$ GeV/$c$", fontsize=12)
        if ip // ncol == nrow - 1:
            ax.set_xlabel(r"$v_z$ [cm]")
        if ip % ncol == 0:
            ax.set_ylabel("Normalized")
    # Shared legend above the panels: theta-bin colors.
    handles = [Line2D([], [], color=c, lw=1.6) for c in TH_COLORS]
    labels = [rf"${t_lo}^\circ \leq \theta < {t_hi}^\circ$" for t_lo, t_hi in TH_BINS]
    title = rf"all sectors combined — {label}"
    if tag:
        title += f" — {tag}"
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 1.0),
               title=title, title_fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    return fig


def plot_theta_overlay(file_path: str = "vz_bins.root", out_path: str = "vz_theta_overlay.pdf",
                       tag: str = ""):
    """Two reference-style pages in one PDF: page 1 reconstructed vz, page 2
    swum vz (identical layout/colors/axes for flip-comparison). `tag` is an
    extra label appended to the legend title (e.g. the solenoid z-shift)."""
    with uproot.open(file_path) as f, PdfPages(out_path) as pdf:
        _set_grid_from(f)
        for prefix, label in (("vz_rec", "reconstructed"), ("vz_swum", "swum (DC)")):
            fig = _theta_overlay_page(f, prefix, label, tag)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved {out_path}")


def plot_rec_swum(file_path: str = "vz_bins.root", out_path: str = "vz_rec_swum.pdf"):
    """One page per p-bin: 2x5 theta panels overlaying rec vz and swum vz."""
    with uproot.open(file_path) as f, PdfPages(out_path) as pdf:
        _set_grid_from(f)
        nrow = _theta_rows()
        for p_lo, p_hi in P_BINS:
            fig, axs = plt.subplots(nrow, NCOL, figsize=(4.0 * NCOL, 3.4 * nrow),
                                    squeeze=False, sharex=True)
            for j in range(len(TH_BINS), nrow * NCOL):  # hide unused panels
                axs[j // NCOL][j % NCOL].axis("off")
            for i, (t_lo, t_hi) in enumerate(TH_BINS):
                ax = axs[i // NCOL][i % NCOL]
                rv, edges = f[_key("vz_rec", p_lo, p_hi, t_lo, t_hi)].to_numpy()
                sv, _ = f[_key("vz_swum", p_lo, p_hi, t_lo, t_hi)].to_numpy()
                n = int(rv.sum())
                if n > 0:
                    rmean, _ = _mean_rms(rv, edges)
                    smean, _ = _mean_rms(sv, edges)
                    histstep(rv, edges, ax=ax, color="tab:blue",
                             label=rf"rec ($\mu$={rmean:.2f})")
                    histstep(sv, edges, ax=ax, color="tab:red",
                             label=rf"swum ($\mu$={smean:.2f})")
                    ax.legend(fontsize=7, title=f"N={n}", title_fontsize=7)
                else:
                    ax.text(0.5, 0.5, "no entries", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="gray")
                ax.set_title(rf"${t_lo} < \theta < {t_hi}^\circ$", fontsize=10)
                if i // NCOL == nrow - 1:
                    ax.set_xlabel(r"$v_z$ [cm]")
                if i % NCOL == 0:
                    ax.set_ylabel("Counts")
            fig.suptitle(rf"$e^-$ $v_z$: rec vs swum (DC R1 $\to$ beamline), "
                         rf"${p_lo} < p < {p_hi}$ GeV")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved {out_path}")


def plot_dvz(file_path: str = "vz_bins.root", out_path: str = "vz_dvz.pdf"):
    """One page per p-bin: 2x5 theta panels of dvz = vz(swum) - vz(rec)."""
    with uproot.open(file_path) as f, PdfPages(out_path) as pdf:
        _set_grid_from(f)
        nrow = _theta_rows()
        for p_lo, p_hi in P_BINS:
            fig, axs = plt.subplots(nrow, NCOL, figsize=(4.0 * NCOL, 3.4 * nrow),
                                    squeeze=False, sharex=True)
            for j in range(len(TH_BINS), nrow * NCOL):  # hide unused panels
                axs[j // NCOL][j % NCOL].axis("off")
            for i, (t_lo, t_hi) in enumerate(TH_BINS):
                ax = axs[i // NCOL][i % NCOL]
                vals, edges = f[_key("dvz", p_lo, p_hi, t_lo, t_hi)].to_numpy()
                n = int(vals.sum())
                if n > 0:
                    mean, rms = _mean_rms(vals, edges)
                    histstep(vals, edges, ax=ax, color="tab:green")
                    ax.axvline(0.0, color="gray", lw=0.8, ls=":")
                    ax.text(0.04, 0.95,
                            f"N={n}\n$\\mu$={mean:.2f} cm\n$\\sigma$={rms:.2f} cm",
                            ha="left", va="top", transform=ax.transAxes, fontsize=8)
                else:
                    ax.text(0.5, 0.5, "no entries", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="gray")
                ax.set_title(rf"${t_lo} < \theta < {t_hi}^\circ$", fontsize=10)
                if i // NCOL == nrow - 1:
                    ax.set_xlabel(r"$v_z^{\rm swum} - v_z^{\rm rec}$ [cm]")
                if i % NCOL == 0:
                    ax.set_ylabel("Counts")
            fig.suptitle(rf"$e^-$ $\Delta v_z$ (swum $-$ rec), ${p_lo} < p < {p_hi}$ GeV")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved {out_path}")


def plot_summary(file_path: str = "vz_bins.root", out_path: str = "vz_summary.pdf"):
    """<dvz> vs theta-bin center, one curve per p-bin (bars = RMS)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    with uproot.open(file_path) as f:
        _set_grid_from(f)  # discover the (p, theta) grid before sizing arrays
        tmid = np.array([0.5 * (t_lo + t_hi) for t_lo, t_hi in TH_BINS])
        colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(P_BINS)))
        for (p_lo, p_hi), color in zip(P_BINS, colors):
            means = np.full(len(TH_BINS), np.nan)
            rms = np.full(len(TH_BINS), np.nan)
            for i, (t_lo, t_hi) in enumerate(TH_BINS):
                vals, edges = f[_key("dvz", p_lo, p_hi, t_lo, t_hi)].to_numpy()
                if vals.sum() > 0:
                    means[i], rms[i] = _mean_rms(vals, edges)
            ax.errorbar(tmid, means, yerr=rms, marker="o", ms=4, lw=1.2, capsize=2,
                        color=color, label=rf"${p_lo} < p < {p_hi}$ GeV")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$\langle v_z^{\rm swum} - v_z^{\rm rec} \rangle$ [cm]  (bars = RMS)")
    ax.set_title(r"$e^-$ swum$-$rec vertex shift vs $\theta$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Render the electron-vz (p, theta) grid figures from a "
                    "vz-swim-hist vz_bins.root.")
    parser.add_argument("file", nargs="?", default="output/cpp/vz_bins.root",
                        help="input ROOT file (default: output/cpp/vz_bins.root)")
    parser.add_argument("-s", "--suffix", default="",
                        help="string inserted before the '.pdf' extension of every output "
                             "figure, e.g. --suffix _018614 -> vz_rec_swum_018614.pdf")
    cli = parser.parse_args()

    out_dir = os.path.join("output", "python", "vz_bins")
    os.makedirs(out_dir, exist_ok=True)

    def op(name: str) -> str:
        """Apply the --suffix and route the figure into output/python/vz_bins/."""
        stem, dot, ext = name.rpartition(".")
        fname = f"{stem}{cli.suffix}.{ext}" if dot else f"{name}{cli.suffix}"
        return os.path.join(out_dir, fname)

    plot_theta_overlay(cli.file, out_path=op("vz_theta_overlay.pdf"))
    plot_rec_swum(cli.file, out_path=op("vz_rec_swum.pdf"))
    plot_dvz(cli.file, out_path=op("vz_dvz.pdf"))
    plot_summary(cli.file, out_path=op("vz_summary.pdf"))
