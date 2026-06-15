"""Per-sector electron-vz diagnostic from vz-swim-hist's (sector x p x theta) grid.

Tests whether the per-sector (azimuthal) vz modulation is:
  - momentum-dependent (amplitude ~ 1/p)  -> a torus-coil field effect (in the map), or
  - momentum-independent (amplitude flat)  -> DC sector alignment (geometry).
Also shows the p-integrated per-sector pattern. Downstream target peak only.

Figures:
  vz_sectors{suffix}.pdf
    page 1: swum-vz peak vs sector (1-6), one curve per theta bin (p-integrated)
    page 2: swum-vz peak vs theta, one curve per sector (p-integrated)
    page 3: azimuthal sinusoid amplitude vs p (and amplitude*p) -> the 1/p test

Usage:
  python plot_vz_sectors.py [vz_bins.root] [--suffix _tag]
"""
import uproot
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os as _os
try:
    import mplhep as hep
    import scienceplots  # noqa: F401
    if _os.environ.get("MPL_USETEX") == "1":
        hep.style.use("CMSTex"); plt.style.use(["science"]); mpl.rc("text", usetex=True)
    else:
        hep.style.use("CMS"); plt.style.use(["science", "no-latex"]); mpl.rc("text", usetex=False)
    _HEP = True
except Exception as exc:  # pragma: no cover
    print(f"[warn] fancy style unavailable ({exc}); using matplotlib defaults")
    mpl.rc("text", usetex=False); _HEP = False

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 12,
    "xtick.direction": "in", "ytick.direction": "in",
})

# Layout (keep in sync with src/vz_swim_hist.cpp).
N_SEC = 6
P_EDGES = np.arange(2, 10 + 1, 1)
P_BINS = list(zip(P_EDGES[:-1], P_EDGES[1:]))
TH_EDGES = np.arange(6, 26 + 1, 2)
TH_BINS = list(zip(TH_EDGES[:-1], TH_EDGES[1:]))
SEC_PHI = np.deg2rad([60 * s for s in range(N_SEC)])  # sector 1..6 centres

# Only the downstream target peak (~ -3 cm).
PK_LO, PK_HI = -5.0, -2.0
MIN_ENTRIES = 500
# theta bins used for the momentum test (low theta = strongest azimuthal signal).
TEST_TH_MAX = 14
TH_COLORS = [plt.cm.tab10(i % 10) for i in range(len(TH_BINS))]
SEC_COLORS = [plt.cm.viridis(i / (N_SEC - 1)) for i in range(N_SEC)]


def _key(sec: int, p_lo: int, p_hi: int, t_lo: int, t_hi: int) -> str:
    return f"vzsec_swum_s{sec}_p{p_lo:02d}_{p_hi:02d}_th{t_lo:02d}_{t_hi:02d}"


def _find_peak(vals, centers, lo, hi):
    if vals.sum() < MIN_ENTRIES:
        return None
    sel = (centers >= lo) & (centers < hi)
    if not np.any(sel) or vals[sel].max() <= 0:
        return None
    idx = np.flatnonzero(sel)
    k = idx[np.argmax(vals[idx])]
    if k == 0 or k == len(vals) - 1:
        return float(centers[k])
    y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
    denom = y0 - 2 * y1 + y2
    if denom >= 0:
        return float(centers[k])
    bw = centers[1] - centers[0]
    return float(centers[k] + 0.5 * bw * (y0 - y2) / denom)


def _hist(f, sec, p_lo, p_hi, t_lo, t_hi):
    key = _key(sec, p_lo, p_hi, t_lo, t_hi)
    if key not in f:
        return None, None
    return f[key].to_numpy()


def peak_pint(f, sec, t_lo, t_hi):
    """Downstream peak of the p-summed swum-vz for (sector, theta)."""
    tot, edges = None, None
    for p_lo, p_hi in P_BINS:
        v, e = _hist(f, sec, p_lo, p_hi, t_lo, t_hi)
        if v is None:
            continue
        edges = e
        tot = v if tot is None else tot + v
    if tot is None:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    return _find_peak(tot, centers, PK_LO, PK_HI)


def peak_pbin(f, sec, p_lo, p_hi, t_lo, t_hi):
    v, e = _hist(f, sec, p_lo, p_hi, t_lo, t_hi)
    if v is None:
        return None
    return _find_peak(v, 0.5 * (e[:-1] + e[1:]), PK_LO, PK_HI)


def sinusoid_amp(peaks_by_sector: dict) -> float:
    """Fit {sector(1-based): peak} to A cosφ + B sinφ + c; return amplitude."""
    if len(peaks_by_sector) < 4:
        return float("nan")
    ph = np.array([SEC_PHI[s - 1] for s in peaks_by_sector])
    ys = np.array(list(peaks_by_sector.values()))
    M = np.column_stack([np.cos(ph), np.sin(ph), np.ones_like(ph)])
    a, b, _ = np.linalg.lstsq(M, ys, rcond=None)[0]
    return float(np.hypot(a, b))


def plot_sectors(file_path="vz_bins.root", out_path="vz_sectors.pdf"):
    with uproot.open(file_path) as f, PdfPages(out_path) as pdf:
        # ---- page 1: p-integrated peak vs sector, one curve per theta ----
        pint = {(s, t_lo): peak_pint(f, s, t_lo, t_hi)
                for s in range(1, N_SEC + 1) for t_lo, t_hi in TH_BINS}
        secs = list(range(1, N_SEC + 1))
        fig, ax = plt.subplots(figsize=(9, 6))
        for (t_lo, t_hi), color in zip(TH_BINS, TH_COLORS):
            ys = [pint.get((s, t_lo), np.nan) for s in secs]
            if np.sum(np.isfinite(ys)) >= 2:
                ax.plot(secs, ys, marker="o", ms=4, lw=1.2, color=color,
                        label=rf"${t_lo}$-${t_hi}^\circ$")
        ax.set_xlabel("sector"); ax.set_ylabel(r"swum $v_z$ peak [cm]"); ax.set_xticks(secs)
        ax.set_title(r"$e^-$ swum-$v_z$ peak vs sector (p-integrated)")
        ax.legend(fontsize=8, ncol=2, title=r"$\theta$")
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ---- page 2: p-integrated peak vs theta, one curve per sector ----
        fig, ax = plt.subplots(figsize=(9, 6))
        for s, color in zip(range(1, N_SEC + 1), SEC_COLORS):
            tm = [0.5 * (a + b) for a, b in TH_BINS if pint.get((s, a)) is not None]
            ys = [pint[(s, a)] for a, b in TH_BINS if pint.get((s, a)) is not None]
            if len(tm) >= 2:
                ax.plot(tm, ys, marker="o", ms=4, lw=1.2, color=color, label=f"sector {s}")
        ax.set_xlabel(r"$\theta$ [deg]"); ax.set_ylabel(r"swum $v_z$ peak [cm]")
        ax.set_title(r"$e^-$ swum-$v_z$ peak vs $\theta$, per sector (p-integrated)")
        ax.legend(fontsize=9, ncol=2)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ---- page 3: azimuthal amplitude vs p (the 1/p test) ----
        pmid, amp_p, amp_x_p = [], [], []
        for p_lo, p_hi in P_BINS:
            amps = []
            for t_lo, t_hi in TH_BINS:
                if t_lo >= TEST_TH_MAX:
                    continue
                by_sec = {s: peak_pbin(f, s, p_lo, p_hi, t_lo, t_hi)
                          for s in range(1, N_SEC + 1)}
                by_sec = {s: v for s, v in by_sec.items() if v is not None}
                a = sinusoid_amp(by_sec)
                if np.isfinite(a):
                    amps.append(a)
            if amps:
                pm = 0.5 * (p_lo + p_hi)
                pmid.append(pm); amp_p.append(np.mean(amps)); amp_x_p.append(np.mean(amps) * pm)

        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        axs[0].plot(pmid, amp_p, marker="o", color="tab:blue")
        axs[0].set_xlabel(r"$p$ [GeV]"); axs[0].set_ylabel("azimuthal amplitude [cm]")
        axs[0].set_title(rf"sector modulation vs $p$ ($\theta<{TEST_TH_MAX}^\circ$)")
        axs[0].set_ylim(bottom=0)
        axs[1].plot(pmid, amp_x_p, marker="s", color="tab:red")
        axs[1].set_xlabel(r"$p$ [GeV]"); axs[1].set_ylabel(r"amplitude $\times\,p$ [cm$\cdot$GeV]")
        axs[1].set_title(r"flat $\Rightarrow$ field ($\propto 1/p$); rising $\Rightarrow$ alignment")
        axs[1].set_ylim(bottom=0)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"Saved {out_path}")
    print("\nmomentum test (downstream peak, theta<%d):" % TEST_TH_MAX)
    print(f"{'p':>5} {'amp [cm]':>9} {'amp*p':>8}")
    for pm, a, ap in zip(pmid, amp_p, amp_x_p):
        print(f"{pm:5.1f} {a:9.3f} {ap:8.3f}")
    if len(amp_p) >= 3:
        # crude verdict: does amp*p rise (alignment) or stay flat (field ~1/p)?
        rng_axp = max(amp_x_p) - min(amp_x_p)
        rng_a = max(amp_p) - min(amp_p)
        print(f"\nspan(amp)= {rng_a:.3f},  span(amp*p)= {rng_axp:.3f}")
        print("amp roughly flat & amp*p rising  -> DC sector ALIGNMENT (p-independent)")
        print("amp*p roughly flat & amp falling -> torus-coil FIELD (~1/p)")


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(
        description="Per-sector swum-vz diagnostic from a vz-swim-hist file.")
    parser.add_argument("file", nargs="?", default="output/cpp/vz_bins.root")
    parser.add_argument("-s", "--suffix", default="")
    cli = parser.parse_args()

    out_dir = os.path.join("output", "python", "vz_sectors")
    os.makedirs(out_dir, exist_ok=True)

    def op(name):
        stem, dot, ext = name.rpartition(".")
        fname = f"{stem}{cli.suffix}.{ext}" if dot else f"{name}{cli.suffix}"
        return os.path.join(out_dir, fname)

    plot_sectors(cli.file, out_path=op("vz_sectors.pdf"))
