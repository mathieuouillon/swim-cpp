#!/usr/bin/env python3
"""Generate the figures for notes/field_alignment.typ from the field-scan output.

The goal of the scan is to align the MEAN of the SECOND swum-vz peak. The
downstream foil sits at a fixed z, so the peak-2 mean mu2 should be the same in
every momentum AND polar-angle bin; the scan finds the field-map modification
that makes it so. For every scanned shift (solenoid x/y/z, torus x/y/z,
torus-scale) this reads the per-value vz_bins.root histograms under
output/python/<param>_scan/ (written by scan_field.py), fits the second-peak
Gaussian in each (p, theta) bin, and writes into notes/figs/:

  swum_vz_spectrum.png   one reference (p, theta)-integrated swum-vz spectrum
                         with the two target windows and the peak-2 Gaussian fit;
  <shift>_mu_pt.png      per shift: peak-2 mean mu2 vs the shift value, ONE PANEL
                         PER MOMENTUM BIN and one curve per theta bin (error
                         bars). Where the theta curves cross, mu2 is
                         theta-independent; the panel-to-panel drift shows its p
                         dependence;
  <shift>_sigma_pt.png   same layout for the peak-2 width sigma2;
  metric_all.png         a grid, one panel per shift: the alignment metric
                         M = RMS over all (p, theta) of mu2, vs the shift value,
                         minimum (= best alignment) marked;
  best_values.csv        per shift, the value that minimises M and that minimum
                         (the Typst note renders this as a table).

The binning and the peak-2 Gaussian fit are imported from scan_field.py so the
figures match what the scan itself computed.

Usage (on the farm, from the repo root):
  python notes/make_figures.py                       # all available shifts
  python notes/make_figures.py --params torus-x torus-scale
  python notes/make_figures.py --min-entries 200 --scan-root output/python
"""
import argparse
import csv
import sys
from math import ceil
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Reuse the exact scan logic (binning, peak-2 Gaussian fit) from the driver.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scan_field as sf  # noqa: E402

_, LO2, HI2 = sf.PEAK_WINDOWS[1]  # ("peak2", -5.5, -0.5)

# Shifts to show, in note order. scale params are dimensionless; the rest cm.
ALL_PARAMS = ["solenoid-x", "solenoid-y", "solenoid-z",
              "torus-x", "torus-y", "torus-z", "torus-scale"]

# One colour per theta bin (tab10 has exactly 10 distinct colours, == n theta bins).
_CMAP = plt.get_cmap("tab10")
TH_COLOR = {t_lo: _CMAP(i) for i, (t_lo, t_hi) in enumerate(sf.TH_BINS)}
TH_LABEL = {t_lo: f"{t_lo}°≤θ<{t_hi}°" for t_lo, t_hi in sf.TH_BINS}


def is_scale(param: str) -> bool:
    return param.endswith("-scale")


def unit_of(param: str) -> str:
    return "" if is_scale(param) else " [cm]"


def scan_values(scan_dir: Path, param: str) -> list[float]:
    """Swept values that actually have a vz_bins.root on disk."""
    prefix = "zshift" if param == "solenoid-z" else param
    vals = []
    for d in scan_dir.glob(f"{prefix}_*"):
        if (d / "vz_bins.root").exists():
            try:
                vals.append(float(d.name.rsplit("_", 1)[1]))
            except ValueError:
                pass
    return sorted(vals)


def integrated_spectrum(root_file: Path):
    """(counts, centers) of the swum-vz histogram summed over the (p, theta) grid."""
    import uproot
    tot, edges = None, None
    with uproot.open(root_file) as fr:
        for p_lo, p_hi in sf.P_BINS:
            for t_lo, t_hi in sf.TH_BINS:
                key = sf._key("vz_swum", p_lo, p_hi, t_lo, t_hi)
                if key not in fr:
                    continue
                hv, edges = fr[key].to_numpy()
                tot = hv if tot is None else tot + hv
    if tot is None:
        return None, None
    return tot, 0.5 * (edges[:-1] + edges[1:])


def peak2_fits_per_p_theta(root_file: Path, min_entries: int
                           ) -> dict[tuple[int, int], sf.Peak2Fit]:
    """{(p_lo, theta_lo): Peak2Fit} for one value: fit peak2 in every (p, theta)
    bin (bins below min_entries or with a failed fit are dropped)."""
    import uproot
    out: dict[tuple[int, int], sf.Peak2Fit] = {}
    with uproot.open(root_file) as fr:
        for p_lo, p_hi in sf.P_BINS:
            for t_lo, t_hi in sf.TH_BINS:
                key = sf._key("vz_swum", p_lo, p_hi, t_lo, t_hi)
                if key not in fr:
                    continue
                vals, edges = fr[key].to_numpy()
                if vals.sum() < min_entries:
                    continue
                centers = 0.5 * (edges[:-1] + edges[1:])
                fit = sf.fit_peak2_gauss(vals, centers, LO2, HI2)
                if fit.ok:
                    out[(p_lo, t_lo)] = fit
    return out


def scan_result(scan_dir: Path, param: str, min_entries: int):
    """(per_value, metric, best) for one shift, where
    per_value = {value: {(p_lo, theta_lo): Peak2Fit}} and the metric is the RMS of
    mu2 over ALL populated (p, theta) bins (foil z is p- and theta-independent)."""
    vals = scan_values(scan_dir, param)
    per_value = {v: peak2_fits_per_p_theta(
        sf.job_dir(scan_dir, param, v) / "vz_bins.root", min_entries) for v in vals}
    metric = {v: float(np.std([f.mean for f in fits.values()]))
              for v, fits in per_value.items() if len(fits) >= 2}
    best = min(metric, key=metric.get) if metric else None
    return per_value, metric, best


# ── figures ──────────────────────────────────────────────────────────────────

def _theta_legend(fig, loc="lower center", anchor=(0.5, -0.02)):
    handles = [Line2D([0], [0], color=TH_COLOR[t], marker="o", ms=3, lw=1.2,
                      label=TH_LABEL[t]) for t, _ in sf.TH_BINS]
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, ncol=5, fontsize=8,
               title=r"$\theta$ bin", columnspacing=1.2, handletextpad=0.4)


def _grid(n: int, ncols: int = 4):
    ncols = min(ncols, n)
    nrows = ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows),
                            squeeze=False)
    flat = axs.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat


def fig_spectrum(scan_dir: Path, param: str, best: float, out: Path):
    """One reference spectrum with the peak-2 Gaussian fit, to illustrate the fit."""
    label = sf.PARAMS[param][1]
    rf = sf.job_dir(scan_dir, param, best) / "vz_bins.root"
    tot, centers = integrated_spectrum(rf)
    if tot is None:
        print(f"[warn] no histograms in {rf}; skipping spectrum")
        return
    fit = sf.fit_peak2_gauss(tot, centers, LO2, HI2)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    bw = centers[1] - centers[0]
    ax.bar(centers, tot, width=bw, color="0.85", edgecolor="0.6", lw=0.3,
           label=r"swum $v_z$ ($p$, $\theta$ integrated)")
    for name, lo, hi in sf.PEAK_WINDOWS:
        col = "tab:green" if name == "peak1" else "tab:blue"
        ax.axvspan(lo, hi, color=col, alpha=0.07)
        ax.text(0.5 * (lo + hi), ax.get_ylim()[1], name, ha="center", va="bottom",
                fontsize=8, color=col)
    if fit.ok:
        ax.plot(fit.xfit, fit.yfit, color="tab:red", lw=2.0, label="peak-2 Gaussian fit")
        ax.scatter(centers[fit.lo:fit.hi + 1], tot[fit.lo:fit.hi + 1], s=12,
                   color="tab:blue", zorder=3, label="fitted bins")
        ax.axvline(fit.mean, color="tab:red", ls=":", lw=1.0,
                   label=rf"$\mu_2$ = {fit.mean:.3f} cm")
        ax.text(0.025, 0.97,
                f"peak-2 fit\n$\\mu$ = {fit.mean:.3f} $\\pm$ {fit.mean_err:.3f} cm\n"
                f"$\\sigma$ = {fit.sigma:.3f} $\\pm$ {fit.sigma_err:.3f} cm",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_xlabel(r"swum $v_z$ [cm]")
    ax.set_ylabel("counts")
    ax.set_title(f"swum-$v_z$ spectrum — {label} {sf.vfmt(best)}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_pt(per_value, best, param, attr, err_attr, ylabel, suptitle, out: Path):
    """Per shift: a peak-2 fit attribute (mean/sigma) vs the shift value, one panel
    per momentum bin and one error-bar curve per theta bin (cf. the reference
    <param>_peak2_fit per-(p, theta) figure)."""
    vals = sorted(per_value)
    fig, axs = _grid(len(sf.P_BINS), ncols=4)
    for ax, (p_lo, p_hi) in zip(axs, sf.P_BINS):
        drew = False
        for t_lo, _ in sf.TH_BINS:
            xs = [v for v in vals if (p_lo, t_lo) in per_value[v]]
            if len(xs) < 2:
                continue
            ys = [getattr(per_value[v][(p_lo, t_lo)], attr) for v in xs]
            es = [getattr(per_value[v][(p_lo, t_lo)], err_attr) for v in xs]
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=2.5, lw=0.9, capsize=1.5,
                        elinewidth=0.6, color=TH_COLOR[t_lo])
            drew = True
        if best is not None and drew:
            ax.axvline(best, color="gray", ls=":", lw=1.0)
        ax.set_title(rf"${p_lo} < p < {p_hi}$ GeV/$c$", fontsize=9)
        ax.set_xlabel(sf.PARAMS[param][1] + unit_of(param), fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
    _theta_legend(fig)
    fig.suptitle(suptitle, y=1.0, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_metric_grid(results, out: Path):
    """One panel per shift: M = RMS over all (p, theta) of mu2 vs value, min marked."""
    params = list(results)
    fig, axs = _grid(len(params))
    for ax, param in zip(axs, params):
        _, metric, best = results[param]
        vs = sorted(metric)
        ax.plot(vs, [metric[v] for v in vs], marker="o", ms=3, color="tab:red")
        if best is not None:
            ax.axvline(best, color="gray", ls=":")
            ax.text(0.5, 0.92, f"best {sf.vfmt(best)}\nRMS {metric[best]:.3f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.set_title(sf.PARAMS[param][1], fontsize=10)
        ax.set_xlabel(sf.PARAMS[param][1] + unit_of(param), fontsize=8)
        ax.set_ylabel(r"RMS of $\mu_2$ over $(p,\theta)$ [cm]", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(r"Alignment metric $\mathcal{M}=\mathrm{RMS}_{(p,\theta)}[\mu_2]$ "
                 r"for every tested shift (minimum = best alignment)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def write_best_csv(results, out: Path):
    """Per-shift best value and metric, for the Typst note's results table."""
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shift", "best value", "RMS of peak-2 mean [cm]"])
        for param, (_, metric, best) in results.items():
            if best is None:
                w.writerow([sf.PARAMS[param][1], "—", "—"])
            else:
                w.writerow([sf.PARAMS[param][1], sf.vfmt(best), f"{metric[best]:.3f}"])
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", nargs="+", default=ALL_PARAMS,
                    choices=sorted(sf.PARAMS),
                    help="shifts to plot (default: all solenoid/torus shifts + scale)")
    ap.add_argument("--spectrum-param", default="torus-x",
                    help="shift whose best value is shown in the reference spectrum")
    ap.add_argument("--min-entries", type=int, default=200,
                    help="min counts in a (p, theta) bin to fit peak2 (default: 200)")
    ap.add_argument("--scan-root", default="output/python", type=Path,
                    help="directory holding <param>_scan/ (default: output/python)")
    ap.add_argument("--out", default=Path(__file__).resolve().parent / "figs",
                    type=Path, help="output directory (default: notes/figs)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results: dict[str, tuple] = {}
    for param in args.params:
        scan_dir = args.scan_root / f"{param}_scan"
        if not scan_dir.is_dir():
            print(f"[warn] {scan_dir} not found; skipping {param}")
            continue
        per_value, metric, best = scan_result(scan_dir, param, args.min_entries)
        if not metric:
            print(f"[warn] no usable peak-2 fits for {param}; skipping")
            continue
        results[param] = (per_value, metric, best)
        bv = "—" if best is None else f"{sf.vfmt(best)} (RMS {metric[best]:.3f} cm)"
        print(f"{sf.PARAMS[param][1]:>16s}: best {bv}")
    if not results:
        sys.exit("error: no scans found under " + str(args.scan_root))

    sp = args.spectrum_param if args.spectrum_param in results else next(iter(results))
    fig_spectrum(args.scan_root / f"{sp}_scan", sp, results[sp][2],
                 args.out / "swum_vz_spectrum.png")
    fig_metric_grid(results, args.out / "metric_all.png")
    write_best_csv(results, args.out / "best_values.csv")
    for param, (per_value, _, best) in results.items():
        label = sf.PARAMS[param][1]
        fig_pt(per_value, best, param, "mean", "mean_err", r"peak2 $\mu$ [cm]",
               rf"2nd-peak mean $\mu_2$ vs {label}, per $(p, \theta)$",
               args.out / f"{param}_mu_pt.png")
        fig_pt(per_value, best, param, "sigma", "sigma_err", r"peak2 $\sigma$ [cm]",
               rf"2nd-peak width $\sigma_2$ vs {label}, per $(p, \theta)$",
               args.out / f"{param}_sigma_pt.png")


if __name__ == "__main__":
    main()
