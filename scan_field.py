#!/usr/bin/env python3
"""Field-alignment scan for the vz swim.

Runs vz-swim-hist once per swept value of a chosen field parameter
(--scan-param: solenoid-x/y/z, torus-x/y/z map shifts [cm], torus-scale /
solenoid-scale field scales, or dc-x/y/z DC start-state shifts) at most
--max-parallel instances at a time, --threads-per-job threads each, with one
live progress bar per running job (tqdm; plain prints without it), one
subfolder per value:

  <scan-dir>/
    <param>_-3.00/vz_bins.root          histograms for that value
    <param>_-3.00/vz_theta_overlay.pdf  theta-overlay figure (rec + swum pages)
    <param>_-3.00/log.txt               full vz-swim-hist output
    ...
    <param>_peaks.csv            swum-vz peak position per (value, theta, peak)
    <param>_summary.pdf          peak position vs theta per value + alignment
                                 metric vs value (minimum = best-aligned value)
  (the solenoid-z subfolders keep the historical "zshift_" prefix.)

Goal: find the field displacement for which the swum-vz target peaks line up
across theta bins (the physical foil positions are theta-independent, so the
RMS of the peak position across theta measures field-map misalignment). The
solenoid z-shift has essentially no lever arm on this; a torus transverse
(x/y) shift is the diagnostic for the 1/tan(theta) residual trend.

NOTE: each instance loads its own field maps (~1 GB torus), so 10 parallel
jobs need ~10+ GB of RAM.

Usage (on the farm, from the repo root, after hipo2root --ccdb ...):
  # Beam offset + field scales come from the tree's CCDB metadata; pass a flag
  # only to override one. geometric (p-flat) part of the vz theta-walk -> DC:
  python scan_field.py particles.root \
      --scan-param dc-z --z-min -2 --z-max 2 --z-step 0.2
  # 1/p (magnetic) part -> field SCALE (sweep absolute value near nominal):
  python scan_field.py particles.root \
      --scan-param torus-scale --shifts -1.06 -1.04 -1.02 -1.00 -0.98 -0.96
  python scan_field.py --scan-param dc-z --plot-only          # redo summary only
"""
import argparse
import csv
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np
import uproot
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.use("Agg")  # headless farm node: only writes PDFs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── Per-job progress bars ────────────────────────────────────────────────────
# Each job's vz-swim-hist runs with its progress bar enabled; we parse the
# percentage / rate / ETA out of its stdout stream and re-render it as one
# tqdm bar per running job (plain decile prints without tqdm). Non-progress
# output goes to the job's log.txt.
try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_TQDM = False

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PCT_RE = re.compile(r"(\d{1,3})%")
RATE_RE = re.compile(r"⚡\s*([^\s│]+)")
ETA_RE = re.compile(r"⏳\s*([^\s│]+)")


def say(msg: str):
    """Print without clobbering active tqdm bars."""
    if _HAVE_TQDM:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


class SlotPool:
    """Reusable terminal-line slots so at most max_parallel bars are on screen."""

    def __init__(self, n: int):
        self._free = list(range(n))
        self._lock = threading.Lock()

    def acquire(self) -> int:
        with self._lock:
            return self._free.pop(0) if self._free else 0

    def release(self, slot: int):
        with self._lock:
            self._free.append(slot)


# Live child processes, so Ctrl-C can terminate them instead of orphaning them.
_PROCS: set = set()
_PROCS_LOCK = threading.Lock()


def _kill_children():
    with _PROCS_LOCK:
        procs = list(_PROCS)
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass

# ── Layout of the vz-swim-hist output (keep in sync with src/vz_swim_hist.cpp
# and plot_vz_bins.py) ───────────────────────────────────────────────────────
P_EDGES = np.arange(2, 10 + 1, 1)        # electron default: 2,3,...,10 GeV -> 8 bins
TH_EDGES = np.arange(6, 26 + 1, 2)       # 6..26 deg (all species)          -> 10 bins
P_BINS = list(zip(P_EDGES[:-1], P_EDGES[1:]))
TH_BINS = list(zip(TH_EDGES[:-1], TH_EDGES[1:]))


def momentum_edges(pid: int):
    """Momentum bin edges for the species (MUST match vz-swim-hist make_grid):
    pions (|pid| == 211) get the soft [0.3, 1, 2, 3, 4, 5, 6] grid; everything
    else keeps the electron [2 .. 10] grid. main() applies it to P_EDGES/P_BINS."""
    if abs(pid) == 211:
        return np.array([0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return np.arange(2, 10 + 1, 1).astype(float)

# The two RG-D target peaks in the swum-vz distribution: search window per peak
# (cm). The histogram axis is [-13, 0].
PEAK_WINDOWS = (("peak1", -11.0, -5.5), ("peak2", -5.5, -0.5))
MIN_ENTRIES = 500  # per-theta statistics threshold (p-integrated)

# Scannable field parameters -> (vz-swim-hist flag, human label). All are rigid
# map displacements in cm; a torus transverse (x/y) shift is the diagnostic for
# a 1/tan(theta) vz trend the solenoid z-shift cannot remove.
PARAMS = {
    # field-map displacements [cm]
    "solenoid-z": ("--solenoid-z-shift", "solenoid z-shift"),
    "solenoid-x": ("--solenoid-x-shift", "solenoid x-shift"),
    "solenoid-y": ("--solenoid-y-shift", "solenoid y-shift"),
    "torus-z": ("--torus-z-shift", "torus z-shift"),
    "torus-x": ("--torus-x-shift", "torus x-shift"),
    "torus-y": ("--torus-y-shift", "torus y-shift"),
    # field SCALE (sweep absolute value, e.g. -1.05..-0.95): targets the 1/p
    # (magnetic) part of the vz theta-walk.
    "torus-scale": ("--torus-scale", "torus scale"),
    "solenoid-scale": ("--solenoid-scale", "solenoid scale"),
    # DC start-state translation [cm]: targets the geometric (p-flat) part.
    "dc-x": ("--dc-x-shift", "DC x-shift"),
    "dc-y": ("--dc-y-shift", "DC y-shift"),
    "dc-z": ("--dc-z-shift", "DC z-shift"),
}


def _ptag(e: float) -> str:
    """Momentum-edge tag matching vz-swim-hist's p_tag: integer GeV -> 2-digit
    ('02'); fractional -> one decimal ('0.3')."""
    r = round(e)
    return f"{int(r):02d}" if abs(e - r) < 1e-6 else f"{e:.1f}"


def _key(prefix: str, p_lo: float, p_hi: float, t_lo: int, t_hi: int) -> str:
    return f"{prefix}_p{_ptag(p_lo)}_{_ptag(p_hi)}_th{int(t_lo):02d}_{int(t_hi):02d}"


# ── Job running ──────────────────────────────────────────────────────────────

def vfmt(v: float) -> str:
    """Signed value string with enough decimals to stay unique for fine steps
    (up to 4), but identical to the old '+.2f' for 2-decimal values so existing
    scan folders still resolve. e.g. -1.0->'-1.00', -0.999->'-0.999'."""
    s = f"{v:+.4f}".rstrip("0")
    intpart, _, frac = s.partition(".")
    frac = (frac + "00")[:max(2, len(frac))]
    return f"{intpart}.{frac}"


def job_dir(scan_dir: Path, param: str, val: float) -> Path:
    # Keep the historical "zshift_" prefix for the solenoid-z scan; use the
    # param key for the others.
    prefix = "zshift" if param == "solenoid-z" else param
    return scan_dir / f"{prefix}_{vfmt(val)}"


# Reconstructed-pid -> short species name, used to route each analysis into its
# own output/python/<species>/ tree and to label its plots.
SPECIES_NAMES = {11: "electron", -11: "positron", 211: "pi+", -211: "pi-",
                 321: "K+", -321: "K-", 2212: "proton", 13: "mu-", -13: "mu+"}


def species_name(pid: int) -> str:
    return SPECIES_NAMES.get(pid, f"pid{pid}")


def _base_cmd(args, out: Path) -> list[str]:
    """vz-swim-hist command shared by all jobs (no swept flags yet)."""
    cmd = [
        args.binary, *args.inputs,
        "--output", str(out),
        "--threads", str(args.threads_per_job),
        "--pid", str(args.pid),
        "--dc-region", str(args.dc_region),
        "--torus", args.torus,
        "--solenoid", args.solenoid,
    ]
    # Forward only the CCDB-derived field config that was overridden on the CLI;
    # anything left unset is taken by vz-swim-hist from the tree's CCDB metadata
    # (beam position + field scales, saved by hipo2root --ccdb).
    for flag, value in (("--beam-x", args.beam_x), ("--beam-y", args.beam_y),
                        ("--torus-scale", args.torus_scale),
                        ("--solenoid-scale", args.solenoid_scale)):
        if value is not None:
            cmd += [flag, f"{value}"]
    if args.max_entries is not None:
        cmd += ["--max-entries", str(args.max_entries)]
    return cmd


def _stream_job(cmd: list[str], log: Path, label: str, slots: SlotPool) -> int:
    """Run `cmd`, re-rendering its progress bar as one tqdm bar; log the rest.
    Returns the process return code."""
    slot = slots.acquire()
    bar = (tqdm(total=100, desc=label, unit="%", position=slot,
                leave=False, dynamic_ncols=True)
           if _HAVE_TQDM else None)
    last_bucket = -1

    with open(log, "w") as lf:
        lf.write(" ".join(cmd) + "\n\n")

        def handle(seg: bytes):
            """One \\r/\\n-delimited chunk: progress repaints feed the bar,
            everything else goes to the log."""
            nonlocal last_bucket
            s = ANSI_RE.sub("", seg.decode("utf-8", "replace")).strip()
            if not s:
                return
            m = PCT_RE.search(s)
            if m and ("Swimming" in s or "╢" in s):
                pct = min(100, int(m.group(1)))
                r = RATE_RE.search(s)
                e = ETA_RE.search(s)
                post = "  ".join(x for x in
                                 [r.group(1) if r else None,
                                  f"ETA {e.group(1)}" if e else None] if x)
                if bar is not None:
                    bar.n = pct
                    if post:
                        bar.set_postfix_str(post, refresh=False)
                    bar.refresh()
                elif pct // 5 > last_bucket:  # plain fallback: one line per 5%
                    last_bucket = pct // 5
                    say(f"  {label}: {pct:3d}%  {post}")
                return
            lf.write(s + "\n")

        # bufsize=0 -> raw pipe: read() returns whatever is available.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=0)
        with _PROCS_LOCK:
            _PROCS.add(proc)
        try:
            pending = b""
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                pending += chunk
                while True:
                    m = re.search(rb"[\r\n]", pending)
                    if m is None:
                        break
                    seg, pending = pending[:m.start()], pending[m.end():]
                    handle(seg)
            handle(pending)
            rc = proc.wait()
        finally:
            with _PROCS_LOCK:
                _PROCS.discard(proc)

    if bar is not None:
        bar.n = 100
        bar.refresh()
        bar.close()
    slots.release(slot)
    return rc


def run_job(args, val: float, slots: SlotPool) -> tuple[float, int]:
    """Run one vz-swim-hist instance sweeping the chosen 1-D field parameter to
    `val`. Returns (val, rc)."""
    d = job_dir(Path(args.scan_dir), args.scan_param, val)
    d.mkdir(parents=True, exist_ok=True)
    cmd = _base_cmd(args, d / "vz_bins.root")
    cmd += [PARAMS[args.scan_param][0], f"{val}"]  # the swept parameter
    # When sweeping anything but solenoid-z, hold the solenoid z-shift fixed;
    # forward it only if overridden on the CLI, else vz-swim-hist uses the CCDB
    # value from the tree.
    if args.scan_param != "solenoid-z" and args.solenoid_z_shift is not None:
        cmd += ["--solenoid-z-shift", f"{args.solenoid_z_shift}"]
    rc = _stream_job(cmd, d / "log.txt", f"{args.scan_param} {val:+.2f}", slots)
    return val, rc


def grid_dir(scan_dir: Path, xyz: tuple[float, float, float]) -> Path:
    x, y, z = xyz
    return scan_dir / f"torus_x{vfmt(x)}_y{vfmt(y)}_z{vfmt(z)}"


def run_grid_job(args, xyz: tuple[float, float, float], slots: SlotPool):
    """Run one vz-swim-hist instance with a (torus x, y, z) shift triple.
    Returns (xyz, rc)."""
    x, y, z = xyz
    d = grid_dir(Path(args.scan_dir), xyz)
    d.mkdir(parents=True, exist_ok=True)
    cmd = _base_cmd(args, d / "vz_bins.root")
    cmd += ["--torus-x-shift", f"{x}", "--torus-y-shift", f"{y}", "--torus-z-shift", f"{z}"]
    if args.solenoid_z_shift is not None:
        cmd += ["--solenoid-z-shift", f"{args.solenoid_z_shift}"]
    rc = _stream_job(cmd, d / "log.txt", f"x{x:+.1f} y{y:+.1f} z{z:+.1f}", slots)
    return xyz, rc


# ── Peak extraction & summary ────────────────────────────────────────────────

def find_peak(vals: np.ndarray, centers: np.ndarray, lo: float, hi: float):
    """Position of the maximum bin inside [lo, hi), refined by a 3-point
    parabola; None if the window is empty."""
    sel = (centers >= lo) & (centers < hi)
    if not np.any(sel) or vals[sel].max() <= 0:
        return None
    idx = np.flatnonzero(sel)
    k = idx[np.argmax(vals[idx])]
    if k == 0 or k == len(vals) - 1:
        return float(centers[k])
    y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
    denom = y0 - 2 * y1 + y2
    if denom >= 0:  # not a local maximum (flat/edge) -> no refinement
        return float(centers[k])
    bw = centers[1] - centers[0]
    return float(centers[k] + 0.5 * bw * (y0 - y2) / denom)


def _gaussian(x: np.ndarray, amp: float, mean: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


class Peak2Fit(NamedTuple):
    """Top-of-peak Gaussian fit of the second swum-vz peak for one scan value.

    `mean`/`sigma` and their 1-sigma `*_err` come from scipy.optimize.curve_fit
    (the square root of the covariance diagonal). `lo`/`hi` are the inclusive
    bin indices of the fitted (top-of-peak) slice; `xfit`/`yfit` are the drawn
    fit curve. `ok` is False (every value NaN/empty) for a skipped or failed
    fit."""
    mean: float
    sigma: float
    mean_err: float
    sigma_err: float
    amp: float
    lo: int
    hi: int
    xfit: np.ndarray
    yfit: np.ndarray
    ok: bool


_FAILED_PEAK2 = Peak2Fit(np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0,
                         np.empty(0), np.empty(0), ok=False)


def fit_peak2_gauss(counts: np.ndarray, centers: np.ndarray, lo: float, hi: float,
                    *, top_frac: float = 0.6, smooth_cm: float = 0.4,
                    min_pts: int = 5) -> Peak2Fit:
    """Gaussian fit of the TOP `top_frac` of the second swum-vz peak in [lo, hi].

    The swum-vz spectrum has two overlapping target peaks; the second (nearest
    vz = 0) sits on the shoulder of the first. Locate its maximum inside the
    window on counts smoothed over `smooth_cm` (wide enough to ignore per-bin
    noise), take the contiguous run of bins above `top_frac` x maximum around it,
    then clip each side at the first real valley (an interior minimum of the
    smoothed flank) so the slice can never bleed into the neighbour peak. A pure
    Gaussian is fit (scipy.optimize.curve_fit) to that top-of-peak slice of the
    ORIGINAL counts. With top_frac = 0.6 the threshold stays above the inter-peak
    valley, so the slice is the peak's own near-symmetric cap and mu tracks the
    true centre rather than being dragged onto the contaminated lower flank.

    Returns a Peak2Fit with the mean and sigma and their 1-sigma errors (from the
    covariance); `ok` is False when the window is empty, the slice is too short,
    or the fit does not converge."""
    idx = np.flatnonzero((centers >= lo) & (centers <= hi))
    if idx.size == 0:
        return _FAILED_PEAK2
    # Smooth over ~smooth_cm (an odd number of bins) only to choose the slice;
    # the fit itself uses the raw counts.
    bw = float(centers[1] - centers[0])
    ks = max(3, int(round(smooth_cm / bw)))
    ks += (ks + 1) % 2  # force odd so the moving average is centred
    smooth = np.convolve(counts, np.ones(ks) / ks, mode="same")
    imax = int(idx[np.argmax(smooth[idx])])
    thresh = top_frac * smooth[imax]
    if thresh <= 0:
        return _FAILED_PEAK2
    # Contiguous run above the top-fraction threshold around the maximum...
    klo = imax
    while klo > 0 and smooth[klo - 1] >= thresh:
        klo -= 1
    khi = imax
    while khi < len(smooth) - 1 and smooth[khi + 1] >= thresh:
        khi += 1
    # ...clipped at the first real valley on each side (the interior minimum of
    # the smoothed flank); a no-op for an isolated peak, but it stops the slice
    # from crossing a deep valley into the neighbour peak when the two overlap
    # enough that the valley rises above the threshold.
    klo += int(np.argmin(smooth[klo:imax + 1]))
    khi = imax + int(np.argmin(smooth[imax:khi + 1]))
    x, y = centers[klo:khi + 1], counts[klo:khi + 1]
    if x.size < min_pts:
        return _FAILED_PEAK2
    # Seed amplitude/centre from the peak bin and sigma from the slice width
    # (the slice spans ~ the FWHM, so sigma ~ width / 2.355).
    amp0 = float(counts[imax])
    mean0 = float(centers[imax])
    width = float(centers[khi] - centers[klo])
    sigma0 = max(width / 2.355, 0.1)
    # Bound the centre to the fitted slice and sigma to its width, so a poor
    # slice can never rail the fit out to the window/axis edges.
    try:
        popt, pcov = curve_fit(
            _gaussian, x, y, p0=[amp0, mean0, sigma0],
            bounds=([0.0, float(centers[klo]), 0.05], [np.inf, float(centers[khi]), width]),
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return _FAILED_PEAK2
    sigma = abs(float(popt[2]))
    # Reject a non-peak: a real peak2 is < ~3 cm wide, and a Gaussian as wide as
    # its own fitting slice (sigma railed to the bound) means the slice held no
    # resolved peak -- e.g. the lowest-theta bin where the two foils merge.
    if sigma > 3.0 or sigma >= 0.9 * width:
        return _FAILED_PEAK2
    perr = np.sqrt(np.diag(pcov))  # 1-sigma parameter errors
    xf = np.linspace(float(x[0]), float(x[-1]), 200)
    yf = _gaussian(xf, *popt)
    return Peak2Fit(float(popt[1]), sigma, float(perr[1]), float(perr[2]),
                    float(popt[0]), klo, khi, xf, yf, ok=True)


def swum_peaks(root_file: Path) -> dict[tuple[int, str], float]:
    """{(theta_lo, peak_name): position} from the p-integrated swum-vz
    histograms (theta bins below MIN_ENTRIES are skipped)."""
    peaks: dict[tuple[int, str], float] = {}
    with uproot.open(root_file) as f:
        for t_lo, t_hi in TH_BINS:
            tot, edges = None, None
            for p_lo, p_hi in P_BINS:
                vals, edges = f[_key("vz_swum", p_lo, p_hi, t_lo, t_hi)].to_numpy()
                tot = vals if tot is None else tot + vals
            if tot.sum() < MIN_ENTRIES:
                continue
            centers = 0.5 * (edges[:-1] + edges[1:])
            for name, lo, hi in PEAK_WINDOWS:
                pos = find_peak(tot, centers, lo, hi)
                if pos is not None:
                    peaks[(t_lo, name)] = pos
    return peaks


def peak_rms(peaks: dict[tuple[int, str], float]) -> float | None:
    """Alignment metric: mean over the two peaks of the RMS of their position
    across theta bins (lower = better aligned). None if no peak has >=2 bins."""
    rms = []
    for name, _, _ in PEAK_WINDOWS:
        pos = [p for (t, n), p in peaks.items() if n == name]
        if len(pos) >= 2:
            rms.append(float(np.std(pos)))
    return float(np.mean(rms)) if rms else None


def make_overlays(scan_dir: Path, param: str, vals: list[float]):
    """A vz_theta_overlay.pdf (rec page + swum page) in each value's subfolder,
    titled with that parameter value. Reuses the plotter from plot_vz_bins.py."""
    from plot_vz_bins import plot_theta_overlay  # pulls in the mplhep style once

    label = PARAMS[param][1]
    for v in vals:
        d = job_dir(scan_dir, param, v)
        rf = d / "vz_bins.root"
        if not rf.exists():
            continue
        plot_theta_overlay(str(rf), out_path=str(d / "vz_theta_overlay.pdf"),
                           tag=f"{label} {v:+.2f} cm")


def write_summary(scan_dir: Path, param: str, vals: list[float]):
    """<param>_peaks.csv + <param>_summary.pdf from the per-value outputs."""
    label = PARAMS[param][1]
    # peaks[val] = {(theta_lo, peak_name): position}
    peaks: dict[float, dict[tuple[int, str], float]] = {}
    for v in vals:
        rf = job_dir(scan_dir, param, v) / "vz_bins.root"
        if rf.exists():
            peaks[v] = swum_peaks(rf)
        else:
            print(f"[warn] missing {rf}, skipping {param} {v:+.2f}")
    if not peaks:
        sys.exit("error: no scan outputs found to summarize")

    csv_path = scan_dir / f"{param}_peaks.csv"
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow([param, "theta_lo", "peak", "vz_peak_cm"])
        for v, d in sorted(peaks.items()):
            for (t_lo, name), pos in sorted(d.items()):
                w.writerow([vfmt(v), t_lo, name, f"{pos:.4f}"])
    print(f"Saved {csv_path}")

    # Alignment metric per value: RMS of the peak position across theta bins,
    # averaged over the two peaks.
    tmid = {t_lo: 0.5 * (t_lo + t_hi) for t_lo, t_hi in TH_BINS}
    vals_sorted = sorted(peaks)
    metric = {v: m for v in vals_sorted if (m := peak_rms(peaks[v])) is not None}
    best = min(metric, key=metric.get) if metric else None

    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(vals_sorted)))
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for (name, _, _), ax in zip(PEAK_WINDOWS, axs[:2]):
        for v, color in zip(vals_sorted, colors):
            th = sorted(t for (t, n) in peaks[v] if n == name)
            if not th:
                continue
            ax.plot([tmid[t] for t in th], [peaks[v][(t, name)] for t in th],
                    marker="o", ms=3, lw=1.1, color=color, label=f"{vfmt(v)}")
        ax.set_xlabel(r"$\theta$ [deg]")
        ax.set_ylabel(rf"$v_z$ {name} position [cm]")
        ax.set_title(f"swum-$v_z$ {name} vs " r"$\theta$")
    axs[1].legend(fontsize=7, ncol=2, title=label)

    ax = axs[2]
    ms = sorted(metric)
    ax.plot(ms, [metric[v] for v in ms], marker="o", color="tab:red")
    if best is not None:
        ax.axvline(best, color="gray", ls=":",
                   label=f"best: {vfmt(best)} (RMS {metric[best]:.3f})")
        ax.legend(fontsize=9)
    ax.set_xlabel(f"{label} [cm]")
    ax.set_ylabel(r"RMS of peak position across $\theta$ [cm]")
    ax.set_title(f"peak alignment vs {label}")

    fig.tight_layout()
    out = scan_dir / f"{param}_summary.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    if best is not None:
        print(f"Best alignment: {label} {vfmt(best)} "
              f"(mean peak RMS across theta = {metric[best]:.3f} cm)")


def write_peak2_fits(scan_dir: Path, param: str, vals: list[float]):
    """<param>_peak2_fit.pdf + .csv: per scan value and per (momentum, theta)
    bin, a Gaussian fit of the top of the SECOND (downstream, ~ -3 cm) swum-vz
    peak with its mean and sigma and their 1-sigma errors.

    For each scan value the top of peak2 is fit in every (p, theta) cell of the
    grid (see fit_peak2_gauss). A per-value page shows the fit in every cell
    (rows = momentum, cols = theta, one tab10 colour per theta as in
    plot_vz_bins.py); the summary pages plot peak2 mean and sigma versus the
    scanned parameter, one panel per momentum bin and one colour per theta."""
    label = PARAMS[param][1]
    species = scan_dir.parent.name  # output/python/<species>/<param>_scan
    _, lo2, hi2 = PEAK_WINDOWS[1]  # ("peak2", -5.5, -0.5)
    th_colors = [plt.cm.tab10(i % 10) for i in range(len(TH_BINS))]
    th_labels = [rf"${t_lo}^\circ \leq \theta < {t_hi}^\circ$" for t_lo, t_hi in TH_BINS]
    n_p, n_th = len(P_BINS), len(TH_BINS)
    fits: dict[float, list[list[Peak2Fit]]] = {}  # value -> Peak2Fit per [ip][it] cell

    zlo, zhi = -9.0, 0.5  # zoom each panel onto peak2 (peak1 sits out near -10)
    out = scan_dir / f"{param}_peak2_fit.pdf"
    with PdfPages(out) as pdf:
        for v in sorted(vals):
            rf = job_dir(scan_dir, param, v) / "vz_bins.root"
            if not rf.exists():
                continue
            cells, edges = {}, None
            with uproot.open(rf) as f:
                for ip, (p_lo, p_hi) in enumerate(P_BINS):
                    for it, (t_lo, t_hi) in enumerate(TH_BINS):
                        key = _key("vz_swum", p_lo, p_hi, t_lo, t_hi)
                        if key in f:
                            hv, edges = f[key].to_numpy()
                            cells[(ip, it)] = hv
            if edges is None:
                continue
            centers = 0.5 * (edges[:-1] + edges[1:])
            bw = centers[1] - centers[0]
            zoom = (centers >= zlo) & (centers <= zhi)

            # Fit every (p, theta) cell and draw it (rows = p, cols = theta).
            vfits = [[_FAILED_PEAK2] * n_th for _ in range(n_p)]
            fig, axs = plt.subplots(n_p, n_th, figsize=(1.5 * n_th, 1.2 * n_p),
                                    squeeze=False, sharex=True)
            for ip, (p_lo, p_hi) in enumerate(P_BINS):
                for it, (t_lo, t_hi) in enumerate(TH_BINS):
                    ax = axs[ip][it]
                    ax.set_xticks([]); ax.set_yticks([])
                    tot = cells.get((ip, it))
                    if tot is not None:
                        fit = (fit_peak2_gauss(tot, centers, lo2, hi2)
                               if tot.sum() >= MIN_ENTRIES else _FAILED_PEAK2)
                        vfits[ip][it] = fit
                        ax.bar(centers[zoom], tot[zoom], width=bw, color="0.86",
                               edgecolor="none", rasterized=True)
                        if fit.ok:
                            ax.plot(fit.xfit, fit.yfit, color=th_colors[it], lw=1.0)
                        ax.set_xlim(zlo, zhi)
                    if ip == 0:
                        ax.set_title(rf"${t_lo}$-${t_hi}^\circ$", color=th_colors[it], fontsize=6)
                    if it == 0:
                        ax.set_ylabel(rf"${p_lo}$-${p_hi}$", fontsize=6)
            fits[v] = vfits
            fig.suptitle(rf"{species} — {label} {vfmt(v)} — peak2 fit per $(p,\theta)$"
                         r"  (rows: $p$ [GeV/$c$], cols: $\theta$)", fontsize=10)
            fig.supxlabel(r"swum $v_z$ [cm]", fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Summary: peak2 mean and sigma vs the scanned value -- one panel per
        # momentum bin, one tab10 colour per theta bin.
        if fits:
            vs = sorted(fits)
            pncol = 4
            pnrow = (n_p + pncol - 1) // pncol
            for attr, errattr, ylabel, ttl in [
                    ("mean", "mean_err", r"peak2 $\mu$ [cm]", "mean"),
                    ("sigma", "sigma_err", r"peak2 $\sigma$ [cm]", "width")]:
                fig, axs = plt.subplots(pnrow, pncol, figsize=(3.3 * pncol, 2.9 * pnrow),
                                        squeeze=False, sharex=True)
                for ip, (p_lo, p_hi) in enumerate(P_BINS):
                    ax = axs[ip // pncol][ip % pncol]
                    for it in range(n_th):
                        y = np.array([getattr(fits[v][ip][it], attr)
                                      if fits[v][ip][it].ok else np.nan for v in vs])
                        if not np.isfinite(y).any():
                            continue
                        ye = np.array([getattr(fits[v][ip][it], errattr)
                                       if fits[v][ip][it].ok else 0.0 for v in vs])
                        ax.errorbar(vs, y, yerr=ye, marker="o", ms=2, lw=0.8, capsize=1.5,
                                    elinewidth=0.5, color=th_colors[it],
                                    label=th_labels[it] if ip == 0 else None)
                    ax.set_title(rf"${p_lo} < p < {p_hi}$ GeV/$c$", fontsize=9)
                    ax.tick_params(labelsize=6)
                    if ip // pncol == pnrow - 1:
                        ax.set_xlabel(label, fontsize=8)
                    if ip % pncol == 0:
                        ax.set_ylabel(ylabel, fontsize=8)
                for j in range(n_p, pnrow * pncol):      # hide any unused panels
                    axs[j // pncol][j % pncol].set_visible(False)
                h, l = axs[0][0].get_legend_handles_labels()
                fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, frameon=False,
                           title=r"$\theta$ bin", bbox_to_anchor=(0.5, -0.01))
                fig.suptitle(rf"{species} — 2nd-peak {ttl} vs {label}, per $(p,\theta)$", fontsize=12)
                fig.tight_layout(rect=(0, 0.08, 1, 0.96))
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    print(f"Saved {out}")

    csv_path = scan_dir / f"{param}_peak2_fit.csv"
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow([param, "p_lo", "p_hi", "theta_lo", "theta_hi", "peak2_mean_cm",
                    "peak2_mean_err_cm", "peak2_sigma_cm", "peak2_sigma_err_cm"])
        for v in sorted(fits):
            for ip, (p_lo, p_hi) in enumerate(P_BINS):
                for it, (t_lo, t_hi) in enumerate(TH_BINS):
                    ft = fits[v][ip][it]
                    if ft.ok:
                        w.writerow([vfmt(v), p_lo, p_hi, t_lo, t_hi,
                                    f"{ft.mean:.4f}", f"{ft.mean_err:.4f}",
                                    f"{ft.sigma:.4f}", f"{ft.sigma_err:.4f}"])
    print(f"Saved {csv_path}")


def write_grid_summary(scan_dir: Path, combos: list[tuple[float, float, float]]):
    """torus_grid.csv + torus_grid_summary.pdf from a 3-D (x,y,z) torus scan:
    the scalar alignment metric per combo, the best combo, and 2-D metric
    heatmaps sliced through it."""
    metric: dict[tuple[float, float, float], float] = {}
    for xyz in combos:
        rf = grid_dir(scan_dir, xyz) / "vz_bins.root"
        if not rf.exists():
            continue
        m = peak_rms(swum_peaks(rf))
        if m is not None:
            metric[xyz] = m
    if not metric:
        sys.exit("error: no grid outputs found to summarize")

    csv_path = scan_dir / "torus_grid.csv"
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["torus_x", "torus_y", "torus_z", "peak_rms_cm"])
        for (x, y, z), m in sorted(metric.items()):
            w.writerow([vfmt(x), vfmt(y), vfmt(z), f"{m:.4f}"])
    print(f"Saved {csv_path}")

    best = min(metric, key=metric.get)
    bx, by, bz = best
    xs = sorted({x for x, _, _ in metric})
    ys = sorted({y for _, y, _ in metric})
    zs = sorted({z for _, _, z in metric})

    def slice_grid(fixed_axis: int, fixed_val: float, ax_a: list, ax_b: list):
        """metric[a, b] on the plane where axis `fixed_axis` == fixed_val."""
        g = np.full((len(ax_b), len(ax_a)), np.nan)
        for ia, a in enumerate(ax_a):
            for ib, b in enumerate(ax_b):
                key = {0: (fixed_val, a, b), 1: (a, fixed_val, b), 2: (a, b, fixed_val)}[fixed_axis]
                if key in metric:
                    g[ib, ia] = metric[key]
        return g

    # Three planes through the best point: (x,y)@bz, (x,z)@by, (y,z)@bx.
    planes = [
        ("torus x [cm]", "torus y [cm]", xs, ys, slice_grid(2, bz, xs, ys), bx, by, f"z={bz:+.1f}"),
        ("torus x [cm]", "torus z [cm]", xs, zs, slice_grid(1, by, xs, zs), bx, bz, f"y={by:+.1f}"),
        ("torus y [cm]", "torus z [cm]", ys, zs, slice_grid(0, bx, ys, zs), by, bz, f"x={bx:+.1f}"),
    ]
    vmin = min(metric.values())
    vmax = max(metric.values())
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.4))
    for ax, (xl, yl, axa, axb, g, ma, mb, ttl) in zip(axs, planes):
        im = ax.imshow(g, origin="lower", aspect="auto", cmap="viridis_r",
                       vmin=vmin, vmax=vmax,
                       extent=[min(axa), max(axa), min(axb), max(axb)])
        ax.plot([ma], [mb], marker="*", ms=16, color="red", mec="white")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"peak RMS, {ttl}")
        fig.colorbar(im, ax=ax, label="mean peak RMS across " r"$\theta$ [cm]")
    fig.suptitle(f"torus shift grid — best (x,y,z)=({bx:+.1f},{by:+.1f},{bz:+.1f}) cm, "
                 f"RMS={metric[best]:.3f} cm")
    fig.tight_layout()
    out = scan_dir / "torus_grid_summary.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print(f"Best alignment: torus shift (x,y,z) = ({bx:+.2f}, {by:+.2f}, {bz:+.2f}) cm "
          f"(mean peak RMS across theta = {metric[best]:.3f} cm)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Scan a field-map displacement with parallel vz-swim-hist "
                    "runs and summarize the swum-vz peak alignment across theta.")
    p.add_argument("inputs", nargs="*", default=["output/cpp/particles.root"],
                   help="input particles.root file(s) (default: output/cpp/particles.root)")
    p.add_argument("--scan-param", default="solenoid-z", choices=sorted(PARAMS),
                   help="which field displacement to scan (default: solenoid-z)")
    p.add_argument("--grid", action="store_true",
                   help="3-D grid scan of the torus (x,y,z) shifts together over the "
                        "same --z-min/max/step range; ignores --scan-param. WARNING: "
                        "N^3 runs.")
    p.add_argument("--binary", default="./build/vz-swim-hist",
                   help="vz-swim-hist executable (default: ./build/vz-swim-hist)")
    p.add_argument("--scan-dir", default=None,
                   help="base output folder (default: <scan-param>_scan)")
    p.add_argument("--z-min", type=float, default=-6.0)
    p.add_argument("--z-max", type=float, default=0.0)
    p.add_argument("--z-step", type=float, default=0.5)
    p.add_argument("--shifts", type=float, nargs="+",
                   help="explicit value list [cm]; overrides --z-min/max/step")
    p.add_argument("--max-parallel", type=int, default=10,
                   help="max concurrent vz-swim-hist instances (default: 10)")
    p.add_argument("--threads-per-job", type=int, default=5,
                   help="--threads passed to each instance (default: 5)")
    p.add_argument("--max-entries", type=int, default=None,
                   help="process only the first N tree entries (particles) per run "
                        "(e.g. 10000000); default: all. Speeds up scans hugely.")
    # CCDB-derived field config: leaving these unset (None) makes vz-swim-hist
    # take them from the tree's CCDB metadata (saved by hipo2root --ccdb); pass
    # one only to override the CCDB value for the scan.
    p.add_argument("--beam-x", type=float, default=None,
                   help="beam x offset [cm] (default: from the tree's CCDB metadata)")
    p.add_argument("--beam-y", type=float, default=None,
                   help="beam y offset [cm] (default: from the tree's CCDB metadata)")
    p.add_argument("--dc-region", type=int, default=3, choices=(1, 3))
    p.add_argument("--pid", type=int, default=11,
                   help="species to analyze by reconstructed pid (11 = e-, 211 = pi+, "
                        "-211 = pi-); routes output to output/python/<species>/")
    p.add_argument("--torus", default="Full_torus_r501_phi361_z501_31Mar2021.dat")
    p.add_argument("--solenoid", default="Symm_solenoid_r601_phi1_z1201_21May2019.dat")
    # Field scales in OUR FieldMap sign convention = -(RUN::config); for RG-D
    # 18614 the CCDB metadata gives torus -1 / solenoid +1 (see docs/coatjava_vz.md).
    # Default None -> taken from the tree's CCDB metadata.
    p.add_argument("--torus-scale", type=float, default=None,
                   help="torus field scale (default: from the tree's CCDB metadata)")
    p.add_argument("--solenoid-scale", type=float, default=None,
                   help="solenoid field scale (default: from the tree's CCDB metadata)")
    # Held fixed when scanning a torus shift (ignored when scanning solenoid-z).
    p.add_argument("--solenoid-z-shift", type=float, default=None,
                   help="solenoid z-shift [cm] held fixed while scanning a torus shift "
                        "(default: from the tree's CCDB metadata)")
    p.add_argument("--force", action="store_true",
                   help="re-run values whose vz_bins.root already exists")
    p.add_argument("--plot-only", action="store_true",
                   help="skip the runs, only rebuild the summary from existing outputs")
    args = p.parse_args()

    # Adopt the species-dependent (p, theta) grid (pions use a softer momentum
    # grid than electrons; in sync with vz-swim-hist make_grid).
    global P_EDGES, P_BINS
    P_EDGES = momentum_edges(args.pid)
    P_BINS = list(zip(P_EDGES[:-1], P_EDGES[1:]))

    if args.shifts:
        vals = sorted(set(args.shifts))
    else:
        n = int(round((args.z_max - args.z_min) / args.z_step)) + 1
        vals = [round(args.z_min + i * args.z_step, 4) for i in range(n)]

    if args.grid:
        run_grid(args, vals)
    else:
        run_1d(args, vals)


def parallel_run(args, todo: list, submit_fn, fmt_fn):
    """Run `submit_fn(args, item, slots)` for each item, <= max_parallel at a
    time. submit_fn returns (item, rc). Returns the list of failed items.
    fmt_fn(item) -> short label for the completion line."""
    failed = []
    slots = SlotPool(args.max_parallel)
    pool = ThreadPoolExecutor(max_workers=args.max_parallel)
    try:
        futures = [pool.submit(submit_fn, args, it, slots) for it in todo]
        done = 0
        for fut in as_completed(futures):
            item, rc = fut.result()
            done += 1
            tag = "ok" if rc == 0 else f"FAILED (rc={rc})"
            say(f"  [{done}/{len(todo)}] {fmt_fn(item)}: {tag}")
            if rc != 0:
                failed.append(item)
        pool.shutdown(wait=True)
    except KeyboardInterrupt:
        say("\ninterrupted: terminating running jobs, cancelling queued ones...")
        pool.shutdown(wait=False, cancel_futures=True)
        _kill_children()
        sys.exit(130)
    if failed:
        say(f"warning: {len(failed)} job(s) failed (see their log.txt)")
    return failed


def run_1d(args, vals: list[float]):
    param = args.scan_param
    scan_dir = (Path(args.scan_dir) if args.scan_dir
                else Path("output", "python", species_name(args.pid), f"{param}_scan"))
    args.scan_dir = str(scan_dir)  # so run_job (reads args.scan_dir) resolves it
    scan_dir.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        todo = [v for v in vals
                if args.force or not (job_dir(scan_dir, param, v) / "vz_bins.root").exists()]
        skipped = len(vals) - len(todo)
        if skipped:
            say(f"{skipped} value(s) already done (use --force to re-run)")
        say(f"scanning {param} over {len(todo)} value(s), <= {args.max_parallel} in parallel, "
            f"{args.threads_per_job} threads each")
        parallel_run(args, todo, run_job, lambda v: f"{param} {v:+.2f}")

    make_overlays(scan_dir, param, vals)
    write_summary(scan_dir, param, vals)
    write_peak2_fits(scan_dir, param, vals)


def run_grid(args, vals: list[float]):
    import itertools
    scan_dir = (Path(args.scan_dir) if args.scan_dir
                else Path("output", "python", species_name(args.pid), "torus_grid_scan"))
    args.scan_dir = str(scan_dir)  # so run_grid_job (reads args.scan_dir) resolves it
    scan_dir.mkdir(parents=True, exist_ok=True)
    combos = [tuple(c) for c in itertools.product(vals, vals, vals)]

    if not args.plot_only:
        todo = [c for c in combos
                if args.force or not (grid_dir(scan_dir, c) / "vz_bins.root").exists()]
        skipped = len(combos) - len(todo)
        if skipped:
            say(f"{skipped} combo(s) already done (use --force to re-run)")
        say(f"torus (x,y,z) grid: {len(todo)} of {len(combos)} run(s) "
            f"({len(vals)}^3), <= {args.max_parallel} in parallel, "
            f"{args.threads_per_job} threads each")
        parallel_run(args, todo, run_grid_job,
                     lambda c: f"x{c[0]:+.1f} y{c[1]:+.1f} z{c[2]:+.1f}")

    write_grid_summary(scan_dir, combos)


if __name__ == "__main__":
    main()
