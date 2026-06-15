#!/usr/bin/env python3
"""Field-displacement scan for the electron-vz swim.

Runs vz-swim-hist once per swept value of a chosen field displacement
(--scan-param: solenoid-z, torus-z, torus-x, torus-y) at most --max-parallel
instances at a time, --threads-per-job threads each, with one live progress
bar per running job (tqdm; plain prints without it), one subfolder per value:

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

Usage (on the farm, from the repo root, after hipo2root):
  # geometric (p-flat) part of the vz theta-walk -> DC alignment:
  python scan_zshift.py particles.root --beam-y -0.18 \
      --scan-param dc-z --z-min -2 --z-max 2 --z-step 0.2
  # 1/p (magnetic) part -> field SCALE (sweep absolute value near nominal):
  python scan_zshift.py particles.root --beam-y -0.18 \
      --scan-param torus-scale --shifts -1.06 -1.04 -1.02 -1.00 -0.98 -0.96
  python scan_zshift.py --scan-param dc-z --plot-only          # redo summary only
"""
import argparse
import csv
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import uproot
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
P_EDGES = np.arange(2, 10 + 1, 1)        # 2,3,...,10 GeV -> 8 bins
TH_EDGES = np.arange(6, 26 + 1, 2)       # 6..26 deg      -> 10 bins
P_BINS = list(zip(P_EDGES[:-1], P_EDGES[1:]))
TH_BINS = list(zip(TH_EDGES[:-1], TH_EDGES[1:]))

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


def _key(prefix: str, p_lo: int, p_hi: int, t_lo: int, t_hi: int) -> str:
    return f"{prefix}_p{p_lo:02d}_{p_hi:02d}_th{t_lo:02d}_{t_hi:02d}"


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


def _base_cmd(args, out: Path) -> list[str]:
    """vz-swim-hist command shared by all jobs (no swept flags yet)."""
    cmd = [
        args.binary, *args.inputs,
        "--output", str(out),
        "--threads", str(args.threads_per_job),
        "--beam-x", f"{args.beam_x}",
        "--beam-y", f"{args.beam_y}",
        "--dc-region", str(args.dc_region),
        "--torus", args.torus,
        "--solenoid", args.solenoid,
        "--torus-scale", f"{args.torus_scale}",
        "--solenoid-scale", f"{args.solenoid_scale}",
    ]
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
    # When sweeping a torus shift, hold the solenoid z-shift at its established
    # value (the C++ default is already -3, but pass it explicitly for the log).
    if args.scan_param != "solenoid-z":
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
    cmd += ["--torus-x-shift", f"{x}", "--torus-y-shift", f"{y}",
            "--torus-z-shift", f"{z}", "--solenoid-z-shift", f"{args.solenoid_z_shift}"]
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


def fit_peak2_top(vals: np.ndarray, centers: np.ndarray, lo: float, hi: float,
                  top_frac: float = 0.5):
    """Gaussian fit of the TOP of the peak in [lo, hi).

    A Gaussian is a parabola in log-counts: ln y = c0 + c1 x + c2 x^2, so a
    quadratic least-squares fit to ln(counts) of the bins near the maximum gives
    mean = -c1 / (2 c2) and sigma = sqrt(-1 / (2 c2)) with no curve_fit / scipy.
    Only bins at >= top_frac * peak-height (and > 0) are used, so it characterises
    the top of the peak rather than its tails. Returns
    (mean, sigma, amp, x_fit, y_fit) or None when there is no usable peak."""
    sel = (centers >= lo) & (centers < hi)
    if not np.any(sel):
        return None
    idx = np.flatnonzero(sel)
    pk = vals[idx].max()
    if pk <= 0:
        return None
    kmax = idx[np.argmax(vals[idx])]
    top = idx[(vals[idx] >= top_frac * pk) & (vals[idx] > 0)]
    if top.size < 3:  # fall back to the 5 bins straddling the maximum
        top = np.array([k for k in range(kmax - 2, kmax + 3)
                        if 0 <= k < len(vals) and vals[k] > 0])
    if top.size < 3:
        return None
    x, y = centers[top], vals[top]
    c2, c1, c0 = np.polyfit(x, np.log(y), 2)
    if c2 >= 0:  # upward parabola -> not a peak
        return None
    mean = -c1 / (2.0 * c2)
    sigma = float(np.sqrt(-1.0 / (2.0 * c2)))
    if not (lo <= mean < hi) or not np.isfinite(sigma):
        return None
    amp = float(np.exp(c0 - c1 * c1 / (4.0 * c2)))  # height at x = mean
    xf = np.linspace(float(x.min()), float(x.max()), 200)
    yf = amp * np.exp(-((xf - mean) ** 2) / (2.0 * sigma ** 2))
    return float(mean), sigma, amp, xf, yf


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
    """<param>_peak2_fit.pdf + .csv: a Gaussian fit of the top of the SECOND
    (downstream, ~ -3 cm) swum-vz peak with its mean and sigma.

    For each scan value the swum-vz histogram is integrated over the whole
    (p, theta) grid, the top of peak2 is fit (see fit_peak2_top), and a page
    shows the spectrum, the fitted Gaussian, and the extracted mean/sigma. A
    final page plots peak2 mean and sigma versus the scanned parameter."""
    label = PARAMS[param][1]
    _, lo2, hi2 = PEAK_WINDOWS[1]  # ("peak2", -5.5, -0.5)
    fits: dict[float, tuple[float, float]] = {}  # value -> (mean, sigma)

    out = scan_dir / f"{param}_peak2_fit.pdf"
    with PdfPages(out) as pdf:
        for v in sorted(vals):
            rf = job_dir(scan_dir, param, v) / "vz_bins.root"
            if not rf.exists():
                continue
            tot, edges = None, None
            with uproot.open(rf) as f:
                for p_lo, p_hi in P_BINS:
                    for t_lo, t_hi in TH_BINS:
                        key = _key("vz_swum", p_lo, p_hi, t_lo, t_hi)
                        if key not in f:
                            continue
                        hv, edges = f[key].to_numpy()
                        tot = hv if tot is None else tot + hv
            if tot is None or tot.sum() < MIN_ENTRIES:
                continue
            centers = 0.5 * (edges[:-1] + edges[1:])
            fit = fit_peak2_top(tot, centers, lo2, hi2)

            fig, ax = plt.subplots(figsize=(7, 5))
            bw = centers[1] - centers[0]
            ax.bar(centers, tot, width=bw, color="0.85", edgecolor="0.6", lw=0.3,
                   label=r"swum $v_z$ (p, $\theta$ integrated)")
            ax.axvspan(lo2, hi2, color="tab:blue", alpha=0.07, label="peak2 window")
            if fit is not None:
                mean, sigma, amp, xf, yf = fit
                fits[v] = (mean, sigma)
                ax.plot(xf, yf, color="tab:red", lw=2.0, label="Gaussian fit (top)")
                ax.axvline(mean, color="tab:red", ls=":", lw=1.0)
                ax.text(0.025, 0.97,
                        f"peak2 Gaussian fit\n$\\mu$ = {mean:.3f} cm\n$\\sigma$ = {sigma:.3f} cm",
                        transform=ax.transAxes, va="top", ha="left", fontsize=10,
                        bbox=dict(boxstyle="round", fc="white", ec="0.7"))
            else:
                ax.text(0.025, 0.97, "peak2 fit failed", transform=ax.transAxes,
                        va="top", ha="left", color="tab:red")
            ax.set_xlabel(r"swum $v_z$ [cm]")
            ax.set_ylabel("counts")
            ax.set_title(f"{label} {v:+.2f} cm — 2nd-peak Gaussian fit")
            ax.legend(fontsize=8, loc="upper right")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if fits:
            vs = sorted(fits)
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
            a1.plot(vs, [fits[v][0] for v in vs], marker="o", color="tab:red")
            a1.set_xlabel(f"{label} [cm]"); a1.set_ylabel(r"peak2 $\mu$ [cm]")
            a1.set_title(r"2nd-peak mean vs " + label)
            a2.plot(vs, [fits[v][1] for v in vs], marker="o", color="tab:blue")
            a2.set_xlabel(f"{label} [cm]"); a2.set_ylabel(r"peak2 $\sigma$ [cm]")
            a2.set_title(r"2nd-peak width vs " + label)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved {out}")

    csv_path = scan_dir / f"{param}_peak2_fit.csv"
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow([param, "peak2_mean_cm", "peak2_sigma_cm"])
        for v in sorted(fits):
            w.writerow([vfmt(v), f"{fits[v][0]:.4f}", f"{fits[v][1]:.4f}"])
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
    # Passed through to vz-swim-hist (defaults match the C++).
    p.add_argument("--beam-x", type=float, default=0.0)
    p.add_argument("--beam-y", type=float, default=0.0)
    p.add_argument("--dc-region", type=int, default=3, choices=(1, 3))
    p.add_argument("--torus", default="Full_torus_r501_phi361_z501_31Mar2021.dat")
    p.add_argument("--solenoid", default="Symm_solenoid_r601_phi1_z1201_21May2019.dat")
    # Scales in OUR FieldMap sign convention = -(RUN::config) for both magnets
    # (see docs/coatjava_vz.md); torus -1 / solenoid +1 reproduces the pass1
    # rec vz for RG-D 18614 (RUN::config torus +1 / solenoid -1).
    p.add_argument("--torus-scale", type=float, default=-1.0)
    p.add_argument("--solenoid-scale", type=float, default=1.0)
    # Held fixed when scanning a torus shift (ignored when scanning solenoid-z).
    p.add_argument("--solenoid-z-shift", type=float, default=-3.0,
                   help="solenoid z-shift held fixed while scanning a torus shift")
    p.add_argument("--force", action="store_true",
                   help="re-run values whose vz_bins.root already exists")
    p.add_argument("--plot-only", action="store_true",
                   help="skip the runs, only rebuild the summary from existing outputs")
    args = p.parse_args()

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
    scan_dir = Path(args.scan_dir) if args.scan_dir else Path("output", "python", f"{param}_scan")
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
    scan_dir = Path(args.scan_dir) if args.scan_dir else Path("output", "python", "torus_grid_scan")
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
