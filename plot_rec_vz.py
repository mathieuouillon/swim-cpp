#!/usr/bin/env python3
"""Reconstructed REC::Particle v_z in (p, theta) bins, straight from the
particles.root tree -- NO swim and NO swim-acceptance gate (only forward +
species). This is the reference for the swum-vz overlay: it shows the raw
per-cell statistics and the target-foil structure that are actually present in
the data, independent of the beamline swim.

  uv run plot_rec_vz.py output/cpp/particles.root --pid 211 -n 100000000
  uv run plot_rec_vz.py output/cpp/particles.root --pid -211          # pi-
"""
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import uproot  # noqa: E402


def species_grid(pid: int):
    """(p_edges, theta_edges), matching vz-swim-hist make_grid."""
    if abs(pid) == 211:
        return np.array([0.3, 1, 2, 3, 4, 5, 6.0]), np.arange(6, 42 + 1, 3.0)
    return np.arange(2, 10 + 1, 1.0), np.arange(6, 26 + 1, 2.0)


def theta_colors(n: int):
    if n <= 10:
        return [plt.cm.tab10(i) for i in range(n)]
    return [plt.cm.turbo(x) for x in np.linspace(0.05, 0.95, n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="particles.root from hipo2root")
    ap.add_argument("--pid", type=int, default=211, help="species (211 pi+, -211 pi-, 11 e-)")
    ap.add_argument("-n", "--max-entries", type=int, default=20_000_000,
                    help="tree entries to scan (-1 = all)")
    ap.add_argument("-o", "--out", default="rec_vz_theta_overlay.pdf")
    args = ap.parse_args()

    p_edges, th_edges = species_grid(args.pid)
    n_p, n_th = len(p_edges) - 1, len(th_edges) - 1
    VZMIN, VZMAX, NB = -13.0, 0.0, 200
    vz_edges = np.linspace(VZMIN, VZMAX, NB + 1)
    counts = np.zeros((n_p * n_th, NB))  # flat (cell, vz-bin)

    seen = 0
    cap = args.max_entries if args.max_entries >= 0 else None
    with uproot.open(args.file) as f:
        tree = f["particles"]
        for ch in tree.iterate(["pid", "status", "px", "py", "pz", "vz"],
                               step_size="200 MB", library="np"):
            take = len(ch["pid"]) if cap is None else min(len(ch["pid"]), cap - seen)
            s = slice(0, take)
            pid, st = ch["pid"][s], ch["status"][s]
            px, py, pz, vz = ch["px"][s], ch["py"][s], ch["pz"][s], ch["vz"][s]
            seen += take
            a = np.abs(st)
            m = (pid == args.pid) & (a >= 2000) & (a < 4000)
            if m.any():
                p = np.sqrt(px * px + py * py + pz * pz)
                with np.errstate(invalid="ignore", divide="ignore"):
                    th = np.degrees(np.arccos(np.clip(np.where(p > 0, pz / p, 0), -1, 1)))
                pb = np.digitize(p, p_edges) - 1
                tb = np.digitize(th, th_edges) - 1
                vzb = np.digitize(vz, vz_edges) - 1
                ok = (m & (pb >= 0) & (pb < n_p) & (tb >= 0) & (tb < n_th)
                      & (vzb >= 0) & (vzb < NB) & np.isfinite(vz))
                cell = pb[ok] * n_th + tb[ok]
                np.add.at(counts, (cell, vzb[ok]), 1.0)
            if cap is not None and seen >= cap:
                break

    counts = counts.reshape(n_p, n_th, NB)
    p_bins = list(zip(p_edges[:-1], p_edges[1:]))
    th_bins = list(zip(th_edges[:-1], th_edges[1:]))
    cols = theta_colors(n_th)
    centers = 0.5 * (vz_edges[:-1] + vz_edges[1:])

    ncol = 4
    nrow = (n_p + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for k in range(n_p, nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    for i, (plo, phi) in enumerate(p_bins):
        ax = axs[i // ncol][i % ncol]
        for j, ((tlo, thi), c) in enumerate(zip(th_bins, cols)):
            h = counts[i, j]
            tot = h.sum()
            if tot > 0:
                ax.step(centers, h / tot, where="mid", color=c, lw=0.9)
        ax.set_title(f"{plo:g} < p < {phi:g} GeV/c", fontsize=10)
        ax.set_xlabel(r"$v_z$ [cm]")
        if i % ncol == 0:
            ax.set_ylabel("Normalized")
    handles = [Line2D([], [], color=c, lw=1.6) for c in cols]
    labels = [rf"${int(tlo)}^\circ\leq\theta<{int(thi)}^\circ$" for tlo, thi in th_bins]
    fig.legend(handles, labels, loc="upper center", ncol=min(7, n_th), fontsize=7, frameon=False)
    fig.suptitle(f"reconstructed $v_z$ (pid {args.pid}) from {args.file} — {seen:,} entries", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")

    print("\nreconstructed counts per (p, theta) cell:")
    hdr = "  ".join(f"{int(t):>4}-{int(h)}" for t, h in th_bins)
    print(f"  p \\ theta   {hdr}")
    for i, (plo, phi) in enumerate(p_bins):
        row = "  ".join(f"{int(counts[i, j].sum()):>7}" for j in range(n_th))
        print(f"  {plo:>4g}-{phi:<4g}  {row}")
    print(f"\n  total {args.pid} in grid: {int(counts.sum()):,} over {seen:,} entries")


if __name__ == "__main__":
    main()
