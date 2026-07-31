#!/usr/bin/env python3
"""Fit the cot(theta) vz-correction coefficients c0, c1 to a run.

Model: the peak2 mean walks linearly in cot(theta) per momentum bin,
  mean(p, theta) = base(p) + a(p) * cot(theta),
and a(p) = a0 + a1 / p. The correction that flattens the walk is
  vz += (c0 + c1/p) * cot(theta)   with  c0 = -a0,  c1 = -a1.
Prints c0, c1 to feed straight into vz-swim-hist --vz-cot-coeff/-p-coeff.

Input can be either a scan_field.py `<param>_peak2_fit.csv` OR an uncorrected
vz_bins.root directly (the peak2 fit is then done here, so no scan is needed).

Usage:
  python fit_vz_cot.py <param>_peak2_fit.csv [--scale -1.00]
  python fit_vz_cot.py uncorrected/vz_bins.root [--pid 11]
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def per_p_from_csv(path, scale):
    """{p_mid: [(cot_theta, peak2_mean), ...]} from a scan peak2_fit CSV."""
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        param = r.fieldnames[0]
        rows = [row for row in r if scale is None or row[param] == scale]
    if not rows:
        raise SystemExit("no rows (check --scale)")
    per_p = defaultdict(list)
    for row in rows:
        p_mid = 0.5 * (float(row["p_lo"]) + float(row["p_hi"]))
        t_mid = 0.5 * (float(row["theta_lo"]) + float(row["theta_hi"]))
        per_p[p_mid].append((1.0 / np.tan(np.radians(t_mid)), float(row["peak2_mean_cm"])))
    return per_p


def per_p_from_root(path, pid):
    """{p_mid: [(cot_theta, peak2_mean), ...]} by fitting peak2 in every cell of
    an uncorrected vz_bins.root (same fitter/grid as scan_field.py)."""
    import scan_field
    scan_field.P_EDGES = scan_field.momentum_edges(pid)
    scan_field.P_BINS[:] = list(zip(scan_field.P_EDGES[:-1], scan_field.P_EDGES[1:]))
    scan_field.TH_EDGES = scan_field.theta_edges(pid)
    scan_field.TH_BINS[:] = list(zip(scan_field.TH_EDGES[:-1], scan_field.TH_EDGES[1:]))
    from plot_correction_presentation import load_cells
    _, mean, _ = load_cells(path)
    per_p = defaultdict(list)
    for ip, (p_lo, p_hi) in enumerate(scan_field.P_BINS):
        p_mid = 0.5 * (p_lo + p_hi)
        for it, (t_lo, t_hi) in enumerate(scan_field.TH_BINS):
            if np.isfinite(mean[ip, it]):
                cot = 1.0 / np.tan(np.radians(0.5 * (t_lo + t_hi)))
                per_p[p_mid].append((cot, mean[ip, it]))
    return per_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="<param>_peak2_fit.csv OR an uncorrected vz_bins.root")
    ap.add_argument("--scale", default=None,
                    help="CSV only: restrict to this scanned-parameter value (e.g. -1.00)")
    ap.add_argument("--pid", type=int, default=11, help="ROOT only: species pid (default 11)")
    args = ap.parse_args()

    if args.input.endswith(".root"):
        per_p = per_p_from_root(args.input, args.pid)
    else:
        per_p = per_p_from_csv(args.input, args.scale)

    # a(p) = slope of mean vs cot(theta).
    p_arr, a_arr = [], []
    print(f"{'p_mid':>6} {'a(p) [cm]':>10} {'n_theta':>8}")
    for p_mid in sorted(per_p):
        pts = np.array(per_p[p_mid])
        if len(pts) < 3:
            continue
        a, base = np.polyfit(pts[:, 0], pts[:, 1], 1)
        p_arr.append(p_mid)
        a_arr.append(a)
        print(f"{p_mid:>6.1f} {a:>10.3f} {len(pts):>8d}")
    p_arr, a_arr = np.array(p_arr), np.array(a_arr)

    # a(p) = a0 + a1/p  ->  fit vs 1/p.
    a1, a0 = np.polyfit(1.0 / p_arr, a_arr, 1)
    c0, c1 = -a0, -a1
    print(f"\na(p) = {a0:+.4f} {a1:+.4f}/p   (a0=p-flat geometric, a1/p=magnetic)")
    print(f"\ncorrection to flatten:  vz += (c0 + c1/p)*cot(theta)")
    print(f"  c0 = {c0:+.4f} cm   c1 = {c1:+.4f} cm.GeV")
    print(f"\n  --vz-cot-coeff {c0:.4f} --vz-cot-p-coeff {c1:.4f}")


if __name__ == "__main__":
    main()
