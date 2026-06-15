#!/usr/bin/env zsh
#
# Drive scan_field.py over every field-alignment parameter, 10 M entries each.
#
# Each row in `scans` below is an independent 1-D sweep: scan_field.py runs
# vz-swim-hist once per value (--max-parallel at a time, each loading its own
# ~1 GB torus map), then fits the swum-vz peaks and writes a summary. Outputs
# land in output/python/<param>_scan/.
#
# Run from the repo root, after hipo2root has written output/cpp/particles.root:
#   ./run_field_scans.zsh [INPUT.root] [extra scan_field.py args ...]
#
# Examples:
#   ./run_field_scans.zsh
#   ./run_field_scans.zsh output/cpp/particles.root --beam-y -0.18 --max-parallel 20
#   ./run_field_scans.zsh --plot-only        # re-fit/plot from existing vz_bins.root (no swimming)
#
# Notes:
#   - The input file is the first NON-flag argument (default
#     output/cpp/particles.root); everything else is forwarded to EVERY
#     scan_field.py call, e.g. --beam-y -0.18 (run 18614 beam offset),
#     --max-parallel N, --threads-per-job N, --force (re-run existing values).
#   - --plot-only re-runs only the fitting/plotting from each scan's existing
#     vz_bins.root (no input needed, no swimming) — seconds, not hours. Use it
#     to re-make every summary after changing the Python plotting/fit code.
#   - The scan ranges below must match the runs you want to (re)plot. Comment
#     out a row to skip that parameter.

emulate -L zsh

# Input = first non-flag argument (its default is fine for --plot-only, which
# needs no input); every other argument is forwarded verbatim to scan_field.py.
if [[ ${1:-} == -* ]]; then
  INPUT=output/cpp/particles.root
else
  INPUT=${1:-output/cpp/particles.root}
  shift 2>/dev/null || true
fi
extra=("$@")

plot_only=0
(( ${extra[(I)--plot-only]} )) && plot_only=1   # --plot-only present among the forwarded args?

PY=${PYTHON:-python3}
script=${0:A:h}/scan_field.py         # scan_field.py sits next to this driver
entries=10000000                      # 10 M tree entries per run

# param         min     max     step          (cm for shifts; dimensionless for scale)
scans=(
  "solenoid-x   -10     10      2"
  "solenoid-y   -10     10      2"
  "solenoid-z   -10     10      2"
  "torus-x      -0.5    0.5     0.05"
  "torus-y      -0.5    0.5     0.05"
  "torus-z      -2      2       0.2"
  "torus-scale  -1.005  -0.995  0.001"   # fine scan around nominal -1.0 (11 runs); widen if needed
)

if (( plot_only )); then
  print -P "%B plot-only: re-fitting/plotting from existing vz_bins.root (no swimming)   forwarding: ${extra} %b"
else
  print -P "%B input: ${INPUT}   forwarding: ${extra:-<none>} %b"
fi

typeset -a failed
for row in $scans; do
  f=(${=row})                         # split the row on whitespace
  param=$f[1] min=$f[2] max=$f[3] step=$f[4]
  (( plot_only )) && mode="plot-only" || mode="${entries} entries"
  print -P "%B%F{cyan}== ${param}: ${min} .. ${max} step ${step}  (${mode}) ==%f%b"
  if $PY $script $INPUT \
       --scan-param $param --z-min $min --z-max $max --z-step $step \
       --max-entries $entries $extra; then
    print -P "%F{green}== ${param}: done ==%f"
  else
    rc=$?
    print -P "%F{red}== ${param}: FAILED (rc=${rc}) ==%f"
    failed+=$param
  fi
done

if (( ${#failed} )); then
  print -P "%B%F{red}field scans with failures: ${failed}%f%b"
  exit 1
fi
print -P "%B%F{green}all field scans complete%f%b"
