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
#
# Notes:
#   - Everything after INPUT is forwarded to EVERY scan_field.py call, e.g.
#     --beam-y -0.18 (run 18614 beam offset), --max-parallel N, --threads-per-job N,
#     --force (re-run existing values), --plot-only (rebuild summaries only).
#   - Comment out a row in `scans` to skip that parameter.
#   - WARNING: the torus-scale row is 2001 runs (-0.995..1.005 step 0.001) — ~100x
#     the others. Coarsen its step (e.g. 0.01) or comment it out unless you need it.

emulate -L zsh

INPUT=${1:-output/cpp/particles.root}
shift 2>/dev/null || true
extra=("$@")                          # forwarded to every scan_field.py call

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
  "torus-scale  -0.995  1.005   0.001"
)

print -P "%B input: ${INPUT}   forwarding: ${extra:-<none>} %b"

typeset -a failed
for row in $scans; do
  f=(${=row})                         # split the row on whitespace
  param=$f[1] min=$f[2] max=$f[3] step=$f[4]
  print -P "%B%F{cyan}== ${param}: ${min} .. ${max} step ${step}  (${entries} entries) ==%f%b"
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
