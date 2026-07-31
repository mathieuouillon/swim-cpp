#!/usr/bin/env zsh
#
# Compare the reconstructed v_z of RG-D run 018614 reconstructed up to three ways:
#   A = our reconstruction with torus scale 0.998   (a recon/ directory)
#   B = the official pass1 production (torus scale 1.0), a single SIDIS train
#       file, capped at MAX_EVENTS events.
#   C = our reconstruction with torus scale 0.9982  (a recon/ directory; optional,
#       a bottom block in the grid -- pass --no-c or set C_IN='' to drop it).
#
# Same physics run, several torus scales -> overlay the EB-reconstructed v_z
# cell-by-cell. Each (species, p, theta) cell is area-normalized in the plotter,
# so this is a v_z *shape* comparison (the samples have very different N).
#
# Steps:
#   1. build   meson compile -C build rec-vz-hist
#   2. extract rec-vz-hist over each input -> one ROOT file each
#              (per-(species,p,theta) cells + the integrated vz_rec_<sp>_all);
#              B is capped at MAX_EVENTS via rec-vz-hist --max-events.
#   3. plot    plot_rec_vz_grid.py: a momentum×theta grid per pass, stacked with
#              the scale-1.0 block (B) on top, the scale-0.998 block (A) next, and
#              the optional scale-0.9982 block (C) below
#              -> output/python/rec_vz_<sp>_grid<suffix>.pdf
#
# v_z window defaults to [-13, 0] cm (the reconstructed-vz reference range).
#
# Run from anywhere (paths resolve against the repo root):
#   ./run_data_vz_compare.zsh                 # build + extract all + plot
#   ./run_data_vz_compare.zsh --jobs 16       # worker processes per stage (default 32)
#   ./run_data_vz_compare.zsh --plot-only     # re-plot from existing ROOT files
#   ./run_data_vz_compare.zsh --no-build      # skip the meson compile
#   ./run_data_vz_compare.zsh --no-extract    # reuse existing rec files; just re-plot
#   ./run_data_vz_compare.zsh --no-c          # drop the third (0.9982) pass
#   ./run_data_vz_compare.zsh --max-events 5000000   # cap on B (default 10M; 0 = all)
#   ./run_data_vz_compare.zsh --vz-min -10 --vz-max 5   # v_z window (default -13..0)
#   ./run_data_vz_compare.zsh --species ele pip    # only these species
#
# Override inputs/labels/outputs with env vars or flags:
#   A_IN=... B_IN=... C_IN=... A_LABEL=... B_LABEL=... C_LABEL=... ./run_data_vz_compare.zsh
# A_IN/B_IN/C_IN may be a directory (its *.hipo are globbed) or a single .hipo file.

emulate -L zsh
set -e  # stop at the first failing step

root=${0:A:h}                              # repo root (this script sits there)
PY=${PYTHON:-python3}
bin=$root/build/rec-vz-hist

# ── Defaults (override via env or the flags below) ──────────────────────────
: ${A_IN:=/volatile/clas12/ouillon/rgd_recon_data_018614_torus_0998/recon}
: ${B_IN:=/cache/clas12/rg-d/production/pass1/recon/CuSn/dst/train/SIDIS/SIDIS_018614.hipo}
: ${A_OUT:=$root/output/cpp/rec_vz_data018614_t0998_vz13.root}
: ${B_OUT:=$root/output/cpp/rec_vz_data018614_t1000_vz13.root}
: ${A_LABEL:=data 0.998}
: ${B_LABEL:=data 1.0}
# Optional third pass: RG-D run 018614 reconstructed with a 0.9982 torus -- a new
# block at the bottom of the grid. Set C_IN empty (or pass --no-c) to drop it and
# fall back to the plain B-vs-A two-block grid. The raw recon/*.hipo here have no
# trailer index (stock hipo-utils shows 0 events); rec-vz-hist reads them anyway.
: ${C_IN:=/volatile/clas12/ouillon/rgd_recon_data_018614_torus_09982/recon}
: ${C_OUT:=$root/output/cpp/rec_vz_data018614_t09982_vz13.root}
: ${C_LABEL:=data 0.9982}
: ${OUTDIR:=$root/output/python}
: ${SUFFIX:=_data_018614}
: ${VZ_MIN:=-13}                           # v_z histogram window [cm] (cells + integrated)
: ${VZ_MAX:=0}
: ${MAX_EVENTS:=10000000}                  # cap applied to B only (the single train file)

jobs=32                 # cap workers (the tool would otherwise use every core)
plot_only=0
do_build=1
do_extract=1
species=(ele pip pim)

# ── Argument parsing ────────────────────────────────────────────────────────
while (( $# )); do
  case $1 in
    -j|--jobs)     jobs=$2; shift 2 ;;
    --plot-only)   plot_only=1; shift ;;
    --no-build)    do_build=0; shift ;;
    --no-extract)  do_extract=0; shift ;;
    --max-events)  MAX_EVENTS=$2; shift 2 ;;
    --vz-min)      VZ_MIN=$2; shift 2 ;;
    --vz-max)      VZ_MAX=$2; shift 2 ;;
    --a-in)        A_IN=$2; shift 2 ;;
    --b-in)        B_IN=$2; shift 2 ;;
    --a-out)       A_OUT=$2; shift 2 ;;
    --b-out)       B_OUT=$2; shift 2 ;;
    --a-label)     A_LABEL=$2; shift 2 ;;
    --b-label)     B_LABEL=$2; shift 2 ;;
    --c-in)        C_IN=$2; shift 2 ;;
    --c-out)       C_OUT=$2; shift 2 ;;
    --c-label)     C_LABEL=$2; shift 2 ;;
    --no-c)        C_IN=''; shift ;;
    --outdir)      OUTDIR=$2; shift 2 ;;
    --suffix)      SUFFIX=$2; shift 2 ;;
    --species)     shift; species=(); while (( $# )) && [[ $1 != -* ]]; do species+=$1; shift; done ;;
    -h|--help)     sed -n '2,39p' $0; exit 0 ;;
    *) print -u2 "unknown argument: $1"; exit 2 ;;
  esac
done

jflag=(--jobs $jobs)
# zsh does not field-split ${SUFFIX:+--suffix $SUFFIX} (it would reach the plotter
# as one "--suffix _x" token), so build the flag as a proper two-element array.
sufflag=(); [[ -n $SUFFIX ]] && sufflag=(--suffix $SUFFIX)

# ── 1. build ────────────────────────────────────────────────────────────────
if (( do_build && !plot_only )); then
  print "── build ─────────────────────────────────────────────"
  meson compile -C $root/build rec-vz-hist
fi

# ── 2. extract ──────────────────────────────────────────────────────────────
# Run rec-vz-hist over one pass's input into OUT. The input may be a directory
# (its *.hipo are globbed) or a single .hipo file. maxev>0 caps the event count
# (only meaningful for a single big file, where the tool shards by record).
extract() {
  local in=$1 out=$2 label=$3 maxev=$4
  local files
  if [[ -d $in ]]; then
    files=( $in/*.hipo(N) )
  elif [[ -f $in ]]; then
    files=( $in )
  else
    files=()
  fi
  if (( ! ${#files} )); then
    print -u2 "error: no .hipo input at '$in' (pass '$label')"; exit 1
  fi
  local capflag=()
  (( maxev > 0 )) && capflag=(--max-events $maxev)
  print "── extract: $label ($#files file(s)${maxev:+, <=$maxev ev}) -> $out ──"
  $bin $jflag[@] --vz-min $VZ_MIN --vz-max $VZ_MAX $capflag[@] --output $out $files[@]
}

if (( do_extract && !plot_only )); then
  [[ -x $bin ]] || { print -u2 "error: $bin not built (run without --no-build)"; exit 1 }
  extract $A_IN $A_OUT $A_LABEL 0
  extract $B_IN $B_OUT $B_LABEL $MAX_EVENTS
  [[ -n $C_IN ]] && extract $C_IN $C_OUT $C_LABEL 0
fi

# ── 3. plot ─────────────────────────────────────────────────────────────────
for f in $A_OUT $B_OUT; do
  [[ -f $f ]] || { print -u2 "error: missing '$f' (run without --plot-only first)"; exit 1 }
done
# Stacked momentum×theta grid: top = B (scale 1.0), then A (scale 0.998), and the
# optional third pass C (scale 0.9982) as a bottom block when its ROOT exists.
gridfiles=($B_OUT $A_OUT)
gridlabels=($B_LABEL $A_LABEL)
note=""
if [[ -n $C_IN ]]; then
  if [[ -f $C_OUT ]]; then
    gridfiles+=($C_OUT); gridlabels+=($C_LABEL); note=" + $C_LABEL"
  else
    print -u2 "warn: pass C enabled but '$C_OUT' missing; skipping it (re-run without --plot-only/--no-extract)"
  fi
fi
print "── plot: grid (top $B_LABEL, then $A_LABEL$note) -> $OUTDIR ──"
$PY $root/plot_rec_vz_grid.py $gridfiles[@] \
    --labels $gridlabels[@] \
    --species $species[@] \
    --outdir $OUTDIR $sufflag[@]

print "done."
