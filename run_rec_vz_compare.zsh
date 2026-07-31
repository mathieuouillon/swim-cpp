#!/usr/bin/env zsh
#
# Compare the reconstructed v_z of two reconstruction passes, end to end, with
# an optional third reconstructed pass (e.g. real data) and an optional "swim"
# overlay.
#
# Steps:
#   1. build   meson compile -C build rec-vz-hist (+ hipo2root, vz-swim-hist)
#   2. extract rec-vz-hist over each pass's recon/*.hipo -> one ROOT file each
#              (per-(species,p,theta) cells + the integrated vz_rec_<sp>_all);
#              passes A, B, and the optional real-data pass C.
#   3. swim    (optional, on by default) take the swim-source recon (default =
#              pass B), dump it to a particles tree with hipo2root, then re-swim
#              the tracks through torus scale SWIM_SCALE (0.998) with
#              vz-swim-hist, one file per species -> vz_swum_<cell>
#   4. plot    plot_rec_vz_compare.py overlays pass A, pass B, the optional
#              real-data pass C, and the swim, cell-by-cell + integrated
#              -> output/python/*.pdf
#
# Defaults compare torus scale 0.998 vs 1.000 for the RG-D CuSn OB sample, plus
# real data (run 018614) reconstructed with a 0.998 torus, and swim the scale-1.0
# recon with a 0.998 torus (does the swim reproduce the true 0.998 recon?).
#
# Run from anywhere (paths resolve against the repo root):
#   ./run_rec_vz_compare.zsh                 # build + extract both + swim + plot
#   ./run_rec_vz_compare.zsh --jobs 16       # worker processes per stage (default 32)
#   ./run_rec_vz_compare.zsh --plot-only     # re-plot from existing ROOT files
#   ./run_rec_vz_compare.zsh --no-build      # skip the meson compile
#   ./run_rec_vz_compare.zsh --no-extract    # reuse existing rec files; redo swim + plot
#   ./run_rec_vz_compare.zsh --no-swim       # just the reconstructed passes
#   ./run_rec_vz_compare.zsh --no-c          # drop the third (real-data) pass
#   ./run_rec_vz_compare.zsh --swim-scale 0.996   # swim torus scale
#   ./run_rec_vz_compare.zsh --vz-min -10 --vz-max 10   # v_z window (default -5..5)
#   ./run_rec_vz_compare.zsh --species ele pip    # only these species
#
# Override inputs/labels/outputs with env vars or flags:
#   A_DIR=... B_DIR=... C_DIR=... SWIM_DIR=... ./run_rec_vz_compare.zsh
#   C_DIR=... C_LABEL=... ./run_rec_vz_compare.zsh   # set the third pass
#   SWIM_ARGS="--solenoid-z-shift -3 --beam-x 0" ./run_rec_vz_compare.zsh
#     (SWIM_ARGS is forwarded verbatim to every vz-swim-hist call for the field
#      config it cannot read from the simulation's RUN::config metadata.)

emulate -L zsh
set -e  # stop at the first failing step

root=${0:A:h}                              # repo root (this script sits there)
PY=${PYTHON:-python3}
bin=$root/build/rec-vz-hist
hipo2root=$root/build/hipo2root
vzswim=$root/build/vz-swim-hist
typeset -A pid_of=(ele 11 pip 211 pim -211)

# ── Defaults (override via env or the flags below) ──────────────────────────
: ${A_DIR:=/volatile/clas12/ouillon/rgd_simu_CuSn_OB_4_torus_scale_0998/recon}
: ${B_DIR:=/volatile/clas12/ouillon/rgd_simu_CuSn_OB_4_torus_scale_1/recon}
: ${A_OUT:=$root/output/cpp/rec_vz_scale_0998.root}
: ${B_OUT:=$root/output/cpp/rec_vz_scale_1.root}
: ${A_LABEL:=scale 0.998}
: ${B_LABEL:=scale 1.0}
# Optional third pass: real data reconstructed with a 0.998 torus. Set C_DIR
# empty (or pass --no-c) to drop it and fall back to the plain A-vs-B compare.
: ${C_DIR:=/volatile/clas12/ouillon/rgd_recon_data_018614_torus_0998/recon}
: ${C_OUT:=$root/output/cpp/rec_vz_data_018614_0998.root}
: ${C_LABEL:=data 0.998}
: ${OUTDIR:=$root/output/python}
: ${SUFFIX:=}
: ${VZ_MIN:=-5}                            # v_z histogram window [cm] for EVERYTHING:
: ${VZ_MAX:=5}                             # rec-vz-hist cells + integrated AND the swim
: ${SWIM_SCALE:=0.998}                     # torus scale the swim re-swims with
: ${PARTICLES:=$root/output/cpp/particles_swim.root}  # hipo2root tree for the swim
: ${SWIM_PREFIX:=$root/output/cpp/swim_t${SWIM_SCALE}}  # per-species swim files
swim_args=(${=SWIM_ARGS})                  # extra vz-swim-hist field flags (env)

jobs=32                 # cap workers (the tools would otherwise use every core)
plot_only=0
do_build=1
do_extract=1
do_swim=1
species=(ele pip pim)

# ── Argument parsing ────────────────────────────────────────────────────────
while (( $# )); do
  case $1 in
    -j|--jobs)     jobs=$2; shift 2 ;;
    --plot-only)   plot_only=1; shift ;;
    --no-build)    do_build=0; shift ;;
    --no-extract)  do_extract=0; shift ;;
    --no-swim)     do_swim=0; shift ;;
    --vz-min)      VZ_MIN=$2; shift 2 ;;
    --vz-max)      VZ_MAX=$2; shift 2 ;;
    --swim-scale)  SWIM_SCALE=$2; SWIM_PREFIX=$root/output/cpp/swim_t$2; shift 2 ;;
    --swim-dir)    SWIM_DIR=$2; shift 2 ;;
    --a-dir)       A_DIR=$2; shift 2 ;;
    --b-dir)       B_DIR=$2; shift 2 ;;
    --a-out)       A_OUT=$2; shift 2 ;;
    --b-out)       B_OUT=$2; shift 2 ;;
    --a-label)     A_LABEL=$2; shift 2 ;;
    --b-label)     B_LABEL=$2; shift 2 ;;
    --c-dir)       C_DIR=$2; shift 2 ;;
    --c-out)       C_OUT=$2; shift 2 ;;
    --c-label)     C_LABEL=$2; shift 2 ;;
    --no-c)        C_DIR=''; shift ;;
    --outdir)      OUTDIR=$2; shift 2 ;;
    --suffix)      SUFFIX=$2; shift 2 ;;
    --species)     shift; species=(); while (( $# )) && [[ $1 != -* ]]; do species+=$1; shift; done ;;
    -h|--help)     sed -n '2,41p' $0; exit 0 ;;
    *) print -u2 "unknown argument: $1"; exit 2 ;;
  esac
done

: ${SWIM_DIR:=$B_DIR}                       # default swim source = pass B (after parsing)
: ${SWIM_LABEL:=swim t=$SWIM_SCALE}
jflag=(--jobs $jobs)
# zsh does not field-split ${SUFFIX:+--suffix $SUFFIX} (it would reach the plotter
# as one "--suffix _x" token), so build the flag as a proper two-element array.
sufflag=(); [[ -n $SUFFIX ]] && sufflag=(--suffix $SUFFIX)
swim_out() { print "${SWIM_PREFIX}_$1.root"; }   # per-species swim file path

# ── 1. build ────────────────────────────────────────────────────────────────
if (( do_build && !plot_only )); then
  print "── build ─────────────────────────────────────────────"
  targets=(rec-vz-hist)
  (( do_swim )) && targets+=(hipo2root vz-swim-hist)
  meson compile -C $root/build $targets[@]
fi

# ── 2. extract ──────────────────────────────────────────────────────────────
# Run rec-vz-hist over one pass's recon/*.hipo into OUT. Errors out if the
# directory has no .hipo files (the (N) qualifier yields an empty list rather
# than a glob error, so we can count it).
extract() {
  local dir=$1 out=$2 label=$3
  local files=( $dir/*.hipo(N) )
  if (( ! ${#files} )); then
    print -u2 "error: no .hipo files in '$dir' (pass '$label')"; exit 1
  fi
  print "── extract: $label ($#files file(s)) -> $out ──"
  $bin $jflag[@] --vz-min $VZ_MIN --vz-max $VZ_MAX --output $out $files[@]
}

if (( do_extract && !plot_only )); then
  [[ -x $bin ]] || { print -u2 "error: $bin not built (run without --no-build)"; exit 1 }
  extract $A_DIR $A_OUT $A_LABEL
  extract $B_DIR $B_OUT $B_LABEL
  [[ -n $C_DIR ]] && extract $C_DIR $C_OUT $C_LABEL
fi

# ── 3. swim ──────────────────────────────────────────────────────────────────
# Dump the swim-source recon to a particles tree once, then re-swim each species
# through torus scale SWIM_SCALE. Field config (solenoid scale, beam) comes from
# the tree's RUN::config metadata; SWIM_ARGS overrides what it lacks.
if (( do_swim && !plot_only )); then
  [[ -x $hipo2root && -x $vzswim ]] || \
    { print -u2 "error: hipo2root/vz-swim-hist not built (run without --no-build)"; exit 1 }
  sfiles=( $SWIM_DIR/*.hipo(N) )
  if (( ! ${#sfiles} )); then
    print -u2 "error: no .hipo files in swim source '$SWIM_DIR'"; exit 1
  fi
  print "── swim: hipo2root $SWIM_DIR ($#sfiles file(s)) -> $PARTICLES ──"
  $hipo2root $jflag[@] --output $PARTICLES $sfiles[@]
  for sp in $species; do
    print "── swim: vz-swim-hist --pid ${pid_of[$sp]} --torus-scale $SWIM_SCALE -> $(swim_out $sp) ──"
    $vzswim $jflag[@] --pid ${pid_of[$sp]} --torus-scale $SWIM_SCALE \
        --vz-min $VZ_MIN --vz-max $VZ_MAX \
        $swim_args[@] --output $(swim_out $sp) $PARTICLES
  done
fi

# ── 4. plot ─────────────────────────────────────────────────────────────────
for f in $A_OUT $B_OUT; do
  [[ -f $f ]] || { print -u2 "error: missing '$f' (run without --plot-only first)"; exit 1 }
done
# Attach the optional third (real-data) pass via --rec when its ROOT is present.
recflags=()
if [[ -n $C_DIR ]]; then
  if [[ -f $C_OUT ]]; then
    recflags+=(--rec $C_OUT $C_LABEL)
  else
    print -u2 "warn: pass C enabled but '$C_OUT' missing; skipping it (re-run without --plot-only/--no-extract)"
  fi
fi
# Attach a --swim-<sp> flag for each species whose swim file exists.
swimflags=()
if (( do_swim )); then
  for sp in $species; do
    [[ -f $(swim_out $sp) ]] && swimflags+=(--swim-$sp $(swim_out $sp))
  done
  (( ${#swimflags} )) && swimflags+=(--swim-label $SWIM_LABEL)
fi
note=""
(( ${#recflags} )) && note+=" + $C_LABEL"
(( ${#swimflags} )) && note+=" + swim"
print "── plot: $A_OUT vs $B_OUT$note -> $OUTDIR ──"
$PY $root/plot_rec_vz_compare.py $A_OUT $B_OUT \
    --labels $A_LABEL $B_LABEL \
    $recflags[@] \
    --species $species[@] \
    $swimflags[@] \
    --outdir $OUTDIR $sufflag[@]

print "done."
