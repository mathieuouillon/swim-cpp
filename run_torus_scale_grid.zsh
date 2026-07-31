#!/usr/bin/env zsh
#
# Stacked momentum×theta grid of reconstructed v_z for a TORUS-SCALE SCAN of one
# RG-D run: one block per torus scale, read straight from each scale's recon
# .hipo files. Built for the fine run-018614 scan (torus 0.9970..1.0010, step
# 0.0002) -> 21 blocks, smallest scale on top (--reverse for largest on top).
#
# The scan directories are <BASE>/rgd_recon_data_<PASS>_<RUN>_torus_<NNNNN>/recon
# (PASS is the re-cook pass that prefixes the dir name; set PASS= empty for the
# original un-prefixed rgd_recon_data_<RUN>_torus_<NNNNN> layout), where the
# 5-digit NNNNN is the torus scale x10000 (10000 = 1.0000, 09980 = 0.9980).
# Their raw recon/*.hipo have no trailer index (stock hipo-utils shows 0 events);
# rec-vz-hist reads them anyway via its sequential-scan fallback (see hipo-check).
#
# Steps:
#   1. build   meson compile -C build rec-vz-hist
#   2. extract rec-vz-hist over each scale's recon/job_*/rec_clas_*.hipo (the
#              reconstructed files only) -> one ROOT file each
#              (skips scales whose ROOT already exists unless --force)
#   3. plot    plot_rec_vz_grid.py: one momentum×theta block per scale, stacked
#              in ascending scale order -> output/python/rec_vz_<sp>_grid<suffix>.pdf
#
# v_z window defaults to [-13, 0] cm (the reconstructed-vz reference range).
# NOTE: 21 stacked blocks make a very tall (~13 ft) figure -- the theta legend is
# drawn once, on the top block.
#
# Run from anywhere (paths resolve against the repo root):
#   ./run_torus_scale_grid.zsh                 # build + extract all scales + plot
#   ./run_torus_scale_grid.zsh --jobs 16       # worker processes per stage (default 32)
#   ./run_torus_scale_grid.zsh --plot-only     # re-plot from existing ROOT files
#   ./run_torus_scale_grid.zsh --no-build      # skip the meson compile
#   ./run_torus_scale_grid.zsh --no-extract    # reuse existing ROOT files; just re-plot
#   ./run_torus_scale_grid.zsh --force         # re-extract even if the ROOT exists
#   ./run_torus_scale_grid.zsh --reverse       # largest scale on top
#   ./run_torus_scale_grid.zsh --run 018614 --base /volatile/clas12/ouillon
#   ./run_torus_scale_grid.zsh --pass 3      # data dir rgd_recon_data_3_<RUN>_torus_*
#   ./run_torus_scale_grid.zsh --pass ''     # original un-prefixed rgd_recon_data_<RUN>_torus_*
#   ./run_torus_scale_grid.zsh --vz-min -10 --vz-max 5   # v_z window (default -13..0)
#   ./run_torus_scale_grid.zsh --species ele pip    # only these species
#
# Override the scan set explicitly (else it globs <BASE>/rgd_recon_data_<RUN>_torus_*):
#   DIRS=(/path/a/recon /path/b/recon) ./run_torus_scale_grid.zsh

emulate -L zsh
set -e  # stop at the first failing step

root=${0:A:h}                              # repo root (this script sits there)
PY=${PYTHON:-python3}
bin=$root/build/rec-vz-hist

# ── Defaults (override via env or the flags below) ──────────────────────────
: ${BASE:=/volatile/clas12/ouillon}        # parent of the rgd_recon_data_* dirs
: ${RUN:=018614}                           # run number in the directory names
: ${PASS:=3}                               # re-cook pass prefixing the dir name
                                           # (rgd_recon_data_<PASS>_<RUN>_torus_*);
                                           # set empty for the original un-prefixed scan
: ${OUTDIR:=$root/output/python}           # where the grid PDFs go
: ${ROOTDIR:=$root/output/cpp}             # where the per-scale rec-vz ROOT files go
: ${VZ_MIN:=-13}                           # v_z histogram window [cm] (cells + integrated)
: ${VZ_MAX:=0}

jobs=32                 # cap workers (rec-vz-hist would otherwise use every core)
plot_only=0
do_build=1
do_extract=1
force=0
reverse=0
species=(ele pip pim)
typeset -a DIRS         # explicit scan set; else globbed below (may be preset in env)

# ── Argument parsing ────────────────────────────────────────────────────────
while (( $# )); do
  case $1 in
    -j|--jobs)     jobs=$2; shift 2 ;;
    --plot-only)   plot_only=1; shift ;;
    --no-build)    do_build=0; shift ;;
    --no-extract)  do_extract=0; shift ;;
    --force)       force=1; shift ;;
    --reverse)     reverse=1; shift ;;
    --run)         RUN=$2; shift 2 ;;
    --pass)        PASS=$2; shift 2 ;;
    --base)        BASE=$2; shift 2 ;;
    --vz-min)      VZ_MIN=$2; shift 2 ;;
    --vz-max)      VZ_MAX=$2; shift 2 ;;
    --outdir)      OUTDIR=$2; shift 2 ;;
    --suffix)      SUFFIX=$2; shift 2 ;;
    --species)     shift; species=(); while (( $# )) && [[ $1 != -* ]]; do species+=$1; shift; done ;;
    -h|--help)     sed -n '2,42p' $0; exit 0 ;;
    *) print -u2 "unknown argument: $1"; exit 2 ;;
  esac
done

jflag=(--jobs $jobs)

# Derived from RUN/PASS *after* parsing so --run/--pass on the command line apply.
# DATABASE names the data dirs; NAMETAG also keys the ROOT/PDF outputs so a new pass
# does not reuse (or clobber) the old scan's outputs.
DATABASE=rgd_recon_data${PASS:+_$PASS}_${RUN}
NAMETAG=${PASS:+${PASS}_}${RUN}
: ${SUFFIX:=_torus_scan_${NAMETAG}}        # appended to the grid PDF names

# zsh keeps a spaces-containing scalar as one array element (no word splitting),
# so SUFFIX/labels survive as single tokens.
sufflag=(); [[ -n $SUFFIX ]] && sufflag=(--suffix $SUFFIX)

# ── discover the scan directories (ascending torus scale) ───────────────────
# Each is <BASE>/<DATABASE>_torus_<NNNNN>/recon; NNNNN is fixed-width so a lexical
# sort of the paths is also the numeric scale order.
# Exactly five digits: matches the scan dirs (torus x10000, e.g. 09980, 10000)
# and skips any older 4-digit dir like ..._torus_0998 that would collide.
if (( ! ${#DIRS} )); then
  DIRS=( $BASE/${DATABASE}_torus_[0-9][0-9][0-9][0-9][0-9]/recon(N) )
fi
DIRS=( ${(o)DIRS} )                        # ascending; reversed below if --reverse
(( reverse )) && DIRS=( ${(Oa)DIRS} )
if (( ! ${#DIRS} )); then
  print -u2 "error: no scan dirs match '$BASE/${DATABASE}_torus_*/recon'"
  exit 1
fi

# scale tag (5-digit NNNNN) and pretty scale string (1.0000) from a recon path.
scale_tag() { local d=${1:h}; print -- ${d##*_torus_} }   # .../torus_NNNNN/recon -> NNNNN
scale_str() { local t=$1; print -- "${t[1]}.${t[2,5]}" }  # 10000 -> 1.0000

# ── 1. build ────────────────────────────────────────────────────────────────
if (( do_build && !plot_only )); then
  print "── build ─────────────────────────────────────────────"
  meson compile -C $root/build rec-vz-hist
fi

# ── 2. extract + collect (per scale) ────────────────────────────────────────
# rec-vz-hist over one scale's reconstructed hipo -> OUT. The recon tree nests the
# files one level down (recon/job_*/rec_clas_*.hipo) and each job dir also holds the
# raw clas_*.hipo inputs, so glob recursively for rec_clas_* ONLY. These files have
# no trailer index; rec-vz-hist recovers them by sequential scan.
extract() {  # $1 = recon dir, $2 = out ROOT, $3 = label
  local files=( $1/**/rec_clas_*.hipo(N) )
  if (( ! ${#files} )); then
    print -u2 "warn: no .hipo in '$1' ($3); skipping"; return 1
  fi
  print "── extract: $3 ($#files file(s)) -> $2 ──"
  $bin $jflag[@] --vz-min $VZ_MIN --vz-max $VZ_MAX --output $2 $files[@]
}

if (( do_extract && !plot_only )) && [[ ! -x $bin ]]; then
  print -u2 "error: $bin not built (run without --no-build)"; exit 1
fi
mkdir -p $ROOTDIR

gridfiles=(); gridlabels=()
for d in $DIRS; do
  tag=$(scale_tag $d)
  out=$ROOTDIR/rec_vz_scan_${NAMETAG}_t${tag}.root
  label="scale $(scale_str $tag)"
  if (( do_extract && !plot_only )); then
    if (( force )) || [[ ! -f $out ]]; then
      extract $d $out $label || continue
    else
      print "── skip (ROOT exists): $label -> $out ──"
    fi
  fi
  if [[ -f $out ]]; then
    gridfiles+=($out); gridlabels+=($label)
  else
    print -u2 "warn: '$out' missing for $label; not plotted"
  fi
done

# ── 3. plot ─────────────────────────────────────────────────────────────────
(( ${#gridfiles} )) || { print -u2 "error: no ROOT files to plot"; exit 1 }
mkdir -p $OUTDIR
print "── plot: ${#gridfiles} scale block(s) (top ${gridlabels[1]}, bottom ${gridlabels[-1]}) -> $OUTDIR ──"
$PY $root/plot_rec_vz_grid.py $gridfiles[@] \
    --labels $gridlabels[@] \
    --species $species[@] \
    --outdir $OUTDIR $sufflag[@]

print "done."
