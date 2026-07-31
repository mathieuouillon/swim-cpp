#!/usr/bin/env zsh
#
# Quick stacked momentum×theta grid of reconstructed v_z for an ARBITRARY set of
# inputs (one block per input) -- no scale-scan or A-vs-B wiring. Each input is a
# .hipo file or a directory (its *.hipo, or its recon/*.hipo, are globbed). Built
# to eyeball a single (re-cooked) file:
#
#   ./run_rec_grid.zsh /volatile/clas12/ouillon/rgd_018614_test/rec_clas_018614.evio.00000-00004.hipo
#
# or to stack a few (inputs FIRST, then --labels):
#   ./run_rec_grid.zsh data.hipo sim.hipo --labels "data yaml" "sim yaml"
#
# Steps: build rec-vz-hist -> extract each input to output/cpp/rec_vz_<name>.root
# -> plot_rec_vz_grid.py. v_z window defaults to [-13, 0] cm.
#
# Inputs come first; flags after:
#   --labels L...   one block title per input (default: each input's name)
#   -j|--jobs N     rec-vz-hist worker processes (default 32)
#   --vz-min/--vz-max   v_z window [cm] (default -13 0)
#   --species ...   species to plot (default ele pip pim)
#   --suffix S      appended to the grid PDF names (default: the lone input's name)
#   --outdir DIR    directory for the PDFs (default output/python)
#   --no-build      skip the meson compile
#   --plot-only     reuse existing ROOT files; just re-plot
#   --force         re-extract even if the ROOT exists

emulate -L zsh
set -e  # stop at the first failing step

root=${0:A:h}                              # repo root (this script sits there)
PY=${PYTHON:-python3}
bin=$root/build/rec-vz-hist

: ${OUTDIR:=$root/output/python}           # where the grid PDFs go
: ${ROOTDIR:=$root/output/cpp}             # where the per-input rec-vz ROOT files go
: ${VZ_MIN:=-13}                           # v_z histogram window [cm] (cells + integrated)
: ${VZ_MAX:=0}
: ${SUFFIX:=}

jobs=32; do_build=1; plot_only=0; force=0
species=(ele pip pim)
labels=(); inputs=()

# ── Argument parsing (inputs first, then flags) ─────────────────────────────
while (( $# )); do
  case $1 in
    --labels)    shift; labels=(); while (( $# )) && [[ $1 != -* ]]; do labels+=($1); shift; done ;;
    --species)   shift; species=(); while (( $# )) && [[ $1 != -* ]]; do species+=($1); shift; done ;;
    -j|--jobs)   jobs=$2; shift 2 ;;
    --vz-min)    VZ_MIN=$2; shift 2 ;;
    --vz-max)    VZ_MAX=$2; shift 2 ;;
    --suffix)    SUFFIX=$2; shift 2 ;;
    --outdir)    OUTDIR=$2; shift 2 ;;
    --no-build)  do_build=0; shift ;;
    --plot-only) plot_only=1; shift ;;
    --force)     force=1; shift ;;
    -h|--help)   sed -n '2,25p' $0; exit 0 ;;
    -*) print -u2 "unknown argument: $1"; exit 2 ;;
    *)  inputs+=($1); shift ;;
  esac
done

(( ${#inputs} )) || { print -u2 "error: no input (give a .hipo file or a recon dir)"; exit 1 }

jflag=(--jobs $jobs)

# name (ROOT stem + default block label) of an input: a directory's own basename,
# else the file's PARENT-dir basename -- which names the cook (e.g. rgd_018614_test).
name_of() { [[ -d $1 ]] && print -- ${1:t} || print -- ${1:h:t} }
sani()    { print -- ${1//[^A-Za-z0-9._-]/_} }

# Default the PDF suffix to a lone input's name, so a single-file run does not
# clobber the generic rec_vz_<sp>_grid.pdf from other runs.
if [[ -z $SUFFIX && ${#inputs} -eq 1 ]]; then SUFFIX="_$(sani $(name_of $inputs[1]))"; fi
sufflag=(); [[ -n $SUFFIX ]] && sufflag=(--suffix $SUFFIX)

# ── 1. build ────────────────────────────────────────────────────────────────
if (( do_build && !plot_only )); then
  print "── build ─────────────────────────────────────────────"
  meson compile -C $root/build rec-vz-hist
fi
if (( !plot_only )) && [[ ! -x $bin ]]; then
  print -u2 "error: $bin not built (run without --no-build)"; exit 1
fi
mkdir -p $ROOTDIR

# ── 2. extract each input -> one ROOT, collect for the grid ─────────────────
gridfiles=(); gridlabels=(); idx=0
for in in $inputs; do
  (( ++idx ))
  nm=$(name_of $in)
  out=$ROOTDIR/rec_vz_$(sani $nm).root
  label=${labels[$idx]:-$nm}
  if (( !plot_only )); then
    if (( force )) || [[ ! -f $out ]]; then
      if [[ -d $in ]]; then
        # Re-cooked datasets keep raw (clas_*.hipo) and reconstructed (rec_*.hipo)
        # side by side, sometimes nested under recon/job_*/ -- take only the rec_
        # files, recursively. Fall back to the flat layouts for older cooks.
        files=( $in/**/rec_*.hipo(N) )
        (( ${#files} )) || files=( $in/*.hipo(N) )
        (( ${#files} )) || files=( $in/recon/*.hipo(N) )
      elif [[ -f $in ]]; then
        files=( $in )
      else
        files=()
      fi
      if (( ! ${#files} )); then print -u2 "warn: no .hipo at '$in'; skipping"; continue; fi
      print "── extract: $label ($#files file(s)) -> $out ──"
      $bin $jflag[@] --vz-min $VZ_MIN --vz-max $VZ_MAX --output $out $files[@]
    else
      print "── skip (ROOT exists): $label -> $out ──"
    fi
  fi
  if [[ -f $out ]]; then gridfiles+=($out); gridlabels+=($label)
  else print -u2 "warn: '$out' missing for $label; not plotted"; fi
done

# ── 3. plot ─────────────────────────────────────────────────────────────────
(( ${#gridfiles} )) || { print -u2 "error: no ROOT files to plot"; exit 1 }
mkdir -p $OUTDIR
print "── plot: ${#gridfiles} block(s) -> $OUTDIR ──"
$PY $root/plot_rec_vz_grid.py $gridfiles[@] \
    --labels $gridlabels[@] \
    --species $species[@] \
    --outdir $OUTDIR $sufflag[@]
print "done."
