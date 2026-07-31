#!/usr/bin/env zsh
#
# Run `hipo-utils -test` on every .hipo file in a directory (or on the files /
# directories given) and classify each one:
#
#   OK       readable with a valid index
#   NOINDEX  "file does not appear to have an index" -> hipo-utils reports 0
#            records and "status check : ERROR", BUT the hipo4 library that
#            rec-vz-hist / vz-swim-hist use reads such files fine (it builds the
#            index on open). Re-index with `hipo-utils -doctor` (or --fix) only
#            if a tool that needs the legacy index will read them.
#   ERROR    a genuine failure (unreadable, 0 records without the no-index note,
#            or a non-zero exit) -- these need attention.
#
# Each check spawns a JVM, so files are tested in parallel (--jobs, default 16).
# A live counter shows progress; ERROR files are printed as found; a summary
# with the per-category breakdown is printed at the end.
#
# Usage (run from anywhere):
#   ./check_hipo_files.zsh                        # default DIR (the 0.998 recon dir)
#   ./check_hipo_files.zsh DIR                    # all *.hipo in DIR
#   ./check_hipo_files.zsh a.hipo b.hipo DIR ...  # mix of files and dirs
#   ./check_hipo_files.zsh --jobs 32 DIR
#   ./check_hipo_files.zsh --bad-list bad.txt DIR # write NOINDEX+ERROR paths to a file
#   ./check_hipo_files.zsh --max-list 0 DIR       # list ALL flagged files (0 = no cap)
#   ./check_hipo_files.zsh --fix DIR              # then re-index NOINDEX/ERROR files
#                                                 #   in place with `hipo-utils -doctor`
#
# Exit status is non-zero if any file is NOINDEX or ERROR. --fix REWRITES files.

emulate -L zsh

# ── hidden single-file workers (re-exec'd by xargs) ─────────────────────────
# --check-one prints "OK<TAB>path", "NOINDEX<TAB>path", or "ERROR<TAB>path<TAB>reason".
if [[ $1 == --check-one ]]; then
  file=$2
  out=$(hipo-utils -test "$file" 2>&1); rc=$?
  if [[ $out == *"does not appear to have an index"* ]]; then
    print -r -- "NOINDEX	$file"
  elif [[ $out == *"status check : ERROR"* ]]; then
    print -r -- "ERROR	$file	status check : ERROR"
  elif [[ $out =~ 'number of[[:space:]]+records[[:space:]]*:[[:space:]]*0([^0-9]|$)' ]]; then
    print -r -- "ERROR	$file	0 records"
  elif (( rc != 0 )); then
    print -r -- "ERROR	$file	hipo-utils exit $rc"
  else
    print -r -- "OK	$file"
  fi
  exit 0
fi
# --doctor-one re-indexes a file in place, then re-tests it to confirm. The file
# counts as FIXED only if the re-test passes the same bar as --check-one's OK.
if [[ $1 == --doctor-one ]]; then
  file=$2
  hipo-utils -doctor "$file" >/dev/null 2>&1
  out=$(hipo-utils -test "$file" 2>&1); rc=$?
  if [[ $out == *"does not appear to have an index"* ]] \
     || [[ $out == *"status check : ERROR"* ]] \
     || [[ $out =~ 'number of[[:space:]]+records[[:space:]]*:[[:space:]]*0([^0-9]|$)' ]] \
     || (( rc != 0 )); then
    print -r -- "FIXFAIL	$file"
  else
    print -r -- "FIXED	$file"
  fi
  exit 0
fi

# ── defaults / argument parsing ─────────────────────────────────────────────
default_dir=/volatile/clas12/ouillon/rgd_recon_data_018614_torus_0998/recon
jobs=16
bad_list=""
max_list=40            # cap per-category listing in the summary (0 = no cap)
do_fix=0
inputs=()
while (( $# )); do
  case $1 in
    -j|--jobs)   jobs=$2; shift 2 ;;
    --bad-list)  bad_list=$2; shift 2 ;;
    --max-list)  max_list=$2; shift 2 ;;
    --fix)       do_fix=1; shift ;;
    -h|--help)   sed -n '2,29p' $0; exit 0 ;;
    --)          shift; inputs+=("$@"); break ;;
    -*)          print -u2 "unknown argument: $1"; exit 2 ;;
    *)           inputs+=($1); shift ;;
  esac
done
(( ${#inputs} )) || inputs=($default_dir)

# ── resolve inputs (a directory globs its *.hipo; a file is taken as-is) ─────
if ! command -v hipo-utils >/dev/null 2>&1; then
  print -u2 "error: hipo-utils not found on PATH"; exit 1
fi
files=()
for in in $inputs; do
  if [[ -d $in ]]; then
    files+=( $in/*.hipo(N) )
  elif [[ -f $in ]]; then
    files+=( $in )
  else
    print -u2 "warn: '$in' is not a file or directory; skipping"
  fi
done
if (( ! ${#files} )); then
  print -u2 "error: no .hipo files to check"; exit 1
fi

# ── run the checks in parallel, stream ERROR lines + a live counter ─────────
tmp=$(mktemp); trap 'rm -f $tmp' EXIT
print "checking ${#files} file(s) with $jobs parallel job(s)..."
typeset -i i=0 total=${#files} nok=0 nidx=0 nerr=0
while IFS=$'\t' read -r tag fname reason; do  # not `path`: that aliases $PATH in zsh
  print -r -- "$tag	$fname	$reason" >> $tmp
  (( ++i ))
  case $tag in
    OK)      (( ++nok )) ;;
    NOINDEX) (( ++nidx )) ;;
    ERROR)   (( ++nerr ))
             print -r -- ""                    # break the \r counter line
             print -r -- "  ✗ ERROR  $fname  ($reason)" ;;
  esac
  print -nr -u2 -- $'\r'"  checked $i/$total  (ok $nok, no-index $nidx, error $nerr)   "
done < <(print -rl -- $files | xargs -P $jobs -I '{}' zsh "${0:A}" --check-one '{}')
print -r -u2 -- ""

# ── summary ─────────────────────────────────────────────────────────────────
list_cat() {  # $1 = TAG, $2 = heading; prints up to $max_list paths
  local tag=$1 head=$2 n
  n=$(grep -c "^$tag	" $tmp)
  (( n )) || return
  print "\n$head ($n):"
  if (( max_list > 0 )); then
    grep "^$tag	" $tmp | cut -f2 | head -n $max_list | sed 's/^/  /'
    (( n > max_list )) && print "  ... and $(( n - max_list )) more (use --bad-list FILE for the full list)"
  else
    grep "^$tag	" $tmp | cut -f2 | sed 's/^/  /'
  fi
}

print "────────────────────────────────────────"
print "checked  : $total"
print "ok       : $nok"
print "no-index : $nidx   (readable by rec-vz-hist/swim; re-index with 'hipo-utils -doctor' or --fix)"
print "error    : $nerr"
list_cat ERROR   "ERROR files (genuine failures)"
list_cat NOINDEX "NO-INDEX files"

if [[ -n $bad_list ]]; then
  grep -E '^(NOINDEX|ERROR)	' $tmp | cut -f2 > $bad_list
  print "\nwrote $(( nidx + nerr )) flagged path(s) to $bad_list"
fi

# ── optional re-index pass (--fix): hipo-utils -doctor on every flagged file ─
if (( do_fix && (nidx + nerr) > 0 )); then
  print "\n── fixing: hipo-utils -doctor on $(( nidx + nerr )) flagged file(s) (rewrites in place) ──"
  typeset -i j=0 ftot=$(( nidx + nerr )) nfixed=0 nfail=0
  while IFS=$'\t' read -r tag fname _; do
    (( ++j ))
    if [[ $tag == FIXED ]]; then (( ++nfixed )); else
      (( ++nfail )); print -r -- ""; print -r -- "  ✗ still bad after doctor: $fname"
    fi
    print -nr -u2 -- $'\r'"  fixed $j/$ftot  (ok $nfixed, fail $nfail)   "
  done < <(grep -E '^(NOINDEX|ERROR)	' $tmp | cut -f2 \
             | xargs -P $jobs -I '{}' zsh "${0:A}" --doctor-one '{}')
  print -r -u2 -- ""
  print "re-indexed: $nfixed   still bad: $nfail"
fi

(( nidx + nerr == 0 ))
