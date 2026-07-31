import argparse
import json
import collections
import subprocess
from pathlib import Path
import os
from typing import List

# ---------------------------------------------------------------------------
# build_workflow_data_clara.py
#
# SWIF2 workflow generator for RECONSTRUCTION OF REAL (decoded) DATA. Each job runs
# many single-file recon-util processes in parallel (THREADS at a time via xargs -P)
# instead of one multi-threaded run-clara / CLARA DPE.
#
# Pass one or more run numbers with --runs to cook several at once. All runs go
# into a SINGLE swif2 workflow; each run's recon is written under a per-run
# subdir (<output>/recon/<run>/job_<i>) mirroring the decoded input tree
# (…/recon/<run>), so the runs never clobber each other. Each job processes a
# BATCH of FILES_PER_JOB decoded files. Per job:
#
#   1. SWIF2 stages the batch's decoded files from /mss to the worker node.
#   2. One recon-util per file, THREADS in parallel:
#         recon-util -i <file> -o <recon dir>/rec_<file> -y <yaml> [-n N]
#      producing <recon dir>/rec_<original-basename> per input.
#
# recon-util uses our locally-built coatjava 13.0.0 (with the DC Lorentz-angle fix); the
# job exports COATJAVA to that dist so the loaded module's coatjava can't override it.
#
# Run this ON THE FARM (ifarm) where /mss is visible. For an off-farm preview,
# pass --data-dir pointing at a directory of placeholder files.
# ---------------------------------------------------------------------------

__JSONFORMAT = {"indent": 2, "separators": (",", ": ")}

# Default run(s) to cook. Pass one or more on the CLI with --runs to cook
# several at once; each run becomes its OWN independent swif2 workflow
# (rgd_vz_<run><suffix>), so they run concurrently and monitor separately.
RUN: str = "018928"
DEFAULT_RUNS: List[str] = [RUN]

# DC drift-distance Lorentz-angle scale (coatjava FittedHit.LORENTZ_ANGLE_SCALE, passed via
# -Ddc.lorentz.scale). MEASURED AND REFUTED as the Vz-shift cause: cooks at 0 vs 1 gave
# c0/c1 identical to <1% (see docs/vz_shift_dc_lorentz_mechanism.md). Keep "1" = production;
# scale 0 also loses ~5% of the tracks. Left in place as a working diagnostic knob.
DC_LORENTZ_SCALE: str = "1"

# Torus field-map displacement [cm] applied IN RECONSTRUCTION (Kalman fit AND vertex swim),
# via coatjava MagFieldsEngine's COAT_MAGFIELD_TORUS{X,Y,Z}SHIFT env -> Swimmer.setShiftX/Y/Z.
# Leading candidate for the polarity-odd (1/p)*cot(theta) Vz walk: a torus map error that is
# NOT a pure scale does not renormalize into q/p, so unlike a torus SCALE it does move the
# vertex in a full re-cook. Scan for the value that drives c1 -> 0; the same displacement
# must flatten BOTH torus polarities. "0" = production geometry (no shift).
TORUS_X_SHIFT: str = "0"
TORUS_Y_SHIFT: str = "0"
TORUS_Z_SHIFT: str = "0"

# Which torus field map the reconstruction uses (selects the recon yaml; see YAML_PATH).
#   "coarse" = dist default Full_torus_r251_phi181_z251_25Jan2021 (131 MB) -- lor0/lor1 used this
#   "fine"   = Full_torus_r501_phi361_z501_31Mar2021 (1.1 GB, 8x the grid), symlinked into
#              coatjava-13.0.0/coatjava/etc/data/magfield/
# Diagnostic: a map that matched the real field would give the SAME vertex either way, so if
# the Vz(p,theta) walk depends on the map, the MAP (not the reconstruction) is implicated.
# The fine map costs ~1.1 GB of heap PER recon-util instance -- see RECON_UTIL_XMX/RAM.
TORUS_MAP: str = "fine"          # "coarse" | "fine"


def _shift_tag() -> str:
    """Compact name tag for any non-zero torus map shift, so each shift pass gets its own
    workflow / output dir. Empty when all shifts are 0, which keeps the shift-0 names
    identical to the cooks already on disk (e.g. rgd_vz_018928_lor1)."""
    parts = []
    for axis, v in (("x", TORUS_X_SHIFT), ("y", TORUS_Y_SHIFT), ("z", TORUS_Z_SHIFT)):
        if float(v) != 0.0:
            parts.append(f"t{axis}{v}".replace("-", "m").replace(".", "p"))
    return ("_" + "_".join(parts)) if parts else ""


def _map_tag() -> str:
    """Name tag for a non-default torus map, so map passes never collide with the
    coarse-map cooks already on disk."""
    return "" if TORUS_MAP == "coarse" else f"_map{TORUS_MAP}"


# Re-cook version suffix: gives every workflow a fresh name + /volatile output dir so
# re-cooks don't collide. Encodes the Lorentz scale, any torus shift and the torus map, so
# every knob combination gets its own workflow/output dir. "" = none.
RECOOK_SUFFIX: str = f"_lor{DC_LORENTZ_SCALE}{_shift_tag()}{_map_tag()}"

# Decoded HIPO on tape: RG-D decoded files under
# /mss/clas12/rg-d/production/decoded/13.0.0/<run>/ (chunk-merged,
# clas_<run>.evio.NNNNN-MMMMM.hipo) — the input this recon yaml (HipoToHipoReader)
# consumes. `{run}` is substituted per run; override a single run's directory with
# --data-dir, or the whole pattern with --data-dir-template.
DATA_DIR_TEMPLATE: str = "/mss/clas12/rg-d/production/decoded/13.0.0/{run}"
DATA_GLOB: str = "*.hipo"   # decoded hipo (clas_<run>.evio.NNNNN-MMMMM.hipo)

# Base /volatile area; the workflow output dir lives below this.
OUTPUT_BASE: str = "/volatile/clas12/ouillon"

# Recon config: RG-D data AI-DST reconstruction chain (repo config/). NOTE its reader is
# HipoToHipoReader -> it reconstructs from DECODED HIPO, not raw evio.
# ONE YAML PER TORUS MAP: MagFieldsEngine.chooseEnvOrYaml checks the YAML *first*, so a
# COAT_MAGFIELD_TORUSMAP env var is silently IGNORED whenever the yaml sets magfieldTorusMap
# (ours does). _torusfine.yaml is a sed-copy of the base differing ONLY in that one line.
# Selected by TORUS_MAP; override per run with --yaml.
_YAML_BY_MAP: dict[str, str] = {
    "coarse": "/work/clas12b/users/ouillon/swim-cpp/config/rgd_250603_data-aidst.yaml",
    "fine": "/work/clas12b/users/ouillon/swim-cpp/config/rgd_250603_data-aidst_torusfine.yaml",
}
YAML_PATH: str = _YAML_BY_MAP[TORUS_MAP]

# run-clara + the CLARA install from our locally-built coatjava 13.0.0 carrying the DC
# Lorentz-angle fix (farm2/coatjava-13.0.0 == /work/clas12b/users/ouillon/swim-cpp/
# coatjava-13.0.0 on ifarm). 13.0.0 matches the decoded data's CCDB era; 14.1.2 needs
# /calibration/dc/v2/tdc_cuts, absent for these runs -> NPE in HitReader.fetch_DCHits.
# The fix lands in the coatjava dist jars (lib/) that recon-util loads.
COATJAVA_DIST: str = "/work/clas12b/users/ouillon/swim-cpp/coatjava-13.0.0/coatjava"
RECON_UTIL: str = f"{COATJAVA_DIST}/bin/recon-util"

# Parallelism: each job runs one single-file recon-util per input, THREADS of them at
# once (xargs -P). FILES_PER_JOB defaults to THREADS so a full job saturates its cores;
# each recon-util is single-threaded (SerialGC), so we request one core per instance.
THREADS: int = 10
FILES_PER_JOB: int = THREADS
RECON_PREFIX: str = "rec_"

DEBUG_COUNT: int = 2          # --debug: number of files for a quick validation
DEBUG_EVENTS: int = 1000      # --debug: events per file (run-clara -n); -1 = all
DEBUG_PARTITION: str = "priority"
DEBUG_TIME_SECS: str = "3600"   # 1 h: the priority partition rejects the 24 h production limit

# ---------------------------------------------------------------------------
# Single clean module-setup prefix (same as the other generators).
# ---------------------------------------------------------------------------
base_command: str = (
    "echo $SHELL; "
    "source /etc/profile.d/modules.sh; "
    "module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles; "
    "module use /scigroup/cvmfs/geant4/modules; "
    "module purge; "
    "module load clas12; "
    'echo "Modules in use:" ; '
    "module list; "
    # Force recon-util to use OUR 13.0.0 build, not the coatjava the module put on PATH
    # (recon-util/env.sh honors $COATJAVA). Without this the stock 14.1.2 would run and
    # NPE on the missing /calibration/dc/v2/tdc_cuts table.
    f"export COATJAVA={COATJAVA_DIST}; "
    # Torus map displacement in RECONSTRUCTION (Kalman fit + vertex swim). MagFieldsEngine
    # reads these env vars only when the yaml has no magfieldTorus*Shift key (ours has none),
    # and logs "run with torus x shift in tracking ... = <v> cm" -- so the job log confirms
    # the shift actually took. Applied via Swimmer -> MagneticFields.getTorus().setShiftX/Y/Z.
    f"export COAT_MAGFIELD_TORUSXSHIFT={TORUS_X_SHIFT}; "
    f"export COAT_MAGFIELD_TORUSYSHIFT={TORUS_Y_SHIFT}; "
    f"export COAT_MAGFIELD_TORUSZSHIFT={TORUS_Z_SHIFT}; "
    # DC drift-distance Lorentz-angle scale for the Vz-shift study: pass
    # -Ddc.lorentz.scale to the tracking JVM (coatjava FittedHit.LORENTZ_ANGLE_SCALE).
    # DC_LORENTZ_SCALE="0" turns the torus-polarity-odd DC drift correction OFF
    # (diagnostic); "1" = production; scan other values to calibrate. Set via
    # _JAVA_OPTIONS so it reaches the CLARA DPE service JVM (DCTBEngine/FittedHit),
    # not just the launcher. Appended to preserve the tmpdir opts already there.
    f'export _JAVA_OPTIONS="${{_JAVA_OPTIONS:+$_JAVA_OPTIONS }}-Ddc.lorentz.scale={DC_LORENTZ_SCALE}"; '
    'echo "_JAVA_OPTIONS=$_JAVA_OPTIONS"; '
)

# Per-job resources are sized to the number of parallel recon-util instances and the
# staged input volume (see sized_resources): one core + ~2 GiB per instance, ~12 GiB
# scratch per staged decoded file (~9 GiB each + output). This base template holds the rest.
_GiB: int = 1024 ** 3
# recon-util's built-in heap is only -Xmx1536m; the AI tracking (MLTD/DCHAI GNN track
# finder) OOMs there, so override it per process via `-- -Xmx<X>` (a jvm option after
# `--`, which recon-util places AFTER its built-in -Xmx on the java line, so it wins).
RECON_UTIL_XMX: str = "8g"        # 6g sufficed with the coarse map; the fine map is ~1.1 GB
RECON_UTIL_RAM: int = 9 * _GiB    # slurm RAM cap per instance: RECON_UTIL_XMX + overhead
BASE_RAM: int = 4 * _GiB          # headroom for the shell / staging
PER_FILE_DISK: int = 12 * _GiB    # ~9 GiB decoded input staged + its rec_ output

recon_resources: dict[str, str] = {
    "constraint": "el9",
    "account": "clas12",
    "partition": "production",
    "track": "osg",
    "shell": "/bin/bash",
    "time_secs": "86400",         # 24 hours
}


def sized_resources(n_files_per_job: int) -> dict[str, str]:
    """Resources for a job that runs up to THREADS parallel recon-util over
    n_files_per_job staged files: one core + RECON_UTIL_RAM per parallel instance,
    PER_FILE_DISK per staged file."""
    n_par = min(THREADS, max(1, n_files_per_job))
    r = dict(recon_resources)
    r["cpu_cores"] = str(n_par)
    r["ram_bytes"] = str(n_par * RECON_UTIL_RAM + BASE_RAM)
    r["disk_bytes"] = str(n_files_per_job * PER_FILE_DISK)
    return r


def remote_uri(path: str) -> str:
    """SWIF2 input URI for a staged file.

    Tape files (under /mss) need the `mss:` scheme; files already on disk
    (e.g. /volatile, /cache) are referenced by their bare path.
    """
    return f"mss:{path}" if path.startswith("/mss") else path


def with_partition(resources: dict[str, str], partition: str) -> dict[str, str]:
    """Return a copy of a resource dict with the partition overridden."""
    updated = dict(resources)
    updated["partition"] = partition
    return updated


def workflow_base_name(runs: List[str]) -> str:
    """Workflow / output-dir base name (with the Lorentz-scale re-cook suffix). A
    single run is rgd_vz_<run><suffix>; several runs use the run range so the one
    shared workflow still names the span it covers."""
    if len(runs) == 1:
        return f"rgd_vz_{runs[0]}{RECOOK_SUFFIX}"
    lo, hi = min(runs), max(runs)
    return f"rgd_vz_{lo}_{hi}{RECOOK_SUFFIX}"


def read_data_files(data_dir: str) -> List[str]:
    """List the run's input files matching DATA_GLOB, sorted by chunk."""
    folder_path: Path = Path(data_dir)
    if folder_path.exists() and folder_path.is_dir():
        files: list[str] = [str(f) for f in folder_path.glob(DATA_GLOB) if f.is_file()]
        files.sort()
        print(f"Found {len(files)} '{DATA_GLOB}' files in: {folder_path}")
        return files
    print(f"Folder does not exist or is not a directory: {folder_path}")
    return []


def chunk(items: List[str], size: int) -> List[List[str]]:
    """Split a list into consecutive batches of at most `size` items."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def build_command(mss_files: List[str], output_recon: str, yaml_path: str,
                  n_events: int = -1) -> str:
    """Reconstruct the staged batch by launching one single-file recon-util per input,
    up to THREADS at a time (xargs -P). Each writes <output_recon>/rec_<basename>.
    `n_events` caps events per file via recon-util -n; -1 (default) = all events.
    xargs exits non-zero if any recon-util fails, so the swif2 job fails on any error.
    """
    local_names = [Path(f).name for f in mss_files]
    nevents_opt = f"-n {n_events} " if n_events != -1 else ""
    names = " ".join(local_names)  # clas_*.evio.*.hipo basenames: no spaces/globs
    steps: list[str] = [
        f"mkdir -p {output_recon}",
        f"printf '%s\\n' {names} | xargs -P {THREADS} -I@ "
        # -d 0: EngineProcessor debug OFF -> normal INFO logging instead of the default
        # debug.properties (ConsoleHandler ALL), which floods stderr with per-event/segment
        # Level.FINE lines (multi-GB .err files). -- -Xmx overrides recon-util's built-in heap.
        f"{RECON_UTIL} -i @ -o {output_recon}/{RECON_PREFIX}@ -y {yaml_path} {nevents_opt}-d 0 "
        f"-- -Xmx{RECON_UTIL_XMX}",
    ]
    return "; ".join(steps)


def create_recon_job(name: str, command: str, mss_files: List[str], total: int,
                     index: int, resources: dict[str, str]) -> collections.OrderedDict:
    """Build one batched run-clara job, staging all inputs to the worker node.

    Every file in the batch is declared as a SWIF2 input so it is staged to the
    node under its basename before the command runs. Tape inputs (/mss) use the
    `mss:` scheme; disk inputs (/volatile, /cache) use their bare path.
    """
    full_command = base_command + f'echo "Processing batch {index + 1}/{total}"; ' + command

    job = collections.OrderedDict(resources)
    job.update({
        "name": name,
        "phase": 0,
        "inputs": [{"local": Path(f).name, "remote": remote_uri(f)} for f in mss_files],
        "command": [full_command],
    })
    return job


def submit_workflow(workflow_name: str, json_path: Path) -> None:
    """Import and run a generated workflow JSON via swif2.

    Runs `swif2 import -file <json>` then `swif2 run -workflow <name>`. Must be
    executed on a node where the swif2 client is available (ifarm).
    """
    import_cmd = ["swif2", "import", "-file", str(json_path)]
    run_cmd = ["swif2", "run", "-workflow", workflow_name]
    for cmd in (import_cmd, run_cmd):
        print(f"  $ {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  ERROR: command failed (exit {result.returncode}); "
                  f"skipping remaining submit steps for {workflow_name}")
            return


def build_run_jobs(run: str, data_dir: str, args: argparse.Namespace,
                   resources: dict[str, str], n_events: int,
                   output_recon: str, base_name: str,
                   yaml_path: str) -> List[collections.OrderedDict]:
    """Build the run-clara jobs for one run of the shared workflow.

    Recon output goes under <output_recon>/<run>/job_<i>, mirroring the decoded
    input tree (…/recon/<run>). Returns the jobs (empty if the run has no
    input files).
    """
    data_files = read_data_files(data_dir)
    if not data_files:
        print(f"  run {run}: no input files ({DATA_GLOB}) in {data_dir}; skipping.")
        return []

    if args.debug:
        data_files = data_files[:args.debug_count]
        print(f"  run {run} DEBUG: {len(data_files)} file(s), {n_events} events/file, "
              f"partition={DEBUG_PARTITION}")

    batches = chunk(data_files, args.files_per_job)
    print(f"Run {run}: {len(batches)} job(s) ({args.files_per_job} files/job, {THREADS} threads)")

    jobs: List[collections.OrderedDict] = []
    for i, batch in enumerate(batches):
        job_name: str = f"{base_name}_{run}_recon_{i}"
        # Each job gets its OWN run-clara output dir, laid out as recon/<run>/job_<i>
        # to mirror the decoded input tree. run-clara writes a per-run filelist.txt
        # (and log/) into its -o directory; if jobs shared one recon dir on
        # /volatile they clobber each other's filelist.txt and die with "Input
        # files do not exist" / "empty list of input files". A per-run, per-job
        # subdir isolates them.
        job_output_recon: str = f"{output_recon}/{run}/job_{i}"
        command = build_command(batch, job_output_recon, yaml_path, n_events)
        jobs.append(create_recon_job(job_name, command, batch, len(batches), i, resources))
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Generate SWIF2 workflow JSON: run-clara recon of real decoded data")
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, metavar="RUN",
                        help="One or more run numbers to cook; each becomes its own swif2 "
                             f"workflow, submitted concurrently (default: {' '.join(DEFAULT_RUNS)})")
    parser.add_argument("--debug", action="store_true",
                        help=f"Small validation workflow ({DEBUG_COUNT} files) on the {DEBUG_PARTITION} partition")
    parser.add_argument("--debug-count", type=int, default=DEBUG_COUNT,
                        help=f"Number of files in --debug mode (default: {DEBUG_COUNT})")
    parser.add_argument("--debug-events", type=int, default=DEBUG_EVENTS,
                        help=f"Events per file in --debug mode (default: {DEBUG_EVENTS})")
    parser.add_argument("--files-per-job", type=int, default=FILES_PER_JOB,
                        help=f"Decoded files reconstructed per job (default: {FILES_PER_JOB})")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override the input directory for a SINGLE run "
                             "(not allowed with multiple --runs; use --data-dir-template instead)")
    parser.add_argument("--data-dir-template", type=str, default=DATA_DIR_TEMPLATE,
                        help="Pattern for each run's input directory; '{run}' is "
                             f"substituted (default: {DATA_DIR_TEMPLATE})")
    parser.add_argument("--yaml", type=str, default=YAML_PATH,
                        help=f"Recon YAML to cook with (default: {YAML_PATH})")
    parser.add_argument("--submit", action="store_true",
                        help="After writing the JSON, submit it with 'swif2 import' + 'swif2 run' (run on ifarm)")
    args = parser.parse_args()

    runs: List[str] = args.runs
    if args.data_dir is not None and len(runs) > 1:
        parser.error("--data-dir overrides a single run's directory; with multiple "
                     "--runs use --data-dir-template (with '{run}') instead")

    n_events = args.debug_events if args.debug else -1

    # Size per-job resources to how many files each job stages/reconstructs in parallel:
    # debug jobs take debug_count files, production jobs take files_per_job.
    files_per_job = args.debug_count if args.debug else args.files_per_job
    resources = sized_resources(files_per_job)
    if args.debug:
        resources = with_partition(resources, DEBUG_PARTITION)
        resources["time_secs"] = DEBUG_TIME_SECS   # priority partition caps below 24 h

    out_dir = Path("json")
    out_dir.mkdir(parents=True, exist_ok=True)

    # One shared workflow for all runs; per-run recon subdirs keep them apart.
    base_name = workflow_base_name(runs)
    workflow_name = base_name
    if args.debug:
        workflow_name = f"{workflow_name}_debug"
    output_recon = f"{OUTPUT_BASE}/{workflow_name}/recon"

    try:
        os.makedirs(output_recon, exist_ok=True)
    except OSError as exc:
        print(f"WARNING: could not create {output_recon} ({exc}). "
              f"Create it on the farm before submitting.")

    yaml_path = args.yaml

    print(f"Generating 1 workflow ({workflow_name}) for run(s): {', '.join(runs)}")

    jobs: List[collections.OrderedDict] = []
    for run in runs:
        data_dir = (args.data_dir if args.data_dir is not None
                    else args.data_dir_template.format(run=run))
        jobs.extend(build_run_jobs(run, data_dir, args, resources, n_events,
                                   output_recon, base_name, yaml_path))

    if not jobs:
        print("ERROR: no jobs built (no input files found for any run). Exiting.")
        return

    workflow = collections.OrderedDict(
        {"name": workflow_name, "site": "jlab/enp", "max_dispatched": 1500, "jobs": jobs})

    output_path = out_dir / f"{workflow_name}.json"
    with output_path.open("w") as f:
        json.dump(workflow, f, **__JSONFORMAT)
    print(f"  {output_path} ({len(jobs)} jobs across {len(runs)} run(s))")

    if args.submit:
        submit_workflow(workflow_name, output_path)

    print(f"Done: workflow JSON written under {out_dir}/")


if __name__ == "__main__":
    main()
