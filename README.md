# swim-cpp — CLAS12 v_z / PID diagnostic analysis

A C++ port (from the Rust `farm` analysis) of a CLAS12 v_z / PID diagnostic:
read HIPO files with the vendored **hipo4** library, match reconstructed tracks
to MC truth via `MC::RecMatch` for charged pions/muons, swim them back through
the CLAS12 magnetic field to the beamline, and write ~1722 v_z / PID diagnostic
histograms to a ROOT file.

- Histograms: **CERN ROOT** (`TFile`/`TH1D`/`TH2D`), zstd-compressed.
- Swimmer: **Boost.odeint** (`runge_kutta_dopri5` dense output + closest-approach
  root-find), replacing the Rust `diffsol` solver.
- HIPO reading + record-level parallelism: vendored `external/hipo4` (`hipo::chain`).

## Dependencies

- Meson + Ninja, a C++23 compiler
- CERN ROOT 6 (discovered via `root-config`)
- Boost (header-only `boost/numeric/odeint.hpp`)
- liblz4 and {fmt} (used by hipo4)

## Build

```sh
meson setup build
ninja -C build
meson test -C build      # swimmer + field-map unit tests
```

The build defines `-D__LZ4__` project-wide — hipo4 gates its LZ4 record
decompression behind that macro, and HIPO files are LZ4-compressed, so without
it no events are read.

## Run

```sh
./build/swim-analysis <input>... [--output FILE] [--threads N] [--quiet] \
    --torus  /path/to/Full_torus_r501_phi361_z501_31Mar2021.dat \
    --solenoid /path/to/Symm_solenoid_r601_phi1_z1201_21May2019.dat
```

| flag                  | default                  | meaning                                   |
|-----------------------|--------------------------|-------------------------------------------|
| `<input>...`          | (required)               | `.hipo` files, quoted globs, dirs, `@list`|
| `--output`, `-o`      | `vz.root`                | output ROOT file                          |
| `--threads`, `-t`     | `0` (auto)               | worker threads (`-j` also accepted)       |
| `--quiet`, `-q`       | off                      | hide the progress bar                     |
| `--torus`             | `Full_torus_…dat`        | torus field map (clas12-cmag binary)      |
| `--solenoid`          | `Symm_solenoid_…dat`     | solenoid field map                        |
| `--torus-scale`       | `-1`                     | torus field scale (inbending)             |
| `--solenoid-scale`    | `-1`                     | solenoid field scale                      |
| `--solenoid-z-shift`  | `-3`                     | solenoid map z translation (cm)           |

`@listfile.txt` holds one path per line (`#` comments and blank lines ignored).
The ~1 GB torus map is loaded once and shared read-only across worker threads.

The output ROOT file is compatible with the Rust analysis's `plot_vz.py`.

## Layout

- `src/constants.hpp` — binning, species, region, correlation-variable tables
- `src/field.{hpp,cpp}` — clas12-cmag binary field-map reader + composite field
- `src/swim.{hpp,cpp}` — Boost.odeint swimmer (dopri5 dense output + root-find)
- `src/bank_access.hpp` — per-detector bank readers over `const hipo::bank*`
- `src/analysis.{hpp,cpp}` — `Analysis` accumulator (build / fill / merge / write)
- `src/accumulator_registry.hpp` — thread-local Analysis registry + merge
- `src/main.cpp` — CLI, input expansion, field load, `chain::process`, summary
- `test/` — swimmer straight-line test + field-map load test
