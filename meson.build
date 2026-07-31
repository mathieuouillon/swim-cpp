project(
  'swim',
  'cpp',
  version : '0.1.0',
  default_options : [
    'cpp_std=c++23',
    'warning_level=1',
    'buildtype=release',
  ],
)

cpp = meson.get_compiler('cpp')

# hipo4 gates its LZ4 (de)compression behind `#ifdef __LZ4__` (file.cpp /
# file_writer.cpp); HIPO records are LZ4-compressed, so this must be defined
# project-wide (before the hipo subdir) or no events can be read.
add_project_arguments('-D__LZ4__', language : 'cpp')

# ---- lz4 (needed by hipo4) ------------------------------------------------
# pkg-config name is usually 'liblz4'; some distros ship 'lz4'. Fall back to a
# bare library lookup so the farm build works even without a .pc file.
lz4_dep = dependency('liblz4', required : false)
if not lz4_dep.found()
  lz4_dep = dependency('lz4', required : false)
endif
if not lz4_dep.found()
  lz4_dep = declare_dependency(dependencies : cpp.find_library('lz4', required : true))
endif

# ---- fmt (needed by hipo4) -------------------------------------------------
# Try the system (pkg-config / cmake), then a macOS Homebrew keg (no
# pkg-config), then a wrap subproject so it also builds on the farm where fmt
# may not be installed.
fmt_dep = dependency('fmt', required : false)
if not fmt_dep.found() and host_machine.system() == 'darwin'
  fmt_lib = cpp.find_library('fmt', dirs : ['/opt/homebrew/opt/fmt/lib'], required : false)
  if fmt_lib.found()
    fmt_dep = declare_dependency(
      include_directories : include_directories('/opt/homebrew/opt/fmt/include', is_system : true),
      dependencies        : fmt_lib,
    )
  endif
endif
if not fmt_dep.found()
  fmt_dep = dependency('fmt', fallback : ['fmt', 'fmt_dep'])
endif

# ---- project_libs MUST exist before the hipo subdir -----------------------
project_libs = []
subdir('external/hipo4')   # defines hipo_dep and appends to project_libs
subdir('external/argparse')  # header-only: defines argparse_dep

# ---- CERN ROOT (keg-only: discover via root-config) -----------------------
root_config = find_program('root-config', required : true)
root_incdir = run_command(root_config, '--incdir', check : true).stdout().strip()
root_libdir = run_command(root_config, '--libdir', check : true).stdout().strip()
root_libs   = run_command(root_config, '--libs',   check : true).stdout().strip().split()
root_dep = declare_dependency(
  include_directories : include_directories(root_incdir, is_system : true),
  link_args           : root_libs + ['-Wl,-rpath,' + root_libdir],
)

# ---- Boost.odeint (header-only) -------------------------------------------
boost_dep = dependency('boost', required : false)
if not boost_dep.found() and host_machine.system() == 'darwin'
  boost_dep = declare_dependency(
    include_directories : include_directories('/opt/homebrew/include', is_system : true),
  )
endif
if not boost_dep.found()
  # On the farm, ensure Boost headers are discoverable (e.g. `module load boost`
  # or set BOOST_ROOT); odeint is header-only.
  boost_dep = dependency('boost', required : true)
endif

# ---- SQLite3 (CCDB snapshot reader, src/common/ccdb.hpp) -------------------
sqlite3_dep = dependency('sqlite3', required : true)

# Shared code lives in src/common/; each program has its own src/<program>/ dir.
common_inc = include_directories('src/common')

# Shared translation units used by more than one program / test.
common_field_swim = files(
  'src/common/field.cpp',
  'src/common/swim.cpp',
)

# ---- swim-analysis: CLAS12 v_z / PID diagnostic ---------------------------
executable(
  'swim-analysis',
  ['src/swim-analysis/main.cpp', 'src/swim-analysis/analysis.cpp'] + common_field_swim,
  dependencies        : [hipo_dep, root_dep, boost_dep, fmt_dep, argparse_dep],
  include_directories : [common_inc, include_directories('src/swim-analysis')],
  install             : true,
)

# ---- hipo2root: dump REC::Particle kinematics to a flat ROOT TTree ---------
executable(
  'hipo2root',
  ['src/hipo2root/hipo2root.cpp'],
  dependencies        : [hipo_dep, root_dep, fmt_dep, argparse_dep, sqlite3_dep],
  include_directories : common_inc,
  install             : true,
)

# ---- rec-vz-hist: reconstructed vz in (p,theta) bins, straight from HIPO ----
# Reads recon .hipo directly (hipo4) and histograms EB-reconstructed vz per
# (species, p, theta) cell for FD e-/pi+/pi- -- no particles tree, no swim, so
# no Boost. Built to overlay two reconstruction passes cell-by-cell.
executable(
  'rec-vz-hist',
  ['src/rec-vz-hist/rec_vz_hist.cpp'],
  dependencies        : [hipo_dep, root_dep, fmt_dep, argparse_dep],
  include_directories : common_inc,
  install             : true,
)

# ---- hipo-check: validate / recover-read a (possibly index-less) HIPO file --
# Forensic reader for files whose trailer index is missing (stock `hipo-utils
# -dump` then shows 0 records). Uses the hipo4 reader's sequential-scan
# fallback to report recoverable record/event counts, banks and a content
# tally. No ROOT, no swim -- just hipo4 + fmt + argparse, so it builds fast.
executable(
  'hipo-check',
  ['src/hipo-check/hipo_check.cpp'],
  dependencies        : [hipo_dep, fmt_dep, argparse_dep],
  install             : true,
)

# ---- vz-swim-hist: electron vz in (p,theta) bins with DC-R1 beamline swim --
# Reads the flat particles.root TTree (no hipo4 library needed); its progress
# bar is the project-local src/common/progresstracker.hpp.
executable(
  'vz-swim-hist',
  ['src/vz-swim-hist/vz_swim_hist.cpp'] + common_field_swim,
  dependencies        : [root_dep, boost_dep, fmt_dep, argparse_dep],
  include_directories : common_inc,
  install             : true,
)

# ---- unit tests (no ROOT/hipo needed) -------------------------------------
test_swim = executable(
  'test_swim',
  ['test/test_swim.cpp'] + common_field_swim,
  dependencies        : [boost_dep],
  include_directories : common_inc,
)
test('swim', test_swim)

test_field = executable(
  'test_field',
  ['test/test_field.cpp', 'src/common/field.cpp'],
  include_directories : common_inc,
)
test('field', test_field, timeout : 60)  # loads the 8.7 MB solenoid map

test_argparse = executable(
  'test_argparse',
  ['test/test_argparse.cpp'],
  dependencies : argparse_dep,
)
test('argparse', test_argparse)

test_ccdb = executable(
  'test_ccdb',
  ['test/test_ccdb.cpp'],
  dependencies        : sqlite3_dep,
  include_directories : common_inc,
)
test('ccdb', test_ccdb)
