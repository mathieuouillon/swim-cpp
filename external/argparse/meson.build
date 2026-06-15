# Header-only C++23 command-line parser. Mirrors the external/hipo4 convention:
# the include dir is `..` (i.e. external/), so consumers write
# `#include "argparse/argparse.hpp"`.
argparse_dep = declare_dependency(
  include_directories: include_directories('..'),
)

install_headers(files('argparse.hpp'), subdir: 'argparse')
