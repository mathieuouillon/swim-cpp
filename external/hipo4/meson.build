hipo_sources = files(
  'file.cpp',
  'file_writer.cpp',
  'schema.cpp',
)
hipo_headers = files(
  'hipo.hpp',
  'constants.h',
  'error.hpp',
  'schema.hpp',
  'views.hpp',
  'file.hpp',
  'file_writer.hpp',
)

hipo_lib = library(
  meson.project_name(),
  hipo_sources,
  dependencies: [ lz4_dep ],
  install: true
)

install_headers(hipo_headers, subdir: meson.project_name())
project_libs += hipo_lib

hipo_dep = declare_dependency(
  include_directories: include_directories('..'), # use `..` so consumers use `#include "hipo4/___.h"`
  link_with: hipo_lib,
)
