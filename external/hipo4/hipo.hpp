//******************************************************************************
//* HIPO — modern C++23 reading API (umbrella header)                          *
//*                                                                            *
//*     #include "hipo4/hipo.hpp"                                              *
//*                                                                            *
//*     hipo::file f("run.hipo");                                              *
//*     auto particle = f.bank("REC::Particle").value();                       *
//*     auto px       = particle.col<float>("px").value();                     *
//*     double sum = 0;                                                        *
//*     for (hipo::event_view ev : f)                                          *
//*         for (float v : ev.get(particle).col(px))                           *
//*             sum += v;                                                      *
//*                                                                            *
//* Writing mirrors the same design (see hipo4/file_writer.hpp):              *
//*                                                                            *
//*     hipo::file_writer w("out.hipo");                                       *
//*     auto particle = w.add_schema(schema).value();                          *
//*     hipo::event_builder ev;                                                *
//*     ev.add(bank);  w.write(ev).value();  w.close().value();                *
//******************************************************************************
#pragma once

#include "error.hpp"
#include "schema.hpp"
#include "views.hpp"
#include "file.hpp"
#include "file_writer.hpp"
