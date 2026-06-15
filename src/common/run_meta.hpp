// Per-run constants saved by hipo2root into particles.root and read back by
// vz-swim-hist: the run number, the field configuration (already in
// vz-swim-hist's sign convention), and the CCDB beam offset for the run. Stored
// as top-level ROOT TParameters so they survive the TFileMerger merge of the
// per-worker part files. Single source of truth for the metadata schema shared
// by the two programs.
#pragma once

#include "TDirectory.h"
#include "TParameter.h"

namespace vz {

struct run_meta {
    int run = 0;
    double torus_scale = -1.0;
    double solenoid_scale = 1.0;
    double solenoid_z_shift = -3.0;
    double beam_x = 0.0;
    double beam_y = 0.0;

    /// Write the constants as top-level TParameters into `dir` (the open file).
    auto write(TDirectory& dir) const -> void {
        dir.cd();
        TParameter<int>("run", run).Write();
        TParameter<double>("torus_scale", torus_scale).Write();
        TParameter<double>("solenoid_scale", solenoid_scale).Write();
        TParameter<double>("solenoid_z_shift", solenoid_z_shift).Write();
        TParameter<double>("beam_x", beam_x).Write();
        TParameter<double>("beam_y", beam_y).Write();
    }

    /// Read constants from `dir`; absent keys keep the current value. Returns
    /// true when the file actually carries run metadata (a `run` parameter).
    static auto read(TDirectory& dir, run_meta& out) -> bool {
        const auto get_d = [&](const char* key, double& v) {
            if (auto* p = dynamic_cast<TParameter<double>*>(dir.Get(key))) v = p->GetVal();
        };
        bool has = false;
        if (auto* p = dynamic_cast<TParameter<int>*>(dir.Get("run"))) {
            out.run = p->GetVal();
            has = true;
        }
        get_d("torus_scale", out.torus_scale);
        get_d("solenoid_scale", out.solenoid_scale);
        get_d("solenoid_z_shift", out.solenoid_z_shift);
        get_d("beam_x", out.beam_x);
        get_d("beam_y", out.beam_y);
        return has;
    }
};

}  // namespace vz
