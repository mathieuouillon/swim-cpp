# How coatjava reconstructs the DC track vertex (Vtx0_z / REC::Particle.vz)

Findings from reading coatjava (`~/Documents/Java/cj2`, 2026-06), used to align
`vz-swim-hist` with the reconstruction. Line numbers refer to that checkout.

## Call chain

1. `DCTBEngine.processDataEvent`
   (`reconstruction/dc/src/main/java/org/jlab/service/dc/DCTBEngine.java:232-272`)
   reads the beam position and runs the time-based track finding.
2. `TrackCandListFinder.setTrackPars`
   (`reconstruction/dc/src/main/java/org/jlab/rec/dc/track/TrackCandListFinder.java:586-692`)
   takes the Kalman-filter **final state at Region 3** (tilted-sector frame,
   superlayer 6), converts it to the lab frame, **negates momentum and
   charge**, and calls
3. `Swim.SwimToBeamLine(xB, yB)`
   (`common-tools/swim-tools/src/main/java/org/jlab/clas/swimtools/Swim.java:967`).
   The result becomes TBTracks `Vtx0_{x,y,z}`.
4. The EventBuilder copies `Vtx0_*` **verbatim** into REC::Particle
   (`common-tools/clas-reco/src/main/java/org/jlab/clas/detector/DetectorData.java:561-565`);
   no vertex recomputation happens in EB.

## The beamline is NOT the z-axis

`DCTBEngine.java:232-241`:

```java
IndexedTable beamOffset = this.getConstantsManager().getConstants(run, Constants.BEAMPOS);
beamXoffset = beamOffset.getDoubleValue("x_offset", 0, 0, 0);   // CCDB /geometry/beam/position
beamYoffset = beamOffset.getDoubleValue("y_offset", 0, 0, 0);
if(event.hasBank("RASTER::position")){
    beamXoffset += raster_bank.getFloat("x", 0);                // per-event raster
    beamYoffset += raster_bank.getFloat("y", 0);
}
```

The swim stops at the **minimum distance to (xB, yB)** (`BeamLineSwimStopper`,
`Swim.java:918-965`), with the minimum tracked only once z < 2 m. A transverse
beam offset Δr moves the swum vz by ≈ Δr/tanθ along the track: ~2 cm at θ = 6°
per 2 mm of offset — strongly θ-dependent.

## Field configuration

- Torus/solenoid **scales** come from the `RUN::config` bank per run
  (`common-tools/swim-tools/.../MagFieldsEngine.java:62-109`). `hipo2root`
  prints them at startup.
- The **solenoid z-shift** comes from CCDB `/geometry/shifts/solenoid` (or the
  YAML key `magfieldSolenoidShift` / env `COAT_MAGFIELD_SOLENOIDSHIFT`); torus
  x/y/z shifts analogously. Our `--solenoid-z-shift` default (−3 cm) is the
  GEMC-study value, not the run value.
- Interpolation is trilinear (`cnuphys/magfield/.../Cell3D.java:107`), same
  maps as ours.

## Field-scale sign convention (run 18614)

`RUN::config` for run 18614 reports **torus +1.0, solenoid −1.0**. `vz-swim-hist`
now swims in this physical (cnuphys / `RUN::config`) polarity and **reverses the
charge**, exactly like coatjava's `TrackCandListFinder` (see "Call chain"), so
the `RUN::config` scales pass straight through:

| RUN::config | our flag |
|---|---|
| torus T | `--torus-scale T` |
| solenoid S | `--solenoid-scale S` |

(`hipo2root --run-config` prints exactly these.) Reversing *both* the charge and
the field leaves the Lorentz force q·(u×B) invariant, so this reproduces the
pass1 vz bit-for-bit identically to the earlier "negated maps, keep charge"
convention it replaced — it just makes the scales match `RUN::config` / CCDB
directly. Swimming with only one of the two reversed (e.g. keep charge + torus
+1) sends every swum vz hundreds of cm off, which is why the old code negated
the maps. Verified against the MC-truth `vz` on run 18614: swum − true median
≈ 0.3 cm across 0.3–6 GeV for both pions and electrons.

## Integrator parity (not the discrepancy)

- Dormand-Prince 5(4) (`cnuphys/swim/Swimmer.java:43`,
  `cnuphys/rk4/ButcherTableau.java:24`); above p = 0.75 GeV a `SwimZ` variant
  (z as independent variable) with `C = 2.99792458e-4 (GeV/c)^-1 kG^-1 cm^-1` —
  numerically identical to our `K0`.
- No energy loss in the swim (momentum magnitude constant); multiple
  scattering enters only the KF covariance, not the vertex swim.
- Tolerances: accuracy 20 µm, initial step 500 µm, max path 9 m.

## REC::Traj provenance

REC::Traj rows are copied verbatim from the tracking trajectory banks
(`DetectorData.java:448-500`): the KF-transported track at the **last layer of
each superlayer** (layer 6 = R1, 18/24… , 36 = R3), positions in cm, unit
direction cosines, lab frame. The layer-36 row is therefore (a smoothed
version of) the same Region-3 state the vertex swim starts from.

## Consequences for `vz-swim-hist`

| coatjava | our pipeline |
|---|---|
| start at KF final state, DC R3 | `--dc-region 3` (default; uses `dc3_*` branches), `--dc-region 1` for the old R1 start |
| DOCA to (xB, yB) = CCDB + raster | `--beam-x/--beam-y` (CCDB values) + per-event `raster_x/raster_y` branches |
| scales from RUN::config | `hipo2root` prints them; pass `--torus-scale/--solenoid-scale` |
| solenoid z-shift from CCDB | pass `--solenoid-z-shift` (look up `/geometry/shifts/solenoid` for the run) |

CCDB lookups for a run (e.g. 18614):

```bash
ccdb dump /geometry/beam/position -r 18614
ccdb dump /geometry/shifts/solenoid -r 18614
```
