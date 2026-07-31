# The origin of the (p, θ) Vz shift in coatjava reconstruction

**Status:** ❌ **REFUTED BY MEASUREMENT (2026-07-15).** The Lorentz-angle mechanism proposed in this
note is **NOT** the driver of the Vz(p, θ) walk — see *Measured result* below. The source analysis
(call map, the even/odd decomposition, why a torus scale is renormalized) remains valid and useful;
the **conclusion does not**. Kept as a record of a hypothesis that was tested and rejected.

## Measured result (RG-D run 018928, decoded 13.0.0, coatjava 13.0.0 + LORENTZ_ANGLE_SCALE knob)

Two full cooks of the same 152 decoded files, identical in every respect except
`-Ddc.lorentz.scale`, fitted with `fit_vz_cot.py` (peak2 mean per (p, θ) cell,
`mean = base(p) + (a0 + a1/p)·cotθ`):

| cook | Lorentz | c0 (p-flat) | c1 (1/p) |
|---|---|---|---|
| `rgd_vz_018928_lor0` | **OFF** (scale 0) | +0.0501 cm | +0.6025 cm·GeV |
| `rgd_vz_018928_lor1` | production (scale 1) | +0.0508 cm | +0.5977 cm·GeV |

Per-p walk a(p) agrees to ~0.001–0.004 cm across 2.5–8.5 GeV. **Disabling the correction entirely
changes the walk by <1%.** The knob was demonstrably active — lor1 reconstructed **5% more tracks**
(7.40M vs 7.04M FD e-; 61 GB vs 58 GB output) — so the DC drift-distance Lorentz modelling
measurably affects *track finding* but **not** the vertex walk.

**Therefore:** the Vz(p, θ) walk is insensitive to the Lorentz/α drift-distance modelling. The cause
lies elsewhere — the leading candidate is now the **polarity-even floor**, above all the **missing
energy-loss correction** (§5: coatjava applies no dE/dx anywhere in DC tracking or the EB, and the
vertex swim conserves |p|), plus DC/beam alignment for the p-flat c0 term.
**Analysis checkout:** `~/Documents/Java/cj2` (tag 14.1.0-9). **Edit target:** `farm2/coatjava` (tag 14.1.2-4). The relevant code is identical between the two; line numbers below are for `farm2/coatjava`.
**Date:** 2026-07-15.

---

## TL;DR

The forward-track vertex `REC::Particle.vz` shows a shift that depends on momentum `p` and polar angle `θ`, empirically `Δvz ≈ (c0 + c1/p)·cot θ`, **and whose sign follows the torus polarity**. It is **not** a field-map error and **not** a vertex-swim error. It is a **reconstructed-momentum bias** that the (transparent, `|p|`-conserving) vertex swim faithfully images into `vz`.

Inside coatjava the momentum is set by the DC Kalman filter fitting the drift distances. Those drift distances carry a **Lorentz-angle ("isochrone twist") correction whose sign is the torus polarity and whose magnitude is a single hard-coded constant**:

```java
// FittedHit.set_TimeToDistance,  reconstruction/dc/.../hit/FittedHit.java:372-376
double theta0 = Math.acos(1 - 0.02*B);                 // B is |B| (Tesla); coeff 0.02 hard-coded
double alpha  = Math.atan(trkAngle);
alpha -= Math.signum(Swimmer.getTorScale()) * theta0;  // SIGN = torus polarity
```

Because the applied correction flips sign with the torus while its calibration (the `0.02` model) is fixed, **any error in that model is a drift-distance bias that is odd in torus polarity → a momentum bias odd in polarity → a `vz` shift that flips sign with the torus.** That is exactly the observed signature.

Two other effects shape the same observable but do **not** flip sign, so they are the "even floor," not the cause of the flip: (a) **no energy-loss correction** anywhere in DC tracking/EB, and (b) DC/beam **alignment** + **T0/start-time** additive offsets.

---

## 1. The observable and what we already knew

- RG-D dual target → two fixed `vz` peaks. A correct reconstruction must give a peak mean independent of `p` and `θ`. The residual `θ`-walk tracks `cot θ` (a transverse-miss signature, `Δvz = Δρ·cot θ`) with a momentum-dependent coefficient `a(p) = c0 + c1/p` (see `notes/field_alignment.typ` and `vz-swim-hist` `--vz-cot-coeff`/`--vz-cot-p-coeff`, fitted `c0≈0.10 cm`, `c1≈0.52 cm·GeV`).
- **Key prior test (the discriminator).** Taking the full-scale reconstructed momentum and re-swimming with `0.998×torus` in the *swimmer only* flattens the shift (**Method A**); but doing the **full reconstruction** at `0.998×torus` reproduces the shift (**Method B**). Method B ⇒ the shift is invariant under a torus scale in a self-consistent fit, because the Kalman fit renormalizes `q/p` to keep the DC curvature matched to the hits (same hits → same `q/p·B` → same trajectory → same vertex). **This rules out a torus field-scale error** and points to a momentum bias that a scale cannot absorb.
- **Hypothesis (confirmed by the source):** a systematic momentum bias in reconstruction whose sign flips with the torus field.

---

## 2. Where the momentum (hence vz) actually comes from — call map

All live files (the `org.jlab.rec.dc.track.fit.*Doca` classes are **dead code**, never instantiated; the production TB Kalman filter is `org.jlab.clas.tracking.kalmanfilter.zReference.*`).

```
seed |p|  (hit-based, ∫B·dl / bend angle, polarity-EVEN)
    TrackCandListFinder.calcInitTrkP : pxz = |LIGHTVEL·iBdl/Δθ|   track/TrackCandListFinder.java:195
TB Kalman filter (full torus+solenoid field; |p| conserved, no dE/dx)
    new KFitter(true,30,1,dcSwim,…)                                service/dc/DCTBEngine.java:300
    EOM  d(tx)/dz = (q/p)·v·Ax(B),  A linear in B                  clas-tracking .../utilities/RungeKuttaDoca.java
    measurement = signed drift DOCA to the wire                    clas-tracking .../kalmanfilter/AMeasVecs.java
    q/p carried across material unchanged: fVec.Q = iVec.Q         zReference/StateVecs.java:74,172,272
    corrForEloss(...) → throws UnsupportedOperationException       zReference/StateVecs.java:387
final |p| at Region 3:  set_P(1/|finalStateVec.Q|)                 service/dc/DCTBEngine.java:313
vertex: negate p⃗ and q, swim R3 → beamline (|p| CONSTANT)          track/TrackCandListFinder.java (SetSwimParameters / SwimToBeamLine)
    SwimToBeamLine: _pTot never modified; DOCA to (xB,yB), z<2m    swim-tools/.../Swim.java (SwimToBeamLine, BeamLineSwimStopper)
EventBuilder copies Vtx0_*/p0_* verbatim into REC::Particle        clas-reco/.../DetectorData.java
```

**The drift DOCA the KF fits is `FittedHit.get_Doca()`**, set by `set_TimeToDistance` (below) and consumed as the KF residual `get_Doca()·LeftRightAmb − projectedDoca`. So **anything that biases the drift distance biases `q/p`, hence the momentum, hence `vz`.**

---

## 3. Mechanism 1 (PRIMARY, polarity-ODD, fixable): the DC Lorentz-angle correction

`reconstruction/dc/.../hit/FittedHit.java`, `set_TimeToDistance` (lines 364-414):

```java
double theta0 = Math.acos(1 - 0.02*B);                 // :372  Lorentz/isochrone-twist angle, grows with |B|
double alpha  = Math.atan(trkAngle);                   // :373  local track incidence angle
this.setAlpha(Math.toDegrees(alpha));
alpha -= Math.signum(Swimmer.getTorScale()) * theta0;  // :376  <-- polarity-signed correction
double ralpha = this.reducedAngle(alpha);              // :380
...
distance = tde.interpolateOnGrid(B, Math.toDegrees(ralpha), beta, this.get_Time(), secIdx, slIdx); // :408
this.set_Doca(distance);                               // :412
```

Facts:
- `B` here is the field **magnitude** (`setB(Math.sqrt(bx²+by²+bz²))`, `cross/CrossListFinder.java`), and the time→distance table is indexed on `|B|` too (`TimeToDistanceEstimator.java:90 double B = Math.abs(Bf)`). So the *magnitude* of the field effect is polarity-even.
- The **sign** of the whole correction is `Math.signum(Swimmer.getTorScale())` — the signed torus scale (`common-tools/swim-tools/.../Swimmer.java`, `TORSCALE`/`getTorScale`). Flip the torus → flip the applied correction.
- The magnitude model is a **single hard-coded constant** `0.02` (`FittedHit.java:372`, the only occurrence). No CCDB table, no per-superlayer / per-sector calibration. For `B≈2 T`, `theta0 ≈ 16°` — a large knob, not a rounding effect.

**Why an imperfect model gives a polarity-ODD momentum bias.** The applied correction is `C = sign(torus)·acos(1−0.02B)`. The true isochrone twist is also odd in the field, `L = sign(torus)·L₀(B)`. The bias is the residual

```
R = C − L = sign(torus) · [ acos(1−0.02B) − L₀(B) ].
```

`R ∝ sign(torus)` — **odd in polarity**. If `0.02` (and the functional form) matched reality, `R=0` at both polarities and there would be no shift. Because it is a crude one-parameter model, the residual is a **coherent** per-hit shift of the fitted coordinate (same sign in every superlayer the track crosses, so it does *not* average out), which biases the fitted sagitta → `q/p` → the momentum → `vz`, with a sign set by the torus. The `α`-dependence of the time→distance map (`cos(30°−α)`) makes the resulting drift-distance bias vary with the local incidence angle (correlated with `p` and `θ`), producing the `(c0 + c1/p)·cot θ` shape.

**This is the mechanism behind the polarity flip, and it is the calibration lever to fix.**

---

## 4. Mechanism 2 (secondary, polarity-ODD, physical — NOT a bug): the solenoid does not flip with the torus

The KF/​swim equations are exactly symmetric only under `B⃗ → −B⃗` **and** `q → −q` (`A` is linear in `B`, so `(q/p)·A(B)` is invariant under that joint flip). Flipping the **torus** polarity flips the torus field but **not** the solenoid (the solenoid keeps its sign in both torus settings). So a `+torus` reconstruction and a `−torus` reconstruction are **not** exact mirror images: the composite field the KF and the back-swim integrate through is not antisymmetric under a torus flip. This leaves a small, `p`/`θ`-dependent, polarity-odd residual (largest at low `p` / forward `θ`, where the solenoid fringe matters most).

This is **real physics, not a reconstruction error** — the reconstruction correctly uses the actual fields. It cannot and should not be "fixed" in code (the solenoid genuinely does not reverse). It is listed because it is a second, independent contributor to any polarity asymmetry and should be kept in mind when interpreting the residual after Mechanism 1 is calibrated. Expected to be sub-dominant to Mechanism 1 for the momentum determination.

---

## 5. Mechanism 3 (the polarity-EVEN floor — shapes the walk, does NOT flip)

These reproduce the `(c0 + c1/p)·cot θ` *shape* but carry the **same sign at both polarities**, so they are not the cause of the flip:

- **No energy-loss correction anywhere in DC tracking or the Event Builder.** The swim conserves `|p|` (`Swim.java`, `_pTot` never modified); the KF carries `q/p` across material unchanged and `corrForEloss` throws (`zReference/StateVecs.java:387`); the EB copies `Vtx0_*/p0_*` verbatim (`DetectorData.java`). So the momentum handed to the swim is the Region-3 momentum, already reduced by energy lost between the vertex and R3 and never restored. Swimming a too-low `|p|` over-bends the back-extrapolation → a `~1/p·cot θ` (`c1`) vertex bias. Polarity-even.
- **Rigid DC alignment** (`DCGeant4Factory.getAlignmentShift`, applied as a clean rigid translation/rotation of the wire endpoints) and **beam position** `(xB,yB)` → a `p`-flat transverse offset → `c0·cot θ`. Polarity-even.
- **T0 / event start time** additive terms (`HitReader.java:401,414-419`) → a roughly `p`-flat drift-distance offset. Polarity-even.
- **Suspect worth a look (p-dependent, not polarity):** in `HitReader.read_HBHits` the flight time *subtracted* to build the drift time is the HB `β≈1` value (`:414`), while the value *stored* on the hit is `β`-corrected (`:411 setTFlight(tFlight[i]/get_Beta0to1())`). For genuinely slow particles (low-`p` pions) the subtracted flight time is then too small → drift time too large → a `β(p)`-dependent DOCA bias. Relevant to the pion `vz` walk; polarity-even.

---

## 6. Decomposition summary

| Contribution | file:line | p-dependence | Flips with torus polarity? | Nature |
|---|---|---|---|---|
| **Lorentz-angle α-correction** `sign(torus)·acos(1−0.02B)` | **FittedHit.java:372-376** | via local angle α (→ p,θ) | **YES** | crude calibration → **fixable** |
| Solenoid not reversing with torus | KF/​swim use composite field | low-p / forward-θ | YES | real physics, not a bug |
| No energy-loss correction | StateVecs.java:387; Swim.java `_pTot` | `~c1/p` | No | missing feature (separate) |
| DC alignment / beam offset | DCGeant4Factory getAlignmentShift | p-flat (`c0`) | No | geometry |
| T0 / start time | HitReader.java:401,414-419 | ~p-flat | No | timing |
| tFlight β mismatch (HB β≈1 subtracted) | HitReader.java:411 vs 414 | p-dependent (low-p) | No | suspected inconsistency |

---

## 7. Why the earlier tests behaved as they did

- **Method B (full recon at 0.998×torus) reproduces the shift:** `signum(0.998·torus)` is unchanged and `0.02·B` moves by only 0.2%, so the Lorentz term (Mechanism 1) is essentially untouched; and the KF renormalizes the 0.998 into `q/p`. The scale never touches the term that causes the shift.
- **Method A (swim full-scale p through 0.998) flattens it:** holding `p` fixed and rescaling the swim field is a pure curvature tweak on the vertex, which cosmetically cancels the miss — it bypasses the momentum, which is where the bias lives. This is why it "works" in the swimmer but evaporates under real tracking.
- **`field_alignment.typ` field/geometry scan:** it minimizes the *swum* μ₂ spread with `p` external, so a field knob can flatten it; but the physical cause is a DC drift-distance calibration residual, so the correction does not survive a self-consistent re-reconstruction.

---

## 8. The fix (calibration knob) and how to use it

The correct magnitude of the Lorentz correction is **not known a priori** — it must be calibrated so that the reconstructed `vz(p,θ)` is flat **at both torus polarities simultaneously** (forcing the odd residual `R→0`). We therefore expose the correction strength as a tunable scale rather than hard-coding a guessed constant. See the code change in `FittedHit.java` (`LORENTZ_ANGLE_SCALE`, default `1.0` = unchanged production behavior), overridable at launch:

```
-Ddc.lorentz.scale=<x>      (JVM system property)     or     DC_LORENTZ_SCALE=<x>   (environment)
```

The applied correction becomes `alpha -= sign(torus) · LORENTZ_ANGLE_SCALE · acos(1−0.02·B)`:

| value | effect |
|---|---|
| `1.0` | production behavior (unchanged) |
| `0.0` | correction **OFF** — diagnostic: removes the polarity-odd drift bias entirely |
| `<0`  | flips the applied sign (test whether the current sign is backwards) |
| scan  | calibrate the value that flattens `vz(p,θ)` at both polarities |

**Recommended sequence:**
1. **Confirm causation first:** re-reconstruct one slice with `DC_LORENTZ_SCALE=0`. If the polarity-odd component of the `vz(p,θ)` walk collapses (leaving only the even floor), Mechanism 1 is confirmed as the driver.
2. **Calibrate:** scan `LORENTZ_ANGLE_SCALE` (e.g. 0, 0.5, 1.0, 1.5, 2.0; include a negative if step 1 suggests the sign) and pick the value that flattens `vz` across `(p,θ)` for **both** polarities. That value is the calibration result; it can then be promoted to a proper per-superlayer CCDB table.
3. The **even floor** (energy loss, alignment/T0) is a separate work item; it will remain after Mechanism 1 is fixed and is best addressed by adding a real dE/dx correction and revisiting alignment/beam offset — not by this knob.

**Caveats.** Values are not yet measured on data; this note localizes the mechanism and provides the lever. Scaling `theta0` (the applied angle) is a bounded, monotonic proxy for recalibrating the Lorentz model; a full fix replaces the one-parameter `acos(1−0.02B)` with a calibrated, field- and superlayer-dependent function.
