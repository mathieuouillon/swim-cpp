#set page(
  paper: "a4",
  margin: (x: 2.4cm, y: 2.6cm),
  numbering: "1",
)
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading: set block(above: 1.3em, below: 0.7em)
#set math.equation(numbering: "(1)")

// small unit helper
#let unit(x) = math.upright(x)

#align(center)[
  #text(17pt, weight: "bold")[
    Aligning the swum-#math.italic[v]#sub[z] target peaks \
    by scanning the magnetic field map
  ]
  #v(0.4em)
  #text(11pt)[Mathieu Ouillon]
  #v(0.2em)
  #text(9pt, style: "italic")[CLAS12 / RG-D — field-map calibration note]
]

#v(0.6em)

#block(
  inset: 10pt,
  radius: 4pt,
  fill: luma(245),
  width: 100%,
)[
  *Abstract.* Charged tracks are swum backward through the CLAS12 magnetic field to the
  beamline; the resulting vertex-#math.italic[z] distribution shows two peaks from the
  RG-D dual target. Because the physical foils sit at fixed #math.italic[z], the
  reconstructed second-peak mean must be independent of the track momentum $p$ and emission
  angle $theta$. We fit the second peak in each $(p, theta)$ bin and use the spread of its
  Gaussian mean over all $(p, theta)$ bins as a misalignment metric, scanning rigid
  displacements and scales of the field map to find the modification that makes the
  second-peak mean line up.
]

= Motivation

The RG-D run period uses a *dual target*: two thin foils placed at fixed, distinct
positions along the beam axis. Electrons (and other charged tracks) reconstructed in the
CLAS12 Forward Detector originate from one of these two foils, so the distribution of their
production vertex along the beam, $v_z$, is expected to show *two peaks* — an upstream
peak (peak 1) and a downstream peak (peak 2).

The key observation is geometric: the foils do not move. A track emitted at a small polar
angle $theta$ and one emitted at a large $theta$ come from the *same* foil and must
therefore reconstruct to the *same* $v_z$. Any residual dependence of a peak position on
$theta$ — in practice a slow drift that tracks $1 \/ tan theta$ — is not physics but a
symptom of an imperfect magnetic field map: a small rigid displacement or mis-scaling of
the solenoid or torus field relative to the true geometry.

This note describes how that misalignment is diagnosed and corrected. The *second peak* is
used as the primary handle: although it partly overlaps the first, its position is a clean,
sensitive probe of the field, and aligning it across $theta$ is the target of the
procedure.

= The swum-$v_z$ observable

For a charged track the field map enters through *swimming*: the track state measured in
the drift chambers is propagated backward through the field toward the beamline by
integrating the equation of motion along the path length $s$,

$ (dif bold(r)) / (dif s) = bold(u), wide
  (dif bold(u)) / (dif s) = K_0 q / p (bold(u) times bold(B)(bold(r))), $

where $bold(r)$ is the position, $bold(u)$ the unit momentum direction, $p$ the momentum,
$q$ the charge, $bold(B)$ the local field, and

$ K_0 = 0.299792458 times 10^(-3) wide unit("(kG·cm)")^(-1) $

converts the field–momentum ratio into a curvature. Positions are in #unit("cm"), the
field in #unit("kG"), and momentum in #unit("GeV") $\/ c$.

The swimmer is advanced until the track reaches its *distance of closest approach* (DOCA)
to the beamline, defined by the beam position $(x_b, y_b)$ rather than the geometric
$z$-axis. The $z$-coordinate at that point of closest approach is the *swum vertex*,
$v_z$. Repeating this for every track and binning in $v_z$ yields the spectrum analysed
below.

#figure(
  image("figs/swum_vz_spectrum.png", width: 80%),
  caption: [Swum-$v_z$ spectrum over $[-13, 0]$ #unit("cm"), integrated over the
  $(p, theta)$ grid for the best-aligned scan value. The two RG-D target windows are
  shaded; the Gaussian fit to the second peak (see @sec:fit) and its $mu$ are overlaid.],
) <fig:spectrum>

= The two target peaks

The swum-$v_z$ histogram spans roughly $[-13, 0]$ #unit("cm"). Each peak is located inside
a fixed search window:

- *Peak 1* (upstream foil): the more negative window, near $v_z approx -8$ #unit("cm").
- *Peak 2* (downstream foil): the less negative window, near $v_z approx -3$ #unit("cm").

The two peaks *overlap*: peak 2 sits on the downstream shoulder of peak 1, separated by a
shallow valley rather than a clean gap. This overlap is the main complication for fitting
the second peak and motivates the valley-aware procedure of @sec:fit.

= Field-map modifications scanned <sec:scan>

The diagnosis proceeds by deliberately perturbing the field map and re-swimming. Two kinds
of perturbation are applied before the field is interpolated: a *rigid translation* of a
map (it is shifted bodily in $x$, $y$, or $z$), and a *field scale* (the field values of a
map are multiplied by a constant). The scanned parameters and what each one physically
probes are summarised in @tab:params.

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, center, left),
    stroke: 0.4pt + luma(180),
    inset: 6pt,
    table.header(
      [*Parameter*], [*Type*], [*What it probes*],
    ),
    [solenoid $x$, $y$ shift], [translation], [transverse placement of the solenoid field],
    [solenoid $z$ shift], [translation], [weak lever arm on the peak positions],
    [torus $x$, $y$ shift], [translation], [primary diagnostic for the $1\/tan theta$ residual trend],
    [torus $z$ shift], [translation], [longitudinal placement of the torus field],
    [torus scale], [scale], [momentum-dependent ($1\/p$) bending strength],
    [solenoid scale], [scale], [solenoid bending strength],
    [DC $x$, $y$, $z$ shift], [geometry], [separates a geometric (DC) cause from a magnetic one],
  ),
  caption: [Scanned field-map (and detector-geometry) parameters. Shifts are in
  #unit("cm"); scales are dimensionless.],
) <tab:params>

Two points guide the interpretation. First, the *solenoid $z$-shift has almost no lever
arm* on the swum-$v_z$ peaks, so it is not the corrective handle. The sensitive parameter
for the angular drift is a *torus transverse ($x$/$y$) shift*, which is the diagnostic for
the $1\/tan theta$ residual. Second, scanning the *detector* position (the drift-chamber
start state) separates a purely *geometric* misalignment, whose effect is flat in momentum,
from a *magnetic* one, which scales with $1\/p$. This lets a true field-map error be
distinguished from a tracking-geometry error before any field modification is adopted.

= Fitting the second peak <sec:fit>

Because peak 2 overlaps peak 1, a naive Gaussian fit over a fixed window is biased by the
neighbouring peak. The fit is therefore restricted to a *valley-bounded* slice built in
four steps:

+ *Locate the peak.* The histogram is lightly smoothed (a moving average over a fraction of
  a #unit("cm")) and the maximum within the peak-2 window is taken. The smoothing is used
  *only to choose the slice* — never to fit.

+ *Take the top region.* Starting from the maximum, the contiguous set of bins lying above
  a fraction (about $60%$) of the peak height is selected. This threshold keeps the slice
  *above the inter-peak valley*, so it does not run down into peak 1.

+ *Clip at the valleys.* On each flank the slice is trimmed back to the first interior
  minimum of the smoothed shape. For an isolated peak the minima fall at the slice edges
  and this does nothing; when a shallow valley toward the neighbouring peak is present, the
  clip removes the bleed-in. This is the anti-overlap guard.

+ *Fit the raw counts.* A single Gaussian,

  $ f(v) = A exp(-1/2 ((v - mu) / sigma)^2), $

  is fit to the *raw* (unsmoothed) counts inside the slice. The fit returns the peak
  position $mu$ and width $sigma$ together with their $1 sigma$ uncertainties from the
  covariance matrix. The amplitude is bounded to be positive, the mean to lie inside the
  slice, and the width to lie between a small floor and the slice width, so the fit cannot
  rail onto a neighbouring structure.

On real, overlapping spectra this prescription is stable: the returned $mu$ tracks the true
peak maximum to better than a fraction of a millimetre and $sigma$ varies smoothly from one
scan point to the next, with no failed fits.

= Alignment criterion <sec:metric>

The swimming is done in bins of momentum $p$ (here $2$–$10$ #unit("GeV") $\/c$ in
$1$ #unit("GeV") $\/c$ steps) and polar angle $theta$ (here $theta$ in roughly
$[6, 26)degree$ in $2degree$ steps). For a fixed scan value, the second peak is fit
(@sec:fit) in *every $(p, theta)$ bin*, giving its Gaussian *mean* $mu_2 (p, theta)$.

The objective is to *align the mean of the second peak*: since the downstream foil sits at
a fixed $z$, $mu_2$ should be the same in every momentum and angle bin. A perfect field map
would give a $mu_2$ that is flat in both $p$ and $theta$. The misalignment is therefore
quantified by the *spread of the second-peak mean over all $(p, theta)$ bins*,

$ cal(M) = op("RMS")_(p, theta) [mu_2 (p, theta)], $

with lower $cal(M)$ meaning the second-peak mean is better aligned. Each shift is scanned
*independently*: for a given shift we fit $mu_2$ at every scan value and plot $cal(M)$
versus the value; its *minimum* identifies the best-aligning value of that shift.

The two dimensions carry complementary information. At fixed $p$, the *crossing* of the
$theta$-bin curves is the value that removes the $theta$-walk of $mu_2$. How that crossing
*drifts with $p$* separates the two causes of @sec:scan: a $p$-stable crossing is a
geometric (foil/DC) effect, whereas a crossing that moves with $p$ is the $1\/p$ magnetic
signature of a field scale or torus displacement.

= Results: every tested shift <sec:results>

Seven field-map modifications were scanned this way — the solenoid $x$, $y$, $z$ shifts,
the torus $x$, $y$, $z$ shifts, and the torus scale. @fig:metric-grid shows the alignment
metric $cal(M) = op("RMS")_(p, theta) [mu_2]$ versus value for all of them at a glance, and
@tab:best lists the best-aligning value of each. The full per-$(p, theta)$ behaviour of
$mu_2$ and $sigma_2$ for each shift then follows in @sec:catalog.

#figure(
  image("figs/metric_all.png", width: 100%),
  caption: [Alignment metric $cal(M) = op("RMS")_(p, theta) [mu_2]$ versus value for every
  tested shift. The marked minimum of each panel is the value that best aligns the
  second-peak mean over all $(p, theta)$ bins.],
) <fig:metric-grid>

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    stroke: 0.4pt + luma(180),
    inset: 6pt,
    ..csv("figs/best_values.csv").flatten().map(c => [#c]),
  ),
  caption: [Per shift: the scanned value that minimises $cal(M)$ and the resulting RMS of
  the second-peak mean over all $(p, theta)$ bins (shifts in #unit("cm"), torus scale
  dimensionless).],
) <tab:best>

== Per-$(p, theta)$ catalogue <sec:catalog>

For each shift the second-peak mean $mu_2$ and width $sigma_2$ are shown versus the scan
value, with *one panel per momentum bin* and *one curve per $theta$ bin*. In a $mu_2$ panel
the value where the $theta$-bin curves cross is the angle-aligned point; the dotted line
marks the global best value of @tab:best. Reading the crossing across the momentum panels
shows its $p$ dependence (@sec:metric).

#let shift-block(stub, name) = [
  === #name
  #figure(image("figs/" + stub + "_mu_pt.png", width: 100%),
    caption: [Second-peak mean $mu_2$ vs #name, per $(p, theta)$.])
  #figure(image("figs/" + stub + "_sigma_pt.png", width: 100%),
    caption: [Second-peak width $sigma_2$ vs #name, per $(p, theta)$.])
]

#shift-block("solenoid-x", [solenoid $x$-shift])
#shift-block("solenoid-y", [solenoid $y$-shift])
#shift-block("solenoid-z", [solenoid $z$-shift])
#shift-block("torus-x", [torus $x$-shift])
#shift-block("torus-y", [torus $y$-shift])
#shift-block("torus-z", [torus $z$-shift])
#shift-block("torus-scale", [torus scale])

= Workflow summary

The full procedure chains the pieces above:

+ choose a field parameter and a list of scan values (@sec:scan);
+ for each value, re-swim all tracks through the modified field map;
+ histogram swum-$v_z$ in each $(p, theta)$ bin;
+ fit the second peak in every $(p, theta)$ bin, recording its Gaussian mean
  $mu_2 (p, theta)$ (@sec:fit);
+ form the $mu_2$-versus-value curves (a panel per $p$, a curve per $theta$) and the
  alignment metric $cal(M) = op("RMS")_(p, theta) [mu_2]$ (@sec:metric);
+ select the scan value that minimises $cal(M)$;
+ repeat for every shift and collect the best value of each (@sec:results).

Per-shift diagnostics — the swum-$v_z$ spectrum with its fit (@fig:spectrum), the
per-$(p, theta)$ $mu_2$ and $sigma_2$ curves (@sec:catalog), and the metric grid
(@fig:metric-grid) — let the result be inspected by eye at each step.

= Summary

The physical RG-D foils are fixed in $z$, so a correct field map must reconstruct a
second-peak mean that is independent of *both* momentum and angle. The best field-map
modification — a rigid solenoid or torus displacement, or a field scale — is the one that
*minimises the spread of the second-peak mean $mu_2$ over all $(p, theta)$ bins*, the
primary calibration handle of this analysis.
