import uproot
import numpy as np
import matplotlib as mpl
mpl.use("Agg")  # non-interactive backend: this script only writes PDFs. Forcing Agg
                # avoids hanging on a dead X11/$DISPLAY forward on a headless farm node.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe

# ── Style ──────────────────────────────────────────────────────────────────
# Mirror of the reference style, but WITHOUT a real LaTeX backend: usetex spawns
# a latex subprocess per label and makes each figure take ~a minute. We use the
# non-Tex mplhep style + scienceplots "no-latex" so all `$...$` render via
# matplotlib's fast built-in mathtext instead. Set MPL_USETEX=1 to opt back in.
import os as _os

try:
    import mplhep as hep
    import scienceplots  # noqa: F401  (registers the "science" style)

    if _os.environ.get("MPL_USETEX") == "1":
        hep.style.use("CMSTex")
        plt.style.use(["science"])
        mpl.rc("text", usetex=True)
        mpl.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{lmodern}")
    else:
        hep.style.use("CMS")
        plt.style.use(["science", "no-latex"])
        mpl.rc("text", usetex=False)
except Exception as exc:  # pragma: no cover - environment dependent
    print(f"[warn] fancy style unavailable ({exc}); using matplotlib defaults")
    import mplhep as hep  # hep.histplot is still required below
    mpl.rc("text", usetex=False)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 12,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": False, "ytick.right": False,
})

# ── Layout of the analysis output ────────────────────────────────────────────
# Histograms are named vz_true{tag}_reco{tag}_p{lo}_{hi}, where {lo}/{hi} are the
# momentum-bin edges in units of 0.1 GeV, zero-padded to two digits (the
# `{:02d}` encoding of std::lround(edge * 10) from src/bank_access.hpp). Species
# order matches SPECIES_TAG: pi+, pi-, mu+, mu-.
SPECIES_TAG: tuple[str, ...] = ("pip", "pim", "mup", "mum")
SPECIES_TEX: dict[str, str] = {
    "pip": r"$\pi^+$", "pim": r"$\pi^-$",
    "mup": r"$\mu^+$", "mum": r"$\mu^-$",
}
P_EDGES: np.ndarray = np.arange(0.0, 6.0 + 1e-9, 0.5)  # 13 edges -> 12 bins
P_BINS: list[tuple[float, float]] = list(zip(P_EDGES[:-1], P_EDGES[1:]))

# tab10 + black + saddle-brown so adjacent lines stay readable for up to 12 bins.
_BIN_PALETTE: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#000000", "#8B4513",
]


def _bin_colors(n: int) -> list[str]:
    if n <= len(_BIN_PALETTE):
        return _BIN_PALETTE[:n]
    raise ValueError(f"_bin_colors only supports up to {len(_BIN_PALETTE)} bins (asked for {n})")


# Track variables — must match TRACK_VARS (order/tags) in src/constants.hpp.
# The analysis is Forward-Detector only, so there is no region split.
TRACK_VARS: tuple[tuple[str, str], ...] = (
    ("p", r"$p$ [GeV]"),
    ("pt", r"$p_T$ [GeV]"),
    ("theta", r"$\theta$ [deg]"),
    ("phi", r"$\phi$ [deg]"),
    ("vx", r"$v_x$ [cm]"),
    ("vy", r"$v_y$ [cm]"),
    ("vz", r"$v_z$ [cm]"),
    ("dbeta", r"$\Delta\beta$"),
    ("chi2pid", r"$\chi^2_{\rm PID}$"),
    ("vt", r"$v_t$ [ns]"),
    ("trkchi2", r"track $\chi^2$"),
    ("trkndf", r"track NDF"),
    ("trkredchi2", r"track $\chi^2/$NDF"),
    ("edge_dc1", r"DC R1 edge [cm]"),
    ("edge_dc2", r"DC R2 edge [cm]"),
    ("edge_dc3", r"DC R3 edge [cm]"),
    ("dc_path", r"DC path [cm]"),
    ("sigp", r"$\sigma_p/p$"),
    ("sigtx", r"$\sigma_{t_x}$"),
    ("sigty", r"$\sigma_{t_y}$"),
    ("sigtheta", r"$\sigma_\theta$ [mrad]"),
    ("sigphi", r"$\sigma_\phi$ [mrad]"),
    ("ftof_chi2", r"FTOF $\chi^2$"),
    ("ftof_dt", r"FTOF $\Delta t$ [ns]"),
    ("ftof_dedx", r"FTOF d$E$/d$x$ [MeV/cm]"),
    ("htcc_nphe", r"HTCC $n_{\rm phe}$"),
    ("ecal_e", r"ECAL $E$ [GeV]"),
    ("ecal_sf", r"$E/p$"),
)


def _pbin_key(prefix: str, truth: str, reco: str, p_lo: float, p_hi: float) -> str:
    return f"{prefix}_true{truth}_reco{reco}_p{round(p_lo * 10):02d}_{round(p_hi * 10):02d}"


def _vz_key(truth: str, reco: str, p_lo: float, p_hi: float) -> str:
    return _pbin_key("vz", truth, reco, p_lo, p_hi)


def _vztheta_key(truth: str, reco: str) -> str:
    return f"vztheta_true{truth}_reco{reco}"


# Momentum bands for the track-variable distributions — must match PT_BANDS in
# src/constants.hpp (tag, legend label, color).
PT_BANDS: tuple[tuple[str, str, str], ...] = (
    ("lo", r"$p<1$ GeV", "tab:blue"),
    ("hi", r"$3<p<6$ GeV", "tab:red"),
)


def _var_key(truth: str, reco: str, band: str, var: str) -> str:
    return f"var_true{truth}_reco{reco}_{band}_{var}"


def plot_confusion_matrix(file_path: str = "vz.root", out_path: str = "vz_confusion_matrix.pdf"):
    """4x4 PID confusion matrix (rows = MC truth, cols = reco), integrated over p.

    Left panel: raw fill counts (log color). Right panel: each row normalized to
    the truth total, i.e. P(reco | truth) — the mis-ID probabilities.
    """
    n = len(SPECIES_TAG)
    mat = np.zeros((n, n))
    with uproot.open(file_path) as f:
        for it, t in enumerate(SPECIES_TAG):
            for ir, r in enumerate(SPECIES_TAG):
                mat[it, ir] = sum(
                    float(f[_vz_key(t, r, lo, hi)].to_numpy()[0].sum()) for lo, hi in P_BINS
                )

    row = mat.sum(axis=1, keepdims=True)
    frac = np.divide(mat, row, out=np.zeros_like(mat), where=row > 0)
    tex = [SPECIES_TEX[s] for s in SPECIES_TAG]

    # interpolation_stage="rgba" applies the colormap/norm to the 4x4 data BEFORE
    # resampling (not to the upsampled image), and "nearest" keeps crisp cells.
    # Without this, LogNorm's log transform runs on the huge resampled array and
    # savefig hangs on a headless render.
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    im0 = axs[0].imshow(np.where(mat > 0, mat, np.nan), cmap="viridis", norm=LogNorm(),
                        aspect="auto", interpolation="nearest", interpolation_stage="rgba")
    axs[0].set_title("Counts")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(frac, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto",
                        interpolation="nearest", interpolation_stage="rgba")
    axs[1].set_title(r"Row-normalized $P(\mathrm{reco}\,|\,\mathrm{truth})$")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    for ax, data, fmt in ((axs[0], mat, "{:.0f}"), (axs[1], frac, "{:.2g}")):
        ax.set_xticks(range(n), labels=tex)
        ax.set_yticks(range(n), labels=tex)
        ax.set_xlabel("reconstructed")
        ax.set_ylabel("MC truth")
        for it in range(n):
            for ir in range(n):
                if data[it, ir] > 0:
                    txt = ax.text(ir, it, fmt.format(data[it, ir]),
                                  ha="center", va="center", fontsize=8, color="white")
                    txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="black")])

    fig.suptitle(r"PID confusion matrix (truth $\times$ reco), integrated $v_z$ counts")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_vz_matrix_integrated(
    file_path: str = "vz.root",
    density: bool = True,
    out_path: str = "vz_matrix_integrated.pdf",
):
    """4x4 grid (rows = truth, cols = reco): v_z summed over all momentum bins."""
    n = len(SPECIES_TAG)
    fig, axs = plt.subplots(n, n, figsize=(3.3 * n, 3.0 * n), squeeze=False, sharex=True)
    with uproot.open(file_path) as f:
        for it, t in enumerate(SPECIES_TAG):
            for ir, r in enumerate(SPECIES_TAG):
                ax = axs[it][ir]
                total, edges = None, None
                for lo, hi in P_BINS:
                    vals, edges = f[_vz_key(t, r, lo, hi)].to_numpy()
                    total = vals if total is None else total + vals
                n_entries = float(total.sum())
                if n_entries > 0:
                    hep.histplot((total, edges), ax=ax, histtype="step", color="tab:blue", density=density)
                ax.text(0.04, 0.93, f"N={n_entries:.0f}", transform=ax.transAxes, fontsize=7, va="top")
                if it == 0:
                    ax.set_title(f"reco {SPECIES_TEX[r]}")
                if ir == 0:
                    ax.set_ylabel(f"truth {SPECIES_TEX[t]}\n" + ("PDF" if density else "Counts"))
                if it == n - 1:
                    ax.set_xlabel(r"$v_z$ [cm]")
    fig.suptitle(r"$v_z$ truth $\times$ reco (integrated over $p$)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_vz_matrix_p(
    file_path: str = "vz.root",
    density: bool = True,
    out_path: str = "vz_matrix_p.pdf",
):
    """4x4 grid (rows = truth, cols = reco): overlay v_z per 0.5 GeV momentum bin.

    Empty (truth, reco, p) cells are skipped. One shared legend for the p bins.
    """
    n = len(SPECIES_TAG)
    colors = _bin_colors(len(P_BINS))
    fig, axs = plt.subplots(n, n, figsize=(3.5 * n, 3.0 * n), squeeze=False, sharex=True)
    with uproot.open(file_path) as f:
        for it, t in enumerate(SPECIES_TAG):
            for ir, r in enumerate(SPECIES_TAG):
                ax = axs[it][ir]
                for (lo, hi), color in zip(P_BINS, colors):
                    vals, edges = f[_vz_key(t, r, lo, hi)].to_numpy()
                    if vals.sum() == 0:
                        continue
                    hep.histplot((vals, edges), ax=ax, histtype="step", color=color, density=density)
                if it == 0:
                    ax.set_title(f"reco {SPECIES_TEX[r]}")
                if ir == 0:
                    ax.set_ylabel(f"truth {SPECIES_TEX[t]}\n" + ("PDF" if density else "Counts"))
                if it == n - 1:
                    ax.set_xlabel(r"$v_z$ [cm]")

    handles = [
        Line2D([0], [0], color=c, label=rf"${lo:.1f}\!-\!{hi:.1f}$")
        for (lo, hi), c in zip(P_BINS, colors)
    ]
    fig.legend(handles=handles, title=r"$p$ [GeV]", loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.suptitle(r"$v_z$ truth $\times$ reco, per momentum bin")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_truth_vs_reco(file_path: str = "vz.root", density: bool = True,
                       out_path: str = "vz_truth_vs_reco.pdf"):
    """4x4 grid: integrated reco v_z (solid) vs MC-truth v_z (dashed) per cell.

    Single-peaked truth + structured reco ⇒ the shift/double-peak is a
    reconstruction effect, not target geometry (expected for a thin foil).
    """
    n = len(SPECIES_TAG)
    fig, axs = plt.subplots(n, n, figsize=(3.3 * n, 3.0 * n), squeeze=False, sharex=True)
    with uproot.open(file_path) as f:
        for it, t in enumerate(SPECIES_TAG):
            for ir, r in enumerate(SPECIES_TAG):
                ax = axs[it][ir]
                reco_tot, truth_tot, edges = None, None, None
                for lo, hi in P_BINS:
                    rv, edges = f[_pbin_key("vz", t, r, lo, hi)].to_numpy()
                    tv, _ = f[_pbin_key("vztrue", t, r, lo, hi)].to_numpy()
                    reco_tot = rv if reco_tot is None else reco_tot + rv
                    truth_tot = tv if truth_tot is None else truth_tot + tv
                if reco_tot.sum() > 0:
                    hep.histplot((reco_tot, edges), ax=ax, histtype="step",
                                 color="tab:blue", density=density, label="reco")
                if truth_tot.sum() > 0:
                    hep.histplot((truth_tot, edges), ax=ax, histtype="step", color="black",
                                 linestyle="--", density=density, label="truth")
                if it == 0:
                    ax.set_title(f"reco {SPECIES_TEX[r]}")
                if ir == 0:
                    ax.set_ylabel(f"truth {SPECIES_TEX[t]}\n" + ("PDF" if density else "Counts"))
                if it == n - 1:
                    ax.set_xlabel(r"$v_z$ [cm]")
                if it == 0 and ir == n - 1:
                    ax.legend(fontsize=7)
    fig.suptitle(r"reco vs MC-truth $v_z$ (integrated over $p$)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_dvz_mean_vs_p(file_path: str = "vz.root", out_path: str = "dvz_mean_vs_p.pdf"):
    """⟨Δv_z⟩ ± RMS vs momentum on the diagonal cells (Δv_z = reco − truth).

    A monotonic drift (typically negative) as p→0 is the energy-loss signature;
    a growing RMS is multiple-scattering / resolution.
    """
    pmid = 0.5 * (P_EDGES[:-1] + P_EDGES[1:])
    colors = _bin_colors(len(SPECIES_TAG))
    fig, ax = plt.subplots(figsize=(7, 5))
    with uproot.open(file_path) as f:
        for s, color in zip(SPECIES_TAG, colors):
            means = np.full(len(P_BINS), np.nan)
            rms = np.full(len(P_BINS), np.nan)
            for i, (lo, hi) in enumerate(P_BINS):
                vals, edges = f[_pbin_key("dvz", s, s, lo, hi)].to_numpy()
                tot = vals.sum()
                if tot <= 0:
                    continue
                centers = 0.5 * (edges[:-1] + edges[1:])
                mean = float(np.sum(centers * vals) / tot)
                means[i] = mean
                rms[i] = float(np.sqrt(max(np.sum((centers - mean) ** 2 * vals) / tot, 0.0)))
            ax.errorbar(pmid, means, yerr=rms, marker="o", ms=4, lw=1.2,
                        color=color, capsize=2, label=SPECIES_TEX[s])
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"$p$ [GeV]")
    ax.set_ylabel(r"$\langle \Delta v_z \rangle$ [cm]  (bars = RMS)")
    ax.set_title(r"$\Delta v_z = v_z^{\rm reco} - v_z^{\rm truth}$ vs $p$ (correctly-ID'd)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_vztheta(file_path: str = "vz.root", out_path: str = "vz_theta_2d.pdf"):
    """4x4 grid of v_z (x) vs θ (y) 2D heatmaps per (truth, reco)."""
    n = len(SPECIES_TAG)
    fig, axs = plt.subplots(n, n, figsize=(3.3 * n, 3.0 * n), squeeze=False,
                            sharex=True, sharey=True)
    with uproot.open(file_path) as f:
        for it, t in enumerate(SPECIES_TAG):
            for ir, r in enumerate(SPECIES_TAG):
                ax = axs[it][ir]
                vals, xe, ye = f[_vztheta_key(t, r)].to_numpy()  # x = v_z, y = theta
                if vals.sum() > 0:
                    ax.pcolormesh(xe, ye, np.ma.masked_where(vals.T <= 0, vals.T),
                                  cmap="viridis", norm=LogNorm())
                if it == 0:
                    ax.set_title(f"reco {SPECIES_TEX[r]}")
                if ir == 0:
                    ax.set_ylabel(f"truth {SPECIES_TEX[t]}\n" + r"$\theta$ [deg]")
                if it == n - 1:
                    ax.set_xlabel(r"$v_z$ [cm]")
    fig.suptitle(r"$v_z$ vs $\theta$ (truth $\times$ reco)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_ptrue_vtx_dcr1(file_path: str = "vz.root", out_path: str = "ptrue_vtx_dcr1.pdf"):
    """Per reco pion: 2D heatmap of true momentum at the vertex (x) vs at DC
    region 1 (y). The dashed diagonal is no-loss; deviation below it is the
    true energy loss between the vertex and the drift chambers."""
    pions = ("pip", "pim")
    fig, axs = plt.subplots(1, len(pions), figsize=(6 * len(pions), 5), squeeze=False)
    axs = axs[0]
    with uproot.open(file_path) as f:
        for ax, s in zip(axs, pions):
            vals, xe, ye = f[f"ptrue_vtx_dcr1_reco{s}"].to_numpy()
            if vals.sum() > 0:
                im = ax.pcolormesh(xe, ye, np.ma.masked_where(vals.T <= 0, vals.T),
                                   cmap="viridis", norm=LogNorm())
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.plot([0, 6], [0, 6], color="red", lw=0.8, ls="--")
            ax.set_title(f"reco {SPECIES_TEX[s]}")
            ax.set_xlabel(r"$p_{\rm vtx}^{\rm true}$ [GeV]")
            ax.set_ylabel(r"$p_{\rm DC1}^{\rm true}$ [GeV]")
    fig.suptitle(r"True momentum: vertex vs DC region 1")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_dp_vtx_dcr1(file_path: str = "vz.root", density: bool = True,
                     out_path: str = "dp_vtx_dcr1.pdf", dprange=None):
    """1D true momentum loss Delta p = p_vtx - p_DC1, reco pi+ and pi- overlaid.
    `dprange=(lo, hi)` zooms the x-axis (the histogram is unchanged)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    with uproot.open(file_path) as f:
        for s, color in (("pip", "tab:blue"), ("pim", "tab:red")):
            vals, edges = f[f"dp_vtx_dcr1_reco{s}"].to_numpy()
            if vals.sum() > 0:
                hep.histplot((vals, edges), ax=ax, histtype="step",
                             color=color, density=density, label=SPECIES_TEX[s])
    ax.axvline(0.0, color="gray", lw=0.8, ls=":")
    if dprange is not None:
        ax.set_xlim(*dprange)
    ax.set_xlabel(r"$\Delta p = p_{\rm vtx}^{\rm true} - p_{\rm DC1}^{\rm true}$ [GeV]")
    ax.set_ylabel("PDF" if density else "Counts")
    ax.set_title(r"True momentum loss, vertex $\to$ DC region 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_dp_vs_p(file_path: str = "vz.root", out_path: str = "dp_vs_p.pdf", dprange=None):
    """Per reco pion: 2D heatmap of true Delta p (y) vs true vertex momentum (x).
    `dprange=(lo, hi)` zooms the y-axis (the histogram is unchanged)."""
    pions = ("pip", "pim")
    fig, axs = plt.subplots(1, len(pions), figsize=(6 * len(pions), 5), squeeze=False)
    axs = axs[0]
    with uproot.open(file_path) as f:
        for ax, s in zip(axs, pions):
            vals, xe, ye = f[f"dp_vs_p_reco{s}"].to_numpy()
            if vals.sum() > 0:
                im = ax.pcolormesh(xe, ye, np.ma.masked_where(vals.T <= 0, vals.T),
                                   cmap="viridis", norm=LogNorm())
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axhline(0.0, color="red", lw=0.8, ls="--")
            if dprange is not None:
                ax.set_ylim(*dprange)
            ax.set_title(f"reco {SPECIES_TEX[s]}")
            ax.set_xlabel(r"$p_{\rm vtx}^{\rm true}$ [GeV]")
            ax.set_ylabel(r"$\Delta p$ [GeV]")
    fig.suptitle(r"True momentum loss vs vertex momentum")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


LUND_DIR = "/volatile/clas12/ouillon/lund_pion"


def _lund_momenta(lund_dir, file_glob, target_pid):
    """|p| (GeV) of particles with pid==target_pid in the LUND files matching
    `file_glob` under `lund_dir`.

    These are single-particle gun files: every event is a 10-number header line
    followed by one 14-number particle line = 24 numbers. We parse the whole
    file's numbers at once and reshape to (-1, 24) — fully vectorized, no
    per-line Python loop (the line loop was the multi-minute bottleneck). Within
    a 24-block the particle fields are at offsets 10+{3=pid, 6=px, 7=py, 8=pz}.
    Falls back to a robust per-line parse if the 24/event assumption fails."""
    from pathlib import Path
    chunks = []
    for path in sorted(Path(lund_dir).glob(file_glob)):
        text = path.read_text()
        nums = None
        try:
            nums = np.array(text.split(), dtype=np.float64)
        except ValueError:
            nums = None
        if nums is not None and nums.size > 0 and nums.size % 24 == 0:
            ev = nums.reshape(-1, 24)
            mask = ev[:, 13].astype(np.int64) == target_pid  # particle pid
            if mask.any():
                px, py, pz = ev[mask, 16], ev[mask, 17], ev[mask, 18]
                chunks.append(np.sqrt(px * px + py * py + pz * pz))
            continue
        # Fallback: ragged / multi-particle files.
        ps = []
        for line in text.splitlines():
            tok = line.split()
            if len(tok) < 14:
                continue
            try:
                if int(float(tok[3])) != target_pid:
                    continue
                px, py, pz = float(tok[6]), float(tok[7]), float(tok[8])
            except ValueError:
                continue
            ps.append((px * px + py * py + pz * pz) ** 0.5)
        if ps:
            chunks.append(np.asarray(ps))
    return np.concatenate(chunks) if chunks else np.empty(0)


def plot_gen_vs_reco_p(file_path: str = "vz.root", out_path: str = "gen_vs_reco_p.pdf",
                       lund_dir: str = LUND_DIR):
    """Per pion charge, three spectra (log-y): LUND-thrown momentum (from the
    generator .lund files, no detector at all), generated MC::Particle (in the
    HIPO, no reco cut), and reconstructed-FD-pion true vertex momentum
    (x-projection of the ptrue 2D). Shows the generator input vs what survives
    to the HIPO vs what is reconstructed."""
    from pathlib import Path
    pions = ("pip", "pim")
    lund_files = {"pip": ("pi_plus__*.lund", 211), "pim": ("pi_minus__*.lund", -211)}
    have_lund = Path(lund_dir).is_dir()
    if not have_lund:
        print(f"[warn] LUND dir not found ({lund_dir}); skipping the thrown curve")
    fig, axs = plt.subplots(1, len(pions), figsize=(6 * len(pions), 5), squeeze=False)
    axs = axs[0]
    with uproot.open(file_path) as f:
        for ax, s in zip(axs, pions):
            gvals, gedges = f[f"genp_{s}"].to_numpy()
            pv, xe, _ye = f[f"ptrue_vtx_dcr1_reco{s}"].to_numpy()
            rvals = pv.sum(axis=1)  # sum over p_DC1 -> reconstructed p_vtx spectrum
            if have_lund:
                fg, tpid = lund_files[s]
                lp = _lund_momenta(lund_dir, fg, tpid)
                if lp.size:
                    lvals, _ = np.histogram(lp, bins=gedges)
                    hep.histplot((lvals, gedges), ax=ax, histtype="step",
                                 color="tab:green", label="LUND (thrown)")
            if gvals.sum() > 0:
                hep.histplot((gvals, gedges), ax=ax, histtype="step",
                             color="tab:gray", label="generated (MC::Particle)")
            if rvals.sum() > 0:
                hep.histplot((rvals, xe), ax=ax, histtype="step",
                             color="tab:blue", label=r"reconstructed (FD $\pi$)")
            ax.set_yscale("log")
            ax.set_title(SPECIES_TEX[s])
            ax.set_xlabel(r"$p$ [GeV]")
            ax.set_ylabel("Counts")
            ax.legend()
    fig.suptitle(r"LUND-thrown vs generated vs reconstructed (FD $\pi$) momentum")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_dpf_vtx_dcr1(file_path: str = "vz.root", density: bool = True,
                      out_path: str = "dpf_vtx_dcr1.pdf", frange=None):
    """1D fractional true momentum loss Delta p / p_vtx, reco pi+ and pi- overlaid.
    `frange=(lo, hi)` zooms the x-axis (the histogram is unchanged)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    with uproot.open(file_path) as f:
        for s, color in (("pip", "tab:blue"), ("pim", "tab:red")):
            vals, edges = f[f"dpf_vtx_dcr1_reco{s}"].to_numpy()
            if vals.sum() > 0:
                hep.histplot((vals, edges), ax=ax, histtype="step",
                             color=color, density=density, label=SPECIES_TEX[s])
    ax.axvline(0.0, color="gray", lw=0.8, ls=":")
    if frange is not None:
        ax.set_xlim(*frange)
    ax.set_xlabel(r"$\Delta p / p_{\rm vtx}^{\rm true}$")
    ax.set_ylabel("PDF" if density else "Counts")
    ax.set_title(r"True fractional momentum loss, vertex $\to$ DC region 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_dpf_vs_p(file_path: str = "vz.root", out_path: str = "dpf_vs_p.pdf", frange=None):
    """Per reco pion: 2D heatmap of fractional true Delta p/p_vtx (y) vs p_vtx (x).
    `frange=(lo, hi)` zooms the y-axis (the histogram is unchanged)."""
    pions = ("pip", "pim")
    fig, axs = plt.subplots(1, len(pions), figsize=(6 * len(pions), 5), squeeze=False)
    axs = axs[0]
    with uproot.open(file_path) as f:
        for ax, s in zip(axs, pions):
            vals, xe, ye = f[f"dpf_vs_p_reco{s}"].to_numpy()
            if vals.sum() > 0:
                im = ax.pcolormesh(xe, ye, np.ma.masked_where(vals.T <= 0, vals.T),
                                   cmap="viridis", norm=LogNorm())
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axhline(0.0, color="red", lw=0.8, ls="--")
            if frange is not None:
                ax.set_ylim(*frange)
            ax.set_title(f"reco {SPECIES_TEX[s]}")
            ax.set_xlabel(r"$p_{\rm vtx}^{\rm true}$ [GeV]")
            ax.set_ylabel(r"$\Delta p / p_{\rm vtx}^{\rm true}$")
    fig.suptitle(r"True fractional momentum loss vs vertex momentum")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_swum_minus_rec(file_path: str = "vz.root", density: bool = True,
                        out_path: str = "swum_minus_rec.pdf"):
    """1D swum vz - reco vz, reco pi+/pi- overlaid (the requested comparison)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    with uproot.open(file_path) as f:
        for s, color in (("pip", "tab:blue"), ("pim", "tab:red")):
            vals, edges = f[f"swum_minus_rec_reco{s}"].to_numpy()
            if vals.sum() > 0:
                hep.histplot((vals, edges), ax=ax, histtype="step",
                             color=color, density=density, label=SPECIES_TEX[s])
    ax.axvline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"$v_z^{\rm swum} - v_z^{\rm rec}$ [cm]")
    ax.set_ylabel("PDF" if density else "Counts")
    ax.set_title(r"Swum vs reconstructed $v_z$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_swum_rec_vs_p(file_path: str = "vz.root", out_path: str = "swum_rec_vs_p.pdf"):
    """Per reco pion: 2D (swum vz - reco vz) vs momentum."""
    pions = ("pip", "pim")
    fig, axs = plt.subplots(1, len(pions), figsize=(6 * len(pions), 5), squeeze=False)
    axs = axs[0]
    with uproot.open(file_path) as f:
        for ax, s in zip(axs, pions):
            vals, xe, ye = f[f"swum_minus_rec_vs_p_reco{s}"].to_numpy()
            if vals.sum() > 0:
                im = ax.pcolormesh(xe, ye, np.ma.masked_where(vals.T <= 0, vals.T),
                                   cmap="viridis", norm=LogNorm())
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axhline(0.0, color="red", lw=0.8, ls="--")
            ax.set_title(f"reco {SPECIES_TEX[s]}")
            ax.set_xlabel(r"$p$ [GeV]")
            ax.set_ylabel(r"$v_z^{\rm swum} - v_z^{\rm rec}$ [cm]")
    fig.suptitle(r"Swum$-$reco $v_z$ vs momentum")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_rec_vs_swum(file_path: str = "vz.root", out_path: str = "rec_vs_swum.pdf"):
    """Per reco pion: 2D reco vz vs swum vz, dashed y=x."""
    pions = ("pip", "pim")
    fig, axs = plt.subplots(1, len(pions), figsize=(6 * len(pions), 5), squeeze=False)
    axs = axs[0]
    with uproot.open(file_path) as f:
        for ax, s in zip(axs, pions):
            vals, xe, ye = f[f"rec_vs_swum_reco{s}"].to_numpy()
            if vals.sum() > 0:
                im = ax.pcolormesh(xe, ye, np.ma.masked_where(vals.T <= 0, vals.T),
                                   cmap="viridis", norm=LogNorm())
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.plot([xe[0], xe[-1]], [xe[0], xe[-1]], color="red", lw=0.8, ls="--")
            ax.set_title(f"reco {SPECIES_TEX[s]}")
            ax.set_xlabel(r"$v_z^{\rm rec}$ [cm]")
            ax.set_ylabel(r"$v_z^{\rm swum}$ [cm]")
    fig.suptitle(r"reco vs swum $v_z$")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_swim_resolution(file_path: str = "vz.root", density: bool = True,
                         out_path: str = "swim_resolution.pdf"):
    """Headline 'does swimming help?': swum-true (blue) vs reco-true (red, the
    diagonal dvz summed over p bins) per reco pion. Narrower blue = swim wins."""
    pions = ("pip", "pim")
    fig, axs = plt.subplots(1, len(pions), figsize=(6 * len(pions), 5), squeeze=False)
    axs = axs[0]
    with uproot.open(file_path) as f:
        for ax, s in zip(axs, pions):
            sv, se = f[f"swum_minus_true_reco{s}"].to_numpy()
            if sv.sum() > 0:
                hep.histplot((sv, se), ax=ax, histtype="step", color="tab:blue",
                             density=density, label="swum - true")
            tot, edges = None, None
            for lo, hi in P_BINS:  # reco-true on the diagonal truth==reco==s
                dv, edges = f[_pbin_key("dvz", s, s, lo, hi)].to_numpy()
                tot = dv if tot is None else tot + dv
            if tot is not None and tot.sum() > 0:
                hep.histplot((tot, edges), ax=ax, histtype="step", color="tab:red",
                             density=density, label="reco - true")
            ax.axvline(0.0, color="gray", lw=0.8, ls=":")
            ax.set_title(f"reco {SPECIES_TEX[s]}")
            ax.set_xlabel(r"$\Delta v_z$ [cm]")
            ax.set_ylabel("PDF" if density else "Counts")
            ax.legend()
    fig.suptitle(r"Vertex resolution: swum$-$true vs reco$-$true")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_vz_reco_swim_p(file_path: str = "vz.root", density: bool = True,
                        out_path: str = "vz_reco_swim_p.pdf"):
    """3x2: reco v_z, true-state swum v_z, and reco-state swum v_z (rows) for reco
    pi+ (left) and pi- (right), each panel overlaying the 12 momentum bins (same
    colors as vz_matrix_p)."""
    pions = ("pip", "pim")
    colors = _bin_colors(len(P_BINS))
    rows = (("recvz", "reco"), ("swumvz", "swum (true)"), ("recoswumvz", "swum (reco)"))
    fig, axs = plt.subplots(len(rows), len(pions), figsize=(6 * len(pions), 4.5 * len(rows)),
                            squeeze=False, sharex=True)
    with uproot.open(file_path) as f:
        for col, s in enumerate(pions):
            for row, (prefix, label) in enumerate(rows):
                ax = axs[row][col]
                for (lo, hi), c in zip(P_BINS, colors):
                    key = f"{prefix}_reco{s}_p{round(lo * 10):02d}_{round(hi * 10):02d}"
                    vals, edges = f[key].to_numpy()
                    if vals.sum() > 0:
                        hep.histplot((vals, edges), ax=ax, histtype="step",
                                     color=c, density=density)
                ax.set_title(rf"{label} $v_z$, {SPECIES_TEX[s]}")
                if row == len(rows) - 1:
                    ax.set_xlabel(r"$v_z$ [cm]")
                if col == 0:
                    ax.set_ylabel("PDF" if density else "Counts")
    handles = [
        Line2D([0], [0], color=c, label=rf"${lo:.1f}\!-\!{hi:.1f}$")
        for (lo, hi), c in zip(P_BINS, colors)
    ]
    fig.legend(handles=handles, title=r"$p$ [GeV]", loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.suptitle(r"reco vs swum $v_z$ per momentum bin")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_recoswum_minus_rec(file_path: str = "vz.root", density: bool = True,
                            out_path: str = "recoswum_minus_rec.pdf"):
    """Per reco pion: swum v_z - reco v_z for the true-state swim (blue) and the
    reco-state swim (red, REC::Traj DC-R1 trajectory). The reco-state swim should
    peak sharply at 0 — it swims the same reconstructed trajectory the
    EventBuilder used to get the vertex (a closure check of our swimmer)."""
    pions = ("pip", "pim")
    fig, axs = plt.subplots(1, len(pions), figsize=(6 * len(pions), 5), squeeze=False)
    axs = axs[0]
    with uproot.open(file_path) as f:
        for ax, s in zip(axs, pions):
            for prefix, color, lab in (("swum_minus_rec", "tab:blue", "true-state swim"),
                                       ("recoswum_minus_rec", "tab:red", "reco-state swim")):
                vals, edges = f[f"{prefix}_reco{s}"].to_numpy()
                if vals.sum() > 0:
                    hep.histplot((vals, edges), ax=ax, histtype="step",
                                 color=color, density=density, label=lab)
            ax.axvline(0.0, color="gray", lw=0.8, ls=":")
            ax.set_title(f"reco {SPECIES_TEX[s]}")
            ax.set_xlabel(r"$v_z^{\rm swum} - v_z^{\rm rec}$ [cm]")
            ax.set_ylabel("PDF" if density else "Counts")
            ax.legend()
    fig.suptitle(r"Swum $-$ reconstructed $v_z$: true-state vs reco-state swim")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_pr1_reco_vs_true(file_path: str = "vz.root", out_path: str = "pr1_reco_vs_true.pdf"):
    """Per reco pion: 2D reconstructed |p| (REC::Particle) vs true p at DC region 1
    (MC::True), dashed y=x. Deviation from the diagonal = momentum reconstruction
    bias/resolution + the vertex->R1 energy loss."""
    pions = ("pip", "pim")
    fig, axs = plt.subplots(1, len(pions), figsize=(6 * len(pions), 5), squeeze=False)
    axs = axs[0]
    with uproot.open(file_path) as f:
        for ax, s in zip(axs, pions):
            vals, xe, ye = f[f"pr1_reco_vs_true_reco{s}"].to_numpy()
            if vals.sum() > 0:
                im = ax.pcolormesh(xe, ye, np.ma.masked_where(vals.T <= 0, vals.T),
                                   cmap="viridis", norm=LogNorm())
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.plot([xe[0], xe[-1]], [xe[0], xe[-1]], color="red", lw=0.8, ls="--")
            ax.set_title(f"reco {SPECIES_TEX[s]}")
            ax.set_xlabel(r"$p_{\rm DC1}^{\rm true}$ [GeV]")
            ax.set_ylabel(r"$p^{\rm rec}$ [GeV]")
    fig.suptitle(r"Reconstructed $|p|$ vs true $p$ at DC region 1")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_pr1_reco_minus_true(file_path: str = "vz.root", density: bool = True,
                             out_path: str = "pr1_reco_minus_true.pdf"):
    """1D reco |p| - true p at DC R1, reco pi+/pi- overlaid."""
    fig, ax = plt.subplots(figsize=(7, 5))
    with uproot.open(file_path) as f:
        for s, color in (("pip", "tab:blue"), ("pim", "tab:red")):
            vals, edges = f[f"pr1_reco_minus_true_reco{s}"].to_numpy()
            if vals.sum() > 0:
                hep.histplot((vals, edges), ax=ax, histtype="step",
                             color=color, density=density, label=SPECIES_TEX[s])
    ax.axvline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"$p^{\rm rec} - p_{\rm DC1}^{\rm true}$ [GeV]")
    ax.set_ylabel("PDF" if density else "Counts")
    ax.set_title(r"Reconstructed $|p|$ $-$ true $p$ at DC region 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_res_matrix_p(file_path: str, prefix: str, xlabel: str,
                      density: bool = True, out_path=None):
    """4x4 grid (rows = truth, cols = reco): overlay a resolution variable per
    0.5 GeV momentum bin (same layout/colors as plot_vz_matrix_p). `prefix` is
    the histogram-name stem (sigtx / sigty / sigtheta)."""
    out_path = out_path or f"{prefix}_matrix_p.pdf"
    n = len(SPECIES_TAG)
    colors = _bin_colors(len(P_BINS))
    fig, axs = plt.subplots(n, n, figsize=(3.5 * n, 3.0 * n), squeeze=False, sharex=True)
    with uproot.open(file_path) as f:
        for it, t in enumerate(SPECIES_TAG):
            for ir, r in enumerate(SPECIES_TAG):
                ax = axs[it][ir]
                for (lo, hi), color in zip(P_BINS, colors):
                    vals, edges = f[_pbin_key(prefix, t, r, lo, hi)].to_numpy()
                    if vals.sum() == 0:
                        continue
                    hep.histplot((vals, edges), ax=ax, histtype="step", color=color, density=density)
                if it == 0:
                    ax.set_title(f"reco {SPECIES_TEX[r]}")
                if ir == 0:
                    ax.set_ylabel(f"truth {SPECIES_TEX[t]}\n" + ("PDF" if density else "Counts"))
                if it == n - 1:
                    ax.set_xlabel(xlabel)

    handles = [
        Line2D([0], [0], color=c, label=rf"${lo:.1f}\!-\!{hi:.1f}$")
        for (lo, hi), c in zip(P_BINS, colors)
    ]
    fig.legend(handles=handles, title=r"$p$ [GeV]", loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.suptitle(rf"{xlabel} truth $\times$ reco, per momentum bin")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_vars(file_path: str = "vz.root", density: bool = True,
              out_path: str = "track_vars.pdf"):
    """Big combined grid of 1D track-variable distributions: rows = variable,
    columns = (truth, reco-pion) cells (confusion-matrix style, reco limited to
    pions). Each panel overlays the low-p (p<1) and high-p (3<p<6) bands,
    density-normalized so the shapes compare.
    """
    pions = ("pip", "pim")
    cells = [(t, r) for r in pions for t in SPECIES_TAG]  # grouped by reco
    nvar = len(TRACK_VARS)
    ncol = len(cells)
    fig, axs = plt.subplots(nvar, ncol, figsize=(2.4 * ncol, 1.8 * nvar),
                            squeeze=False, sharex="row")
    with uproot.open(file_path) as f:
        for j, (var, vlabel) in enumerate(TRACK_VARS):
            for i, (t, r) in enumerate(cells):
                ax = axs[j][i]
                for band, blabel, color in PT_BANDS:
                    vals, edges = f[_var_key(t, r, band, var)].to_numpy()
                    if vals.sum() > 0:
                        hep.histplot((vals, edges), ax=ax, histtype="step",
                                     color=color, density=density, label=blabel)
                if j == 0:
                    ax.set_title(f"R {SPECIES_TEX[r]} / T {SPECIES_TEX[t]}", fontsize=8)
                if j == nvar - 1:
                    ax.set_xlabel(vlabel, fontsize=8)
                if i == 0:
                    ax.set_ylabel(vlabel + ("\nPDF" if density else "\nCounts"), fontsize=7)
                if j == 0 and i == ncol - 1:
                    ax.legend(fontsize=6)
    fig.suptitle(r"Track variables by truth $\times$ reco-pion ($p<1$ vs $3<p<6$ GeV)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Render the v_z / PID diagnostic figures from a swim-analysis vz.root."
    )
    parser.add_argument("file", nargs="?", default="vz.root",
                        help="input ROOT file (default: vz.root)")
    parser.add_argument("-s", "--suffix", default="",
                        help="string inserted before the '.pdf' extension of every output "
                             "figure, e.g. --suffix _rgd -> vz_confusion_matrix_rgd.pdf")
    cli = parser.parse_args()
    file_path = cli.file

    def op(name: str) -> str:
        """Apply the --suffix to an output filename (before the extension)."""
        stem, dot, ext = name.rpartition(".")
        return f"{stem}{cli.suffix}.{ext}" if dot else f"{name}{cli.suffix}"

    # Optional pretty progress bar; falls back to plain prints if tqdm is absent.
    try:
        from tqdm import tqdm
        _HAVE_TQDM = True
    except Exception:
        _HAVE_TQDM = False

    # (label, callable) for every figure, run in order with progress.
    steps = [
        ("confusion_matrix", lambda: plot_confusion_matrix(file_path, out_path=op("vz_confusion_matrix.pdf"))),
        ("vz_matrix_integrated", lambda: plot_vz_matrix_integrated(file_path, out_path=op("vz_matrix_integrated.pdf"))),
        ("vz_matrix_p", lambda: plot_vz_matrix_p(file_path, out_path=op("vz_matrix_p.pdf"))),
        ("truth_vs_reco", lambda: plot_truth_vs_reco(file_path, out_path=op("vz_truth_vs_reco.pdf"))),
        ("dvz_mean_vs_p", lambda: plot_dvz_mean_vs_p(file_path, out_path=op("dvz_mean_vs_p.pdf"))),
        ("vztheta", lambda: plot_vztheta(file_path, out_path=op("vz_theta_2d.pdf"))),
        ("track_vars", lambda: plot_vars(file_path, out_path=op("track_vars.pdf"))),
        ("sigtx_matrix_p", lambda: plot_res_matrix_p(file_path, "sigtx", r"$\sigma_{t_x}$", out_path=op("sigtx_matrix_p.pdf"))),
        ("sigty_matrix_p", lambda: plot_res_matrix_p(file_path, "sigty", r"$\sigma_{t_y}$", out_path=op("sigty_matrix_p.pdf"))),
        ("sigtheta_matrix_p", lambda: plot_res_matrix_p(file_path, "sigtheta", r"$\sigma_\theta$ [mrad]", out_path=op("sigtheta_matrix_p.pdf"))),
        ("ptrue_vtx_dcr1", lambda: plot_ptrue_vtx_dcr1(file_path, out_path=op("ptrue_vtx_dcr1.pdf"))),
        ("dp_vtx_dcr1", lambda: plot_dp_vtx_dcr1(file_path, out_path=op("dp_vtx_dcr1.pdf"))),
        ("dp_vs_p", lambda: plot_dp_vs_p(file_path, out_path=op("dp_vs_p.pdf"))),
        ("dp_vtx_dcr1_zoom", lambda: plot_dp_vtx_dcr1(file_path, out_path=op("dp_vtx_dcr1_zoom.pdf"), dprange=(-0.01, 0.05))),
        ("dp_vs_p_zoom", lambda: plot_dp_vs_p(file_path, out_path=op("dp_vs_p_zoom.pdf"), dprange=(-0.01, 0.05))),
        ("dpf_vtx_dcr1", lambda: plot_dpf_vtx_dcr1(file_path, out_path=op("dpf_vtx_dcr1.pdf"))),
        ("dpf_vs_p", lambda: plot_dpf_vs_p(file_path, out_path=op("dpf_vs_p.pdf"))),
        ("dpf_vtx_dcr1_zoom", lambda: plot_dpf_vtx_dcr1(file_path, out_path=op("dpf_vtx_dcr1_zoom.pdf"), frange=(-0.005, 0.02))),
        ("dpf_vs_p_zoom", lambda: plot_dpf_vs_p(file_path, out_path=op("dpf_vs_p_zoom.pdf"), frange=(-0.005, 0.02))),
        ("gen_vs_reco_p (reads LUND files)", lambda: plot_gen_vs_reco_p(file_path, out_path=op("gen_vs_reco_p.pdf"))),
        ("swum_minus_rec", lambda: plot_swum_minus_rec(file_path, out_path=op("swum_minus_rec.pdf"))),
        ("swum_rec_vs_p", lambda: plot_swum_rec_vs_p(file_path, out_path=op("swum_rec_vs_p.pdf"))),
        ("rec_vs_swum", lambda: plot_rec_vs_swum(file_path, out_path=op("rec_vs_swum.pdf"))),
        ("swim_resolution", lambda: plot_swim_resolution(file_path, out_path=op("swim_resolution.pdf"))),
        ("vz_reco_swim_p", lambda: plot_vz_reco_swim_p(file_path, out_path=op("vz_reco_swim_p.pdf"))),
        ("recoswum_minus_rec", lambda: plot_recoswum_minus_rec(file_path, out_path=op("recoswum_minus_rec.pdf"))),
        ("pr1_reco_vs_true", lambda: plot_pr1_reco_vs_true(file_path, out_path=op("pr1_reco_vs_true.pdf"))),
        ("pr1_reco_minus_true", lambda: plot_pr1_reco_minus_true(file_path, out_path=op("pr1_reco_minus_true.pdf"))),
    ]

    n = len(steps)
    say = tqdm.write if _HAVE_TQDM else print
    print(f"Plotting {n} figures from {file_path}"
          + (f" (suffix '{cli.suffix}')" if cli.suffix else ""))
    bar = tqdm(total=n, unit="fig", desc="plots") if _HAVE_TQDM else None
    t_all = time.perf_counter()
    for i, (name, fn) in enumerate(steps, 1):
        say(f"[{i:2d}/{n}] {name} ...")
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # keep going so one bad figure doesn't abort all
            say(f"[{i:2d}/{n}] {name} FAILED: {exc}")
        else:
            say(f"[{i:2d}/{n}] {name} done ({time.perf_counter() - t0:.1f}s)")
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    print(f"All {n} figures done in {time.perf_counter() - t_all:.1f}s")
