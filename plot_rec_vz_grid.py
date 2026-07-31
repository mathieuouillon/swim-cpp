"""Stacked momentum×theta grids of reconstructed v_z, one block per pass.

Blocks are stacked top-to-bottom in argument order (first file on top). Within a
block there is one panel per momentum bin and one area-normalized v_z curve per
theta bin (colored), matching the field-alignment note's (p, theta) figure style.
Built to show the same run reconstructed at several torus scales (e.g. scale 1.0
on top, 0.998 and 0.9982 below) and read the theta/momentum dependence off each
block directly.

Reads the vz_rec_<sp>_pAA_BB_thCC_DD histograms written by rec-vz-hist (the
loaders/style are shared with plot_rec_vz_compare) and writes one figure per
species: rec_vz_<sp>_grid<suffix>.pdf.

Usage:
  python plot_rec_vz_grid.py block1.root block2.root [block3.root ...] \
      --labels "scale 1.0" "scale 0.998" ["scale 0.9982" ...] \
      [--species ele pip pim] [--suffix _tag]
"""
import argparse
import os
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

# Reuse the ROOT loaders, the area-normalized step drawer, the mean helper and
# the style block (mplhep + scienceplots, set up on import) from the overlay
# plotter so both figures parse the same files and render identically.
import plot_rec_vz_compare as cmp


def theta_color_map(t_bins):
    """One distinct color per theta bin; tab10 for <=10 bins (the electron grid,
    matching the reference figure), tab20 above (the 12-bin pion grid)."""
    cmap = plt.get_cmap("tab10" if len(t_bins) <= 10 else "tab20")
    return {t: cmap(i) for i, t in enumerate(t_bins)}


def plot_block(subfig, sp, cells, label, p_bins, t_bins, tcolor, win, nrow, ncol,
               show_legend=True):
    """One pass's block: a nrow×ncol momentum grid with an area-normalized v_z
    curve per theta bin in each panel, a theta legend and a title on top. The top
    ~18% of the subfigure is reserved for the title (row 1) and legend (row 2).
    show_legend=False drops the per-block theta legend (used for all but the top
    block when many blocks are stacked, e.g. a torus-scale scan, to cut clutter)."""
    axs = subfig.subplots(nrow, ncol, squeeze=False)
    for ax in axs.flat:
        ax.set_visible(False)
    for k, (plo, phi) in enumerate(p_bins):
        ax = axs.flat[k]
        ax.set_visible(True)
        for tb in t_bins:
            cell = (plo, phi, *tb)
            if cell not in cells:
                continue
            counts, edges = cells[cell]
            cmp.histstep(counts, edges, ax=ax, color=tcolor[tb], density=True,
                         norm_range=win, linewidth=1.0)
        if win:
            ax.set_xlim(*win)
        ax.set_title(rf"${plo:.1f} < p < {phi:.1f}$ GeV/$c$", fontsize=9)
        ax.tick_params(labelsize=7)
        if k % ncol == 0:
            ax.set_ylabel("Normalized", fontsize=8)
        if k + ncol >= len(p_bins):  # bottom-most filled panel in its column
            ax.set_xlabel(r"$v_z$ [cm]", fontsize=8)
    subfig.subplots_adjust(top=0.82, bottom=0.09, left=0.06, right=0.98,
                           hspace=0.32, wspace=0.22)
    if show_legend:
        handles = [Line2D([0], [0], color=tcolor[(tlo, thi)], lw=1.6,
                          label=rf"${tlo}^\circ \leq \theta < {thi}^\circ$") for tlo, thi in t_bins]
        subfig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93),
                      ncol=5, fontsize=8, columnspacing=1.2, handletextpad=0.4, frameon=False)
    subfig.suptitle(rf"{cmp.SPECIES_TEX.get(sp, sp)}   all sectors combined — "
                    rf"reconstructed — {label}", fontsize=11, y=0.985)


def plot_species_grid(sp, datasets, pdf):
    """datasets = [(cells_dict, label), ...] top-to-bottom; one stacked block each."""
    all_cells = set()
    for cells, _ in datasets:
        all_cells |= set(cells)
    if not all_cells:
        print(f"[warn] no {sp} cells in any file; skipping")
        return
    p_bins = sorted({(plo, phi) for plo, phi, _, _ in all_cells})
    t_bins = sorted({(tlo, thi) for _, _, tlo, thi in all_cells})
    win = None
    for cells, _ in datasets:
        for counts, edges in cells.values():
            win = (float(edges[0]), float(edges[-1]))
            break
        if win:
            break

    ncol = min(4, len(p_bins))
    nrow = ceil(len(p_bins) / ncol)
    tcolor = theta_color_map(t_bins)

    n = len(datasets)
    fig = plt.figure(figsize=(3.6 * ncol, (3.0 * nrow + 1.4) * n))
    subfigs = fig.subfigures(n, 1, hspace=0.04)
    if n == 1:
        subfigs = [subfigs]
    # With many blocks (e.g. a 21-point torus-scale scan) one shared theta legend
    # on the top block is enough; repeating it per block is just clutter.
    many = n > 4
    for i, (subfig, (cells, label)) in enumerate(zip(subfigs, datasets)):
        plot_block(subfig, sp, cells, label, p_bins, t_bins, tcolor, win, nrow, ncol,
                   show_legend=(not many or i == 0))
    # Faint full-width rule between blocks (top reconstruction vs bottom).
    for i in range(1, n):
        fig.add_artist(Line2D([0.03, 0.97], [1 - i / n, 1 - i / n], color="0.7",
                              lw=0.8, transform=fig.transFigure))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+",
                    help="rec-vz ROOT files, one stacked block each (top to bottom)")
    ap.add_argument("--labels", nargs="+",
                    help="block titles, one per file (top to bottom); "
                         "defaults to the file basenames")
    ap.add_argument("--species", nargs="+", default=["ele", "pip", "pim"],
                    choices=["ele", "pip", "pim"], help="species to plot")
    ap.add_argument("--suffix", default="", help="suffix added to output PDF names")
    ap.add_argument("--outdir", default=".", help="directory for the output PDFs")
    args = ap.parse_args()

    labels = args.labels
    if labels is None:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in args.files]
    elif len(labels) != len(args.files):
        ap.error(f"--labels takes one label per file "
                 f"({len(args.files)} file(s), {len(labels)} label(s) given)")

    loaded = [cmp.load_cells(f) for f in args.files]  # [(cells, integrated), ...]
    os.makedirs(args.outdir, exist_ok=True)

    for sp in args.species:
        # One block per file that carries this species, kept in argument order.
        datasets = [(cells[sp], label)
                    for (cells, _), label in zip(loaded, labels) if sp in cells]
        if not datasets:
            print(f"[warn] {sp} absent in all {len(args.files)} file(s); skipping")
            continue
        out = os.path.join(args.outdir, f"rec_vz_{sp}_grid{args.suffix}.pdf")
        with PdfPages(out) as pdf:
            plot_species_grid(sp, datasets, pdf)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
