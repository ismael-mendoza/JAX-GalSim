#!/usr/bin/env python3
"""Figure 2: one full 250 x 250 pixel scene, GalSim versus JAX-GalSim.

The scene is drawn the same way as in the timing runs (`scripts/timing_tests_general.py`).

* GalSim "default": GalSim loops over galaxies and picks the stamp size and FFT size
  for each one.
* JAX-GalSim: galaxies are grouped into size bins; each bin has a fixed stamp size,
  FFT size and batch size (stratified approach, scatter-add into the scene).
* GalSim "matched": GalSim draws each galaxy with the stamp size and FFT size of its
  JAX-GalSim bin (`draw_all_galsim_stratified`).

Panels: GalSim scene; JAX-GalSim scene; residual (JAX-GalSim - GalSim default);
residual (JAX-GalSim - GalSim matched). Residuals are relative to the peak of the
GalSim scene, as in Figure 1. No noise is added.

The scene (galaxies, PSF, bins) is defined in `fig2_scene.py`, shared with the
CPU-versus-GPU appendix figure.

Draft notebook: notebooks/2026-11-09-paper-figure2-full-scene.ipynb
"""

import os

# before importing jax; the scene itself (random positions) depends on it
os.environ["JAX_ENABLE_X64"] = "True"

import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import typer
from fig2_scene import (
    BUFFER,
    FFT_SIZE_BINS,
    IMAGE_SLEN,
    STAMP_SLEN_BINS,
    draw_scene_jgs,
    load_scene,
    psf_galsim,
)
from matplotlib.colors import AsinhNorm, SymLogNorm
from paper_style import IMAGE_CMAP, RES_CMAP, TEXT_WIDTH, pixel_axes, set_style

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from draw_scene_functions import draw_all_galsim_stratified, draw_galsim  # noqa: E402

# ---------------------------------------------------------------- parameters
# display
SCENE_LINEAR_WIDTH = 0.02  # asinh linear width, in units of the peak
SCENE_TICKS = (0, 0.01, 0.1, 1)
PIXEL_TICKS = (0, 100, 200)  # pixels from the scene's lower-left corner
# symlog linear threshold for each residual panel, in units of the peak
RES_LINTHRESH = {"default": 1e-5, "matched": 1e-11}
# ----------------------------------------------------------------------------


def main(
    out: Path = typer.Option(..., help="Output PNG path."),
    catsim_fpath: Path = typer.Option(..., help="CatSim OneDegSq.fits file."),
    cache_dir: Path = typer.Option(..., help="Folder to cache GalSim good sizes."),
):
    set_style()
    psf_gs = psf_galsim()
    sample, n_gals, gsizes = load_scene(catsim_fpath, cache_dir)

    # GalSim default; also checks each JAX-GalSim bin is at least as large as GalSim's stamp
    gs_default = draw_galsim(
        sample,
        n_gals,
        ilen=IMAGE_SLEN,
        psf=psf_gs,
        check_stamp_sizes=True,
        max_slen=STAMP_SLEN_BINS[-1],
        good_sizes=gsizes,
        buffer=BUFFER,
        size_bins=STAMP_SLEN_BINS,
    )

    # GalSim with the same stamp and FFT sizes as JAX-GalSim
    gs_matched = draw_all_galsim_stratified(
        sample,
        ilen=IMAGE_SLEN,
        psf=psf_gs,
        gsizes=gsizes,
        stamp_slen_bins=STAMP_SLEN_BINS,
        fft_size_bins=FFT_SIZE_BINS,
        buffer=BUFFER,
    )

    # JAX-GalSim, as in the timing runs (on whatever device JAX picks)
    jgs_scene, _ = draw_scene_jgs(sample, gsizes, jax.devices()[0])

    # residuals relative to the GalSim peak
    peak = gs_default.max()
    res = {"default": jgs_scene - gs_default, "matched": jgs_scene - gs_matched}
    print(f"scene peak: {peak:.4g} electrons")
    for name, r in res.items():
        print(
            f"{name:8s} max|res|/peak={np.abs(r).max() / peak:.2e}  "
            f"dflux/flux={r.sum() / gs_default.sum():+.2e}"
        )

    # each residual panel gets its own symmetric color scale
    res_rel = {k: v / peak for k, v in res.items()}
    res_norms = {}
    for name, linthresh in RES_LINTHRESH.items():
        vmax = np.abs(res_rel[name]).max()
        res_norms[name] = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)
    scene_norm = AsinhNorm(linear_width=SCENE_LINEAR_WIDTH, vmin=0, vmax=1)

    extent = (0, IMAGE_SLEN, 0, IMAGE_SLEN)  # pixel edges

    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH, 2.0), constrained_layout=True)
    for ax, arr, title in zip(
        axes[:2], [gs_default, jgs_scene], ["GalSim", "JAX-GalSim"]
    ):
        ims = ax.imshow(
            arr / peak, origin="lower", extent=extent, cmap=IMAGE_CMAP, norm=scene_norm
        )
        ax.set_title(title)
    cbs = fig.colorbar(ims, ax=axes[:2], shrink=0.85, pad=0.01, ticks=SCENE_TICKS)
    cbs.ax.set_yticklabels([f"{t:g}" for t in SCENE_TICKS])
    cbs.set_label("flux / peak")

    for ax, name, title in zip(
        axes[2:],
        ["default", "matched"],
        ["residual\n(default sizes)", "residual\n(matched sizes)"],
    ):
        imr = ax.imshow(
            res_rel[name],
            origin="lower",
            extent=extent,
            cmap=RES_CMAP,
            norm=res_norms[name],
        )
        ax.set_title(title)
        cbr = fig.colorbar(imr, ax=ax, shrink=0.85, pad=0.02)
    cbr.set_label("(JAX-GalSim $-$ GalSim) / peak")

    pixel_axes(axes, PIXEL_TICKS)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    typer.run(main)
