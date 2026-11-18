#!/usr/bin/env python3
"""Appendix figure: the matched-size residual of Figure 2 comes from the Moffat table.

JAX-GalSim does not evaluate the Fourier transform of an (untruncated) Moffat profile
directly. It tabulates it at 700 evenly spaced points between k = 0 and maxk and uses
Akima interpolation between them (`Moffat._kValue_untrunc_interp_coeffs`), while GalSim
evaluates it directly. This leaves a residual of about 1e-9 of the peak when both codes
use the same stamp size and FFT size (the "matched" panel of Figure 2).

Here one bulge + disk galaxy, built and drawn as in Figure 2 (same profiles, PSF,
pixel scale, and the stamp size and FFT size of Figure 2's first bin), is drawn with:

* GalSim;
* JAX-GalSim as released (700 table points);
* JAX-GalSim with a finer table (enough points to reach float64 round-off).

The last panel shows the maximum residual against the number of table points, and the
round-off floor: the same galaxy with a Gaussian PSF, which needs no table.

The library is not changed. The number of table points is changed only inside this
script, by temporarily replacing `Moffat._kValue_untrunc_interp_coeffs` with a copy
that takes the number of points as a parameter. The script checks that the copy with
700 points gives exactly the same image as the library.
"""

import os
import sys

os.environ["JAX_ENABLE_X64"] = "True"

import contextlib
from pathlib import Path

import galsim
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import AsinhNorm, Normalize, SymLogNorm
from paper_style import IMAGE_CMAP, RES_CMAP, TEXT_WIDTH, pixel_axes, set_style

import jax_galsim as jgs
from jax_galsim.core.interpolate import akima_interp_coeffs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from draw_scene_functions import get_bd_galsim, get_bd_jgs  # noqa: E402

# ---------------------------------------------------------------- parameters
PIXEL_SCALE = 0.2  # arcsec / pixel

# galaxy: bulge (Spergel, nu = -0.6) + disk (exponential), as in Figure 2
GALAXY = dict(
    flux_d=7e4,  # electrons
    flux_b=3e4,  # electrons
    hlr_d=0.6,  # arcsec
    hlr_b=0.25,  # arcsec
    q_d=0.6,
    q_b=0.8,
    beta=30.0,  # degrees
)

# PSF: round Moffat, as in Figures 1 and 2
PSF_FWHM = 0.8  # arcsec
PSF_BETA = 2.5

# stamp size and FFT size of the first bin in Figure 2
STAMP = 63
FFT = 128

# galaxy center, in pixels (not on a pixel center or corner, as in a scene)
CENTER = (32.37, 31.81)

# Moffat table
DEFAULT_N_PTS = 700  # hard-coded in jax_galsim/moffat.py
N_PTS_SCAN = tuple(DEFAULT_N_PTS * 2**j for j in range(11))
FINE_N_PTS = DEFAULT_N_PTS * 2**9  # first point that reaches the round-off floor

# display
IMAGE_LINEAR_WIDTH = 1e-2  # asinh linear width, in units of the peak
IMAGE_TICKS = (0, 0.01, 0.1, 1)
RES_LINTHRESH = 1e-11  # same as the matched panel of Figure 2
PIXEL_TICKS = (-25, 0, 25)  # pixels from the galaxy center
TITLE_SIZE = 7  # points
# ----------------------------------------------------------------------------


@contextlib.contextmanager
def moffat_table(n_pts):
    """Use `n_pts` points in JAX-GalSim's Moffat k-space table.

    Same as `Moffat._kValue_untrunc_interp_coeffs` in jax_galsim/moffat.py, except
    for the number of points. JAX caches are cleared on entry and exit, so jitted
    functions traced with another table are not reused.
    """
    orig = jgs.Moffat._kValue_untrunc_interp_coeffs

    def coeffs(self):
        k = jnp.linspace(0, self._maxk, n_pts)
        beta = jax.lax.stop_gradient(self.beta)
        vals = self._kValue_untrunc_func(
            beta, k, self._knorm_bis, self._knorm, self._r0
        )
        aval = self._kValue_untrunc_asymp_func(beta, k[-1], self._knorm_bis, self._r0)
        slp = (vals[-1] / aval - 1) * k[-1] * self._r0
        return k, vals, akima_interp_coeffs(k, vals), slp

    jgs.Moffat._kValue_untrunc_interp_coeffs = coeffs
    jax.clear_caches()
    try:
        yield
    finally:
        jgs.Moffat._kValue_untrunc_interp_coeffs = orig
        jax.clear_caches()


def draw_galsim(psf):
    obj = get_bd_galsim(**GALAXY, psf=psf).withGSParams(
        minimum_fft_size=FFT, maximum_fft_size=FFT
    )
    im = obj.drawImage(
        center=galsim.PositionD(*CENTER),
        wcs=galsim.PixelScale(PIXEL_SCALE),
        nx=STAMP,
        ny=STAMP,
        dtype=np.float64,
    )
    return im.array, im.bounds


def draw_jgs(psf):
    gsparams = jgs.GSParams(minimum_fft_size=FFT, maximum_fft_size=FFT)
    obj = get_bd_jgs(**GALAXY, psf=psf).withGSParams(gsparams)
    im = obj.drawImage(
        center=jgs.PositionD(*CENTER),
        wcs=jgs.PixelScale(PIXEL_SCALE),
        nx=STAMP,
        ny=STAMP,
        dtype=jnp.float64,
    )
    return np.asarray(im.array)


def main(out: Path = typer.Option(..., help="Output PNG path.")):
    set_style()

    moffat_gs = galsim.Moffat(fwhm=PSF_FWHM, beta=PSF_BETA, flux=1.0)
    moffat_jgs = jgs.Moffat(fwhm=PSF_FWHM, beta=PSF_BETA, flux=1.0)

    ref, bounds = draw_galsim(moffat_gs)
    peak = ref.max()

    # JAX-GalSim as released
    jax.clear_caches()
    default = draw_jgs(moffat_jgs)

    # the copied table code with the default number of points is the library
    with moffat_table(DEFAULT_N_PTS):
        assert np.array_equal(draw_jgs(moffat_jgs), default)

    # residual against the number of table points
    max_res = []
    for n in N_PTS_SCAN:
        with moffat_table(n):
            r = draw_jgs(moffat_jgs) - ref
        max_res.append(np.abs(r).max() / peak)
        print(f"n_pts={n:8d}  max|res|/peak={max_res[-1]:.2e}")
    max_res = np.array(max_res)

    with moffat_table(FINE_N_PTS):
        fine = draw_jgs(moffat_jgs)

    # round-off floor: same galaxy with a Gaussian PSF (no table)
    gauss_ref, _ = draw_galsim(galsim.Gaussian(fwhm=PSF_FWHM, flux=1.0))
    gauss = draw_jgs(jgs.Gaussian(fwhm=PSF_FWHM, flux=1.0))
    floor = np.abs(gauss - gauss_ref).max() / gauss_ref.max()

    res_default = (default - ref) / peak
    res_fine = (fine - ref) / peak
    print(f"default ({DEFAULT_N_PTS} points): max|res|/peak={np.abs(res_default).max():.2e}")
    print(f"fine ({FINE_N_PTS} points):    max|res|/peak={np.abs(res_fine).max():.2e}")
    print(f"Gaussian PSF floor:          max|res|/peak={floor:.2e}")
    assert np.abs(res_fine).max() < 3 * floor, "fine table did not reach the round-off floor"

    # pixel edges relative to the galaxy center
    extent = (
        bounds.xmin - 0.5 - CENTER[0],
        bounds.xmax + 0.5 - CENTER[0],
        bounds.ymin - 0.5 - CENTER[1],
        bounds.ymax + 0.5 - CENTER[1],
    )

    fig = plt.figure(figsize=(TEXT_WIDTH, 1.8), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, width_ratios=[2.75, 1.0], wspace=0.02)
    axes = subfigs[0].subplots(1, 3)

    im0 = axes[0].imshow(
        ref / peak,
        origin="lower",
        extent=extent,
        cmap=IMAGE_CMAP,
        norm=AsinhNorm(linear_width=IMAGE_LINEAR_WIDTH, vmin=0, vmax=1),
    )
    axes[0].set_title("GalSim\n(bulge + disk)", fontsize=TITLE_SIZE)
    cb0 = subfigs[0].colorbar(
        im0, ax=axes[0], shrink=0.85, pad=0.02, ticks=IMAGE_TICKS
    )
    cb0.ax.set_yticklabels([f"{t:g}" for t in IMAGE_TICKS])
    cb0.set_label("flux / peak")

    # each residual panel gets its own symmetric color scale: symlog for the default
    # table (as in Figure 2's matched panel), linear for the round-off left with the
    # finer table
    vd = np.abs(res_default).max()
    vf = np.abs(res_fine).max()
    for ax, res, norm, title in zip(
        axes[1:],
        [res_default, res_fine],
        [
            SymLogNorm(linthresh=RES_LINTHRESH, vmin=-vd, vmax=vd),
            Normalize(vmin=-vf, vmax=vf),
        ],
        [
            f"default table\n({DEFAULT_N_PTS:,} points)",
            f"finer table\n({FINE_N_PTS:,} points)",
        ],
    ):
        imr = ax.imshow(res, origin="lower", extent=extent, cmap=RES_CMAP, norm=norm)
        ax.set_title(title, fontsize=TITLE_SIZE)
        cbr = subfigs[0].colorbar(imr, ax=ax, shrink=0.85, pad=0.02)
    cbr.formatter.set_powerlimits((0, 0))
    cbr.ax.yaxis.get_offset_text().set_fontsize(6)
    cbr.set_label("(JAX-GalSim $-$ GalSim) / peak")
    pixel_axes(axes, PIXEL_TICKS)

    # convergence with the number of table points
    axc = subfigs[1].subplots()
    axc.loglog(N_PTS_SCAN, max_res, "o-", color="0.2", ms=2.5, lw=1)
    for n, color in [(DEFAULT_N_PTS, "C3"), (FINE_N_PTS, "C0")]:
        i = N_PTS_SCAN.index(n)
        axc.loglog(n, max_res[i], "o", color=color, ms=4, zorder=3)
    axc.axhline(floor, color="0.5", ls="--", lw=0.8)
    axc.text(
        N_PTS_SCAN[0],
        floor / 2.5,
        "Gaussian PSF (round-off)",
        fontsize=6,
        color="0.4",
        va="top",
    )
    axc.set_xlabel("Moffat table points")
    axc.set_ylabel(r"max $|$residual$|$ / peak")
    axc.set_ylim(floor / 40, max_res.max() * 5)
    axc.tick_params(which="both", direction="in", top=True, right=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    typer.run(main)
