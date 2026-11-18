#!/usr/bin/env python3
"""Figure 1: effect of the stamp size and FFT size on one galaxy, in GalSim only.

One isolated, elliptical exponential disk convolved with a round Moffat PSF, drawn
with GalSim. Reference: GalSim with default settings (GalSim picks the stamp size and
FFT size). Other panels: GalSim with a stamp size or FFT size smaller / larger than
its default choice, set only through the public API (`nx`, `ny` and `GSParams`).
JAX-GalSim is not used, so the residuals come from the numerics of drawing with FFTs,
not from differences between the two codes.

How GalSim picks the FFT size (`GSObject.drawFFT_makeKImage`):
`max(good_fft_size(max(getGoodImageSize, stamp)), minimum_fft_size)`.
So:

* "stamp smaller" / "stamp larger": only `nx`, `ny` change; the FFT size stays at
  GalSim's default because `good_fft_size(stamp) <= minimum_fft_size`.
* "FFT larger": `minimum_fft_size` raised above the default.
* "FFT smaller": `minimum_fft_size` lowered below the default (128), so GalSim uses
  `good_fft_size(getGoodImageSize)`. GalSim cannot use an FFT smaller than
  `good_fft_size(stamp)`.

The FFT size GalSim actually used is recorded for every panel and checked.

Every stamp is drawn at the same position and pasted into the center of a larger
canvas, so residuals are compared on a common grid. Residuals are relative to the peak
of the reference image, with a shared color scale.

Draft notebook: notebooks/2026-11-09-paper-figure1-fixed-sizes.ipynb
"""

import contextlib
from pathlib import Path

import galsim
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import AsinhNorm, SymLogNorm
from paper_style import IMAGE_CMAP, RES_CMAP, TEXT_WIDTH, pixel_axes, set_style

# ---------------------------------------------------------------- parameters
PIXEL_SCALE = 0.2  # arcsec / pixel

# galaxy: exponential disk only
HLR = 0.8  # arcsec
Q = 0.6  # axis ratio
BETA = 30.0  # degrees
FLUX = 1e5

# PSF: round Moffat (same as `--psf-type moffat` in timing_tests_general.py)
PSF_FWHM = 0.8  # arcsec
PSF_BETA = 2.5

# canvas the stamps are pasted into, and the central region shown in the figure
CANVAS = 192
CROP = 100  # must be larger than the largest stamp (STAMP_LARGE)

# smaller / larger than GalSim's default choice
STAMP_SMALL_FACTOR = 0.5  # times the default stamp size
STAMP_LARGE = 86  # pixels; good_fft_size(86) = 96 <= default FFT size, so the FFT size is unchanged
FFT_SMALL_MIN = 64  # minimum_fft_size below the default, so GalSim's own choice wins
FFT_LARGE_FACTOR = 2.0  # minimum_fft_size, times the default FFT size

# display
IMAGE_LINEAR_WIDTH = 1e-2  # asinh linear width, in units of the peak
IMAGE_TICKS = (0, 0.01, 0.1, 1)
RES_LINTHRESH = 1e-6  # symlog linear threshold, in units of the peak
PIXEL_TICKS = (-40, 0, 40)  # pixels from the galaxy center
TITLE_SIZE = 7  # points; smaller than the default so titles clear the colorbars
# ----------------------------------------------------------------------------

canvas_center = galsim.PositionD(CANVAS / 2 + 0.5, CANVAS / 2 + 0.5)


def make_obj(gsparams=None):
    gal = galsim.Exponential(half_light_radius=HLR, flux=FLUX).shear(
        q=Q, beta=BETA * galsim.degrees
    )
    psf = galsim.Moffat(beta=PSF_BETA, fwhm=PSF_FWHM)
    return galsim.Convolve([gal, psf], gsparams=gsparams)


@contextlib.contextmanager
def record_fft_sizes():
    """Record the FFT size GalSim uses in drawFFT, without changing anything."""
    orig = galsim.GSObject.drawFFT_makeKImage
    sizes = []

    def recording(self, image):
        kimage, wrap_size = orig(self, image)
        sizes.append(wrap_size)
        return kimage, wrap_size

    galsim.GSObject.drawFFT_makeKImage = recording
    try:
        yield sizes
    finally:
        galsim.GSObject.drawFFT_makeKImage = orig


def draw(stamp_size=None, gsparams=None):
    """Draw at the canvas center; return (canvas array, stamp size, FFT size used)."""
    kwargs = {} if stamp_size is None else dict(nx=stamp_size, ny=stamp_size)
    with record_fft_sizes() as sizes:
        im = make_obj(gsparams).drawImage(
            scale=PIXEL_SCALE, center=canvas_center, **kwargs
        )
    assert len(sizes) == 1
    out = np.zeros((CANVAS, CANVAS))
    b = im.bounds
    out[b.ymin - 1 : b.ymax, b.xmin - 1 : b.xmax] = im.array
    return out, im.array.shape[0], sizes[0]


def crop(x):
    lo = (CANVAS - CROP) // 2
    return x[lo : lo + CROP, lo : lo + CROP]


def main(out: Path = typer.Option(..., help="Output PNG path.")):
    set_style()
    assert CROP > STAMP_LARGE

    ref, gs_stamp, gs_fft = draw()
    peak = ref.max()
    min_fft = galsim.GSParams().minimum_fft_size
    print(f"default stamp size: {gs_stamp}")
    print(f"default FFT size: {gs_fft} (minimum_fft_size = {min_fft})")
    print(f"getGoodImageSize: {make_obj().getGoodImageSize(PIXEL_SCALE)}")

    cases = {
        "stamp smaller": dict(stamp_size=int(gs_stamp * STAMP_SMALL_FACTOR)),
        "stamp larger": dict(stamp_size=STAMP_LARGE),
        "FFT smaller": dict(gsparams=galsim.GSParams(minimum_fft_size=FFT_SMALL_MIN)),
        "FFT larger": dict(
            gsparams=galsim.GSParams(minimum_fft_size=int(gs_fft * FFT_LARGE_FACTOR))
        ),
    }

    res_rel, sizes = {}, {}
    for name, kwargs in cases.items():
        arr, nstamp, nfft = draw(**kwargs)
        r = arr - ref
        res_rel[name] = crop(r) / peak
        sizes[name] = (nstamp, nfft)
        print(
            f"{name:15s} stamp={nstamp:4d} fft={nfft:4d}  "
            f"max|res|/peak={np.abs(r).max() / peak:.2e}  dflux/flux={r.sum() / ref.sum():+.2e}"
        )

    # each panel changes exactly one of the two sizes, in the stated direction
    assert sizes["stamp smaller"][0] < gs_stamp and sizes["stamp smaller"][1] == gs_fft
    assert sizes["stamp larger"][0] > gs_stamp and sizes["stamp larger"][1] == gs_fft
    assert sizes["FFT smaller"][0] == gs_stamp and sizes["FFT smaller"][1] < gs_fft
    assert sizes["FFT larger"][0] == gs_stamp and sizes["FFT larger"][1] > gs_fft

    vmax = max(np.abs(r).max() for r in res_rel.values())
    res_norm = SymLogNorm(linthresh=RES_LINTHRESH, vmin=-vmax, vmax=vmax)

    # pixel edges relative to the galaxy center (which sits on a pixel corner, at 0)
    half = CROP / 2
    extent = (-half, half, -half, half)

    fig, axes = plt.subplots(1, 5, figsize=(TEXT_WIDTH, 1.75), constrained_layout=True)
    im0 = axes[0].imshow(
        crop(ref) / peak,
        origin="lower",
        extent=extent,
        cmap=IMAGE_CMAP,
        norm=AsinhNorm(linear_width=IMAGE_LINEAR_WIDTH, vmin=0, vmax=1),
    )
    axes[0].set_title(
        f"GalSim default\n(stamp {gs_stamp}, FFT {gs_fft})", fontsize=TITLE_SIZE
    )
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.85, pad=0.02, ticks=IMAGE_TICKS)
    cb0.ax.set_yticklabels([f"{t:g}" for t in IMAGE_TICKS])
    cb0.set_label("flux / peak")

    for ax, name in zip(axes[1:], cases):
        nstamp, nfft = sizes[name]
        imr = ax.imshow(
            res_rel[name], origin="lower", extent=extent, cmap=RES_CMAP, norm=res_norm
        )
        ax.set_title(f"{name}\n(stamp {nstamp}, FFT {nfft})", fontsize=TITLE_SIZE)
    cbr = fig.colorbar(imr, ax=axes[1:], shrink=0.85, pad=0.01)
    cbr.set_label("(changed $-$ default) / peak")

    pixel_axes(axes, PIXEL_TICKS)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    typer.run(main)
