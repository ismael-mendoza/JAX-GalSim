"""The full scene of Figure 2: parameters, galaxies, and the JAX-GalSim drawing.

Shared by `figure2_full_scene.py` (GalSim versus JAX-GalSim) and
`scene_cpu_gpu.py` (JAX-GalSim on CPU versus GPU), so both use the same scene.

The scene is drawn the same way as in the timing runs (`scripts/timing_tests_general.py`):
galaxies are grouped into size bins; each bin has a fixed stamp size, FFT size and
batch size (stratified approach, scatter-add into the scene). The galaxies are those of
sample `SAMPLE_INDEX` of a timing run with the same seed and number of samples (the PSF
here is fixed instead of varying per scene).
"""

import os

os.environ["JAX_ENABLE_X64"] = "True"

import sys
from functools import partial
from pathlib import Path

import galsim
import jax
import jax.numpy as jnp
import numpy as np
from jax import device_put, jit, random

import jax_galsim as jgs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from draw_scene_functions import (  # noqa: E402
    draw_all_bins_jgs,
    draw_jgs_vmap_scatter_stamps,
    get_good_sizes_galsim,
    get_one_full_sample,
    prepare_catalog,
    prepare_per_bin_samples,
)

# ---------------------------------------------------------------- parameters
IMAGE_SLEN = 250  # pixels (0.2 arcsec / pixel)
MAX_MAG = 27.0  # catalog cut on i_ab, as in the timing runs

# PSF: round Moffat (same as `--psf-type moffat` in timing_tests_general.py)
PSF_FWHM = 0.8  # arcsec
PSF_BETA = 2.5

# which galaxies: same keys as a timing run with this seed and number of samples
SEED = 45
N_SAMPLES_RUN = 1000
SAMPLE_INDEX = 0

# JAX-GalSim bins (250-pixel run with outliers, scripts/gpu-tests-scatter-vmap.sbatch)
STAMP_SLEN_BINS = (63, 83, 103, 250)
FFT_SIZE_BINS = (128, 128, 128, 256)
MAX_N_GALS_BINS = (65, 20, 5, 3)
MAX_N_ITERS = 5
MAX_N_GALS_GLOBAL = 150
BUFFER = 4  # pixels added to GalSim's good size before picking a bin
# ----------------------------------------------------------------------------


def psf_galsim():
    return galsim.Moffat(fwhm=PSF_FWHM, beta=PSF_BETA, flux=1.0)


def psf_jgs():
    return jgs.Moffat(fwhm=PSF_FWHM, beta=PSF_BETA, flux=1.0)


def load_scene(catsim_fpath: Path, cache_dir: Path):
    """Galaxies of the Figure 2 scene: (sample, number of galaxies, GalSim good sizes)."""
    # the random positions (and so the scene) differ in float32; scripts must set
    # JAX_ENABLE_X64 before importing jax
    assert jax.config.jax_enable_x64, "set JAX_ENABLE_X64 before importing jax"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # catalog, good sizes and cuts, as in timing_tests_general.py (with outliers)
    cat = prepare_catalog(str(catsim_fpath), min_hlr=0.0, max_mag=MAX_MAG)
    good_sizes, good_fft_sizes = get_good_sizes_galsim(
        cat=cat,
        psf=psf_galsim(),
        out_path=cache_dir,
        suffix=f"moffat-fwhm{PSF_FWHM}-imag{MAX_MAG:g}",
    )
    cat["good_size"] = good_sizes
    cat["good_fft_size"] = good_fft_sizes
    cat = cat[cat["good_size"] + BUFFER <= IMAGE_SLEN]
    assert max(STAMP_SLEN_BINS) >= max(cat["good_size"]) + BUFFER
    for s, f in zip(STAMP_SLEN_BINS, FFT_SIZE_BINS):
        assert np.all(cat[cat["good_size"] <= s]["good_fft_size"] <= f)
    print(f"galaxies in catalog after cuts: {len(cat)}")

    # one scene
    rkeys = random.split(random.key(SEED), N_SAMPLES_RUN)
    _, k2 = random.split(rkeys[SAMPLE_INDEX])
    sample, n_gals, gsizes = get_one_full_sample(
        k2, cat=cat, ilen=IMAGE_SLEN, max_n_gals=MAX_N_GALS_GLOBAL
    )
    bin_idx = np.searchsorted(STAMP_SLEN_BINS, gsizes + BUFFER)
    print(f"number of galaxies: {n_gals}")
    print(f"galaxies per bin: {np.bincount(bin_idx, minlength=len(STAMP_SLEN_BINS))}")
    return sample, n_gals, gsizes


def draw_scene_jgs(sample, gsizes, device):
    """Draw the scene with JAX-GalSim on `device`, as in the timing runs.

    Returns `(scene, draw)`, where `scene` is a NumPy array and `draw()` draws the
    scene again on the device from the same device inputs (no transfer, no
    compilation), returning the device array.
    """
    draw_fncs = [
        jit(
            partial(
                draw_jgs_vmap_scatter_stamps,
                ilen=IMAGE_SLEN,
                slen=s,
                fft_size=f,
                max_n_gals=m,
            )
        )
        for s, f, m in zip(STAMP_SLEN_BINS, FFT_SIZE_BINS, MAX_N_GALS_BINS)
    ]
    all_draw_fnc = jit(
        partial(
            draw_all_bins_jgs,
            ilen=IMAGE_SLEN,
            draw_fncs=draw_fncs,
            device=device,
            n_bins=len(STAMP_SLEN_BINS),
        )
    )
    samples_per_bin, n_iters_per_bin = prepare_per_bin_samples(
        sample,
        gsizes,
        stamp_slen_bins=STAMP_SLEN_BINS,
        max_n_gals_bins=MAX_N_GALS_BINS,
        max_n_iters=MAX_N_ITERS,
        buffer=BUFFER,
    )
    inputs = jax.block_until_ready(
        device_put(
            (samples_per_bin, jnp.asarray(n_iters_per_bin), psf_jgs()), device=device
        )
    )

    def draw():
        return jax.block_until_ready(all_draw_fnc(*inputs))

    scene = draw()
    assert scene.devices() == {device}, f"scene drawn on {scene.devices()}, not {device}"
    return np.asarray(scene), draw
