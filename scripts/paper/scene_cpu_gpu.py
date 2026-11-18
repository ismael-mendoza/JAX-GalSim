#!/usr/bin/env python3
"""Draw the Figure 2 scene with JAX-GalSim on the CPU and on the GPU; save both.

Runs on a GPU node (see `scripts/gpu-master.sbatch`). The scene, PSF and size bins are
those of Figure 2 (`fig2_scene.py`), drawn the same way as in the timing runs. Both
draws use the same code and inputs; only the device changes.

Writes an npz file with `scene_cpu`, `scene_gpu` (electrons per pixel, float64) and
some metadata. The appendix figure is made from it separately, on any machine:
`make -C scripts/paper figureA2` (`figureA2_cpu_gpu.py`).

`--second-device cpu` draws twice on the CPU, to test the script without a GPU.
"""

import os

# before importing jax; the scene itself (random positions) depends on it
os.environ["JAX_ENABLE_X64"] = "True"

from pathlib import Path

import jax
import numpy as np
import typer
from fig2_scene import SAMPLE_INDEX, SEED, draw_scene_jgs, load_scene

import jax_galsim as jgs


def main(
    out: Path = typer.Option(..., help="Output .npz path."),
    catsim_fpath: Path = typer.Option(..., help="CatSim OneDegSq.fits file."),
    cache_dir: Path = typer.Option(..., help="Folder to cache GalSim good sizes."),
    second_device: str = typer.Option("gpu", help="'gpu', or 'cpu' to test."),
):
    sample, n_gals, gsizes = load_scene(catsim_fpath, cache_dir)

    cpu = jax.devices("cpu")[0]
    other = jax.devices(second_device)[0]
    print(f"devices: {cpu} and {other}")

    scene_cpu, _ = draw_scene_jgs(sample, gsizes, cpu)
    scene_gpu, _ = draw_scene_jgs(sample, gsizes, other)
    assert scene_cpu.dtype == scene_gpu.dtype == np.float64

    peak = scene_cpu.max()
    r = scene_gpu - scene_cpu
    print(f"scene peak: {peak:.4g} electrons")
    print(f"max|GPU - CPU|/peak = {np.abs(r).max() / peak:.2e}")
    print(f"rms(GPU - CPU)/peak = {np.sqrt((r**2).mean()) / peak:.2e}")
    print(f"dflux/flux = {r.sum() / scene_cpu.sum():+.2e}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        scene_cpu=scene_cpu,
        scene_gpu=scene_gpu,
        n_gals=n_gals,
        seed=SEED,
        sample_index=SAMPLE_INDEX,
        device_cpu=str(cpu.device_kind),
        device_gpu=str(other.device_kind),
        jax_version=jax.__version__,
        jax_galsim_version=jgs.__version__,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    typer.run(main)
