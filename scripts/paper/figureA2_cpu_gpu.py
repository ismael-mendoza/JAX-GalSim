#!/usr/bin/env python3
"""Appendix figure: the Figure 2 scene drawn by JAX-GalSim on a GPU versus a CPU.

Reads the npz written on a GPU node by `scene_cpu_gpu.py` (part of
`scripts/gpu-master.sbatch`). One panel: the residual (GPU - CPU) relative to the peak
of the CPU scene, on a linear symmetric color scale. The scene itself is not shown; it
is the JAX-GalSim panel of Figure 2.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import Normalize
from paper_style import COLUMN_WIDTH, RES_CMAP, pixel_axes, set_style

# ---------------------------------------------------------------- parameters
PIXEL_TICKS = (0, 100, 200)  # pixels from the scene's lower-left corner, as in Figure 2
FIG_HEIGHT = 2.75  # inches, at one column width
# ----------------------------------------------------------------------------


def main(
    npz: Path = typer.Option(..., help="npz written by scene_cpu_gpu.py."),
    out: Path = typer.Option(..., help="Output PNG path."),
):
    set_style()
    d = np.load(npz)
    cpu, gpu = d["scene_cpu"], d["scene_gpu"]
    print(f"devices: {d['device_cpu']} vs {d['device_gpu']}; JAX {d['jax_version']}")

    peak = cpu.max()
    res = (gpu - cpu) / peak
    print(f"scene peak: {peak:.4g} electrons")
    print(f"max|GPU - CPU|/peak = {np.abs(res).max():.2e}")
    print(f"rms(GPU - CPU)/peak = {np.sqrt((res**2).mean()):.2e}")
    print(f"dflux/flux = {(gpu - cpu).sum() / cpu.sum():+.2e}")

    n = cpu.shape[0]
    extent = (0, n, 0, n)  # pixel edges

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, FIG_HEIGHT), constrained_layout=True)
    vmax = max(np.abs(res).max(), np.finfo(float).tiny)  # all zeros if same device
    imr = ax.imshow(
        res,
        origin="lower",
        extent=extent,
        cmap=RES_CMAP,
        norm=Normalize(vmin=-vmax, vmax=vmax),
    )
    cbr = fig.colorbar(imr, ax=ax, shrink=0.9, pad=0.02)
    cbr.formatter.set_powerlimits((0, 0))
    cbr.set_label("(GPU $-$ CPU) / peak")

    pixel_axes([ax], PIXEL_TICKS)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    typer.run(main)
