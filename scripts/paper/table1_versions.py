"""Table 1 versions: write the JAX-GalSim and GalSim versions the API coverage table describes.

Table 1 (`tables/table1_api_coverage.txt` in the paper repo) is written by hand; it
`\\input`s the file written here, which defines two LaTeX macros:

- `\\jaxgalsimversion`: the latest JAX-GalSim release tag before the checked-out commit,
  plus that commit when it is not the release itself (e.g. "v2026.7.0 + commit 58fd01d").
  Needs the upstream release tags: `git fetch upstream --tags`.
- `\\galsimversion`: the latest GalSim release on PyPI.
"""

import json
import subprocess
import urllib.request
from pathlib import Path

import typer

# ---------------------------------------------------------------- parameters
REPO = Path(__file__).resolve().parents[2]
RELEASE_TAGS = ("v[0-9]*", "[0-9]*")  # release tags, with or without the "v"
GALSIM_PYPI_URL = "https://pypi.org/pypi/GalSim/json"
# ----------------------------------------------------------------------------


def jax_galsim_version() -> str:
    cmd = ["git", "-C", str(REPO), "describe", "--tags", "--long", "--abbrev=7"]
    for pattern in RELEASE_TAGS:
        cmd += ["--match", pattern]
    # e.g. "v2026.7.0-155-g58fd01d"
    tag, distance, commit = (
        subprocess.check_output(cmd, text=True).strip().rsplit("-", 2)
    )
    tag = tag if tag.startswith("v") else f"v{tag}"
    return tag if distance == "0" else f"{tag} + commit {commit.removeprefix('g')}"


def latest_galsim_release() -> str:
    with urllib.request.urlopen(GALSIM_PYPI_URL, timeout=30) as response:
        return json.load(response)["info"]["version"]


def main(out: Path = typer.Option(..., help="Output .txt path (LaTeX).")):
    jgs_version = jax_galsim_version()
    galsim_version = latest_galsim_release()
    print(f"JAX-GalSim {jgs_version}, GalSim {galsim_version}")
    lines = [
        "% Versions described by Table 1. Written by JAX-GalSim"
        " scripts/paper/table1_versions.py;",
        "% do not edit by hand. Input by tables/table1_api_coverage.txt.",
        f"\\def\\jaxgalsimversion{{{jgs_version}}}",
        f"\\def\\galsimversion{{{galsim_version}}}",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved {out}")


if __name__ == "__main__":
    typer.run(main)
