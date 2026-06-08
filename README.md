# vid2scene Replicate (cog)

## Project Overview

[Replicate Cog](https://github.com/replicate/cog) deployment of **[vid2scene](https://github.com/samuelm2/vid2scene)** by **[Sam M² / SamuSynth](https://samusynth.com)** — a **video → 3D Gaussian splat** pipeline (frame extraction → [hloc](https://github.com/cvg/Hierarchical-Localization) + [GLOMAP](https://github.com/colmap/glomap) structure-from-motion → [gsplat](https://github.com/nerfstudio-project/gsplat) training → `.ply`).

Deployed at **https://replicate.com/kfarr/vid2scene** (L40S GPU).

This wraps **only** vid2scene's standalone `vid2scene_core/` reconstruction pipeline. It deliberately uses **none** of the vid2scene SaaS (Django web, Postgres, Redis/django-rq, Azurite blob, Stripe billing) — [3DStreet](https://3dstreet.org) supplies the queue, storage, auth, and tokens. The contract is intentionally tiny: **a video in → one `.ply` Gaussian splat out.**

Modeled after [`kfarr/sharp-ml-replicate`](https://github.com/kfarr/sharp-ml-replicate) and [`3DStreet/hy-world-2-replicate`](https://github.com/3DStreet/hy-world-2-replicate).

> **Why this exists:** vid2scene is [winding down its web operations](https://vid2scene.com/winddown/) (June 13, 2026) and the team has generously open-sourced the entire codebase under **Apache 2.0**. Huge thanks to Sam and the SamuSynth team — this Cog just repackages their excellent work so the reconstruction pipeline keeps living on, callable as a Replicate model. All the hard parts (the SfM + gsplat pipeline, the tooling, the tuning) are theirs.

## Files
- `cog.yaml` — Build config: CUDA 12.4, Python 3.10, COLMAP 3.10 built from source, GLOMAP + gsplat + hloc from the pinned upstream submodules, slim worker deps. Mirrors the relevant parts of the upstream `Worker_Dockerfile` (minus conda + the SaaS).
- `predict.py` — Cog `Predictor` wrapping `vid2scene_core/vid2scene.py::process_video_to_scene(...)`.
- `LICENSE` — Apache 2.0 (this wrapper). Upstream vid2scene is also Apache 2.0.

Upstream is **pinned** to [`samuelm2/vid2scene@04559366`](https://github.com/samuelm2/vid2scene/commit/04559366ce7b3bb4a2ef261f364379c9761623b6) in `cog.yaml` for reproducible rebuilds.

## Inputs
| Input | Default | Notes |
| --- | --- | --- |
| `video` | — | A slow, steady **orbit of a static subject** works best. `.mp4`, `.mov` (incl. iPhone HEVC), etc. — extraction is via ffmpeg, format-agnostic. Aspect ratio is **preserved** (downscaled so the long edge ≤ 1920 px; never cropped). |
| `reconstruction_method` | `glomap` | `glomap` (global SfM) needs no model weights. `colmap` (incremental) also available. |
| `target_framecount` | `600` | Frames sampled from the video. Drives SfM cost. |
| `training_num_steps` | `30000` | gsplat training steps. Lower = faster + cheaper, lower quality. |
| `training_max_num_gaussians` | `500000` | **Primary control on `.ply` size** (~164 bytes/Gaussian). Capped at **600000 (~98 MB)** so output stays under 3DStreet's 100 MB save ceiling; default 500k ≈ 82 MB. |
| `equirectangular` | `false` | Treat input as 360°/equirectangular video (rig-based SfM + perspective split). |
| `use_background_sphere` | `false` | Add a background sphere for distant/sky content (outdoor / 360 captures). |
| `remove_background` | `false` | InSPyReNet background removal per frame — good for isolating a single object; leave **off** for scenes/environments. |
| `apriltag_size_meters` | unset | If an AprilTag of known physical size (meters) is visible, scales the reconstruction to **real-world units**. |

## Output
A single `.ply` Gaussian splat (`output_dir/ply/splat.ply` upstream). Drop straight into 3DStreet, [Spark](https://github.com/sparkjsdev/spark), or any splat viewer. The gsplat exporter writes position + SH-degree-2 color + opacity/scale/rotation (41 floats = 164 bytes per Gaussian).

## Build & Push

> **You do NOT need a GPU to build.** The CUDA toolkit cross-compiles for the
> arches in `TORCH_CUDA_ARCH_LIST` (`7.5;8.6;8.9`). A GPU is only needed to *run*
> (`cog predict`) or serve. This image was built on a **CPU-only** box and
> smoke-tested on Replicate. The build is heavy (~30 GB image, compiles
> COLMAP/Ceres/GLOMAP): use **≥32 GB RAM and ≥100 GB free disk**.

### Prerequisites
```bash
# Docker + cog on an x86_64 Linux host:
curl -fsSL https://get.docker.com | sh
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog
git clone https://github.com/3DStreet/vid2scene-cog.git && cd vid2scene-cog
```

### Build
```bash
cog build -t vid2scene-cog        # ~20-25 min clean; compiles COLMAP/GLOMAP/gsplat
```

### Push
```bash
# Auth: see "Build Issues" #6 — docker login with the API token is the reliable path.
docker login r8.im -u <your-replicate-username>   # paste API token as the password
cog push r8.im/<owner>/vid2scene                  # create the model on replicate.com first
```

### Local test (needs a GPU)
```bash
cog predict -i video=@orbit.mp4 -i training_num_steps=7000
```

### Calling from the Replicate API
```python
import replicate
output = replicate.run(
    "kfarr/vid2scene:<version>",
    input={"video": open("orbit.mp4", "rb"), "target_framecount": 300},
)
# `output` is a URL to the resulting .ply — download it.
```

### Point 3DStreet at it
Set `REPLICATE_MODELS.vid2scene.modelName` in `3dstreet/public/functions/replicate-models.js` to `<owner>/vid2scene` (currently `kfarr/vid2scene`). The backend resolves the latest version at runtime, so re-pushes don't need a code change.

## Build Issues Encountered & Fixes

### 1. Submodules live at the repo top level (FIXED)
GLOMAP, gsplat, Hierarchical-Localization, vggt, etc. are top-level submodules of vid2scene — not under `third_party/` or `vid2scene_core/`. `cog.yaml` initializes **only** the three the `.ply` path needs (`glomap gsplat Hierarchical-Localization`) to keep the image small.

### 2. COLMAP must be built from source (FIXED)
Upstream gets the `colmap` binary from a conda env (`colmap=3.10`); Cog is pip-only. The pipeline shells out to the `colmap` CLI (`matches_importer`, `image_registrator`) **and** GLOMAP's `find_package(COLMAP)` needs it, so `cog.yaml` builds **COLMAP 3.10 from source**. This needs Boost `filesystem`/`test`/`iostreams` dev packages that conda otherwise provided. Built with `-DGUI_ENABLED=OFF -DTESTS_ENABLED=OFF` (no Qt5) — but COLMAP/SiftGPU still require **OpenGL + GLEW**, so `libgl1-mesa-dev`, `libglu1-mesa-dev`, `libglew-dev` stay.

### 3. The 57 GB image was auto-disabled by Replicate (FIXED — this is the big one)
The first build was ~57 GB and Replicate auto-disabled the version: *"consistently fails to complete setup"* — it couldn't cold-boot/extract an image that large within the platform's limits. `setup()` itself imports in ~1.4 s, so it was purely size. **Slimmed to ~31.8 GB** and it boots fine (cold ~4-5 min, warm ~1 min). Keep the image small. What was dropped (none on the video→`.ply` path): the `vggt` submodule + its ~6.25 GB of deps, and `open3d`/`depth-anything-3`/`torchaudio`/`decord`/`pycocotools` from `requirements_worker.txt`.

### 4. `vggt` is a hard module-level import (FIXED)
`vid2scene.py` imports `vggt_to_colmap` at module top, which imports the whole `vggt` package — so the pipeline won't import without it. Since `predict.py` only exposes `glomap`/`colmap` (not the `vggt` method), `cog.yaml` `sed`-stubs that import to a clear error and skips the submodule + its deps.

### 5. `transparent-background` needs `flet==0.27.6` pinned (FIXED)
`transparent_background` 1.3.3's `__init__` imports a flet-based GUI module, and only **flet 0.27.x** has the API it expects. An unpinned/newer flet (pulled transitively) breaks `import transparent_background`. So flet is kept pinned even though we don't use its GUI. The InSPyReNet "fast" weight is **pre-baked** into the image so an enabled `remove_background` run doesn't download ~180 MB on a cold container.

### 6. Authentication for push (gotcha)
`cog login --token-stdin` wants a **CLI auth token** (https://replicate.com/auth/token), which needs a browser session. On a headless build box, `docker login r8.im -u <username>` with the **API token** (https://replicate.com/account/api-tokens) as the password works, and `cog push` then uses Docker's stored credentials.

### 7. `predict.py` paths (FIXED)
`setup()` adds the repo root to `sys.path` (so `import vggt`'s stub resolves), adds `/usr/local/bin` to `PATH` (colmap/glomap), and sets `GSPLAT_SCRIPT` (the pipeline raises `ValueError` without it). Output `.ply` resolves at `output_dir/ply/splat.ply`.

## Known Risks
- **GPU arch lock:** kernels are compiled for `TORCH_CUDA_ARCH_LIST=7.5;8.6;8.9`, so the image runs on **T4 (7.5) or L40S (8.9)** only. **A100 (8.0) and H100 (9.0) would NOT load the kernels** — add `8.0`/`9.0` (or `+PTX`) and rebuild to use those.
- **Upstream layout drift:** submodule paths / `Worker_Dockerfile` steps move occasionally — hence the pinned commit. Re-verify against upstream before bumping the pin.
- **Wheel availability:** `pycolmap-cuda12`, `pyceres`, etc. are pinned for **Python 3.10**; changing the Python version may force source builds.
- **CPU bundle adjustment:** apt `libceres-dev` has no CUDA, so GLOMAP's bundle adjustment runs on CPU (works, just slower on big scenes). Build Ceres with CUDA/cuDSS if SfM time becomes a bottleneck.
- **Dropped capabilities:** the `vggt`/`sam3`/`quest` reconstruction methods, open3d visualization, and depth-anything are intentionally not bundled.

## Architecture Notes
- Default path: `extract_frames` (ffmpeg) → `hloc` features + `colmap`/`glomap` SfM → `gsplat` `simple_trainer.py mcmc` → `splat.ply`.
- `.ply` size is governed by Gaussian count (the mcmc densifier grows toward `training_max_num_gaussians`), capped here to stay under 3DStreet's 100 MB ceiling.
- 3DStreet's existing `generateReplicateSplat` flow streams the `.ply` into the user's gallery; the downstream RAD/LOD Cloud Run pipeline optimizes it — nothing past "produce the `.ply`" needs to change.

## Hardware
**L40S** is the recommended Replicate tier — 48 GB VRAM, Ada (**sm_89**, which matches the build's arches), ample headroom for the 600k-Gaussian cap, ~30% cheaper/sec than A100 80GB. **T4** also works (sm_75) but is much slower and 16 GB risks OOM at higher Gaussian counts. **A100/H100 require a recompile** (see Known Risks). Configure the hardware in the Replicate dashboard before `cog push`.

## Attribution
- **Pipeline, models, and tooling: [vid2scene](https://github.com/samuelm2/vid2scene) by Sam M² / [SamuSynth](https://samusynth.com)** — Apache 2.0. This is the project that does all the real work; please credit it. See the [wind-down notice](https://vid2scene.com/winddown/).
- Built on [GLOMAP](https://github.com/colmap/glomap) + [COLMAP](https://github.com/colmap/colmap), [hloc](https://github.com/cvg/Hierarchical-Localization), and [gsplat](https://github.com/nerfstudio-project/gsplat).
- Cog wrapper structure inspired by [`kfarr/sharp-ml-replicate`](https://github.com/kfarr/sharp-ml-replicate) and [`3DStreet/hy-world-2-replicate`](https://github.com/3DStreet/hy-world-2-replicate).
- Wrapper code in this repo: Apache 2.0 (see `LICENSE`).
