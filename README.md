# vid2scene-cog

A [Replicate Cog](https://github.com/replicate/cog) packaging of **[vid2scene](https://github.com/samuelm2/vid2scene)** by **[Sam M² / SamuSynth](https://samusynth.com)** — a **video → 3D Gaussian splat** pipeline (frame extraction → [hloc](https://github.com/cvg/Hierarchical-Localization) + [GLOMAP](https://github.com/colmap/glomap) structure-from-motion → [gsplat](https://github.com/nerfstudio-project/gsplat) training → `.ply`).

The image is **platform-independent**: it's a self-contained container that takes **a video in** and produces **one `.ply` Gaussian splat out**. The same image runs on Replicate, on [Modal](https://modal.com), on a bare GPU box, or anywhere that can run a container with a GPU. This repo wraps **only** vid2scene's standalone `vid2scene_core/` reconstruction pipeline — **none** of the vid2scene SaaS (Django web, Postgres, Redis, blob storage, billing).

> **Why this exists:** vid2scene is [winding down its web operations](https://vid2scene.com/winddown/) (June 13, 2026) and the team generously open-sourced the codebase under **Apache 2.0**. Huge thanks to Sam and the SamuSynth team. This Cog repackages their excellent work so the reconstruction pipeline keeps living on. All the hard parts (the SfM + gsplat pipeline, the tooling, the tuning) are theirs.

Upstream is **pinned** to [`3DStreet/vid2scene@04559366`](https://github.com/3DStreet/vid2scene/commit/04559366ce7b3bb4a2ef261f364379c9761623b6) in `cog.yaml` for reproducible rebuilds (a byte-identical fork of `samuelm2/vid2scene` at the same commit).

---

## Contents
- [The pipeline (platform-independent)](#the-pipeline-platform-independent)
- [Inputs](#inputs) · [Output](#output)
- [Quality vs. cost: the knobs that matter](#quality-vs-cost-the-knobs-that-matter)
- [Hardware & cost (COGS)](#hardware--cost-cogs)
- [Build & push the image](#build--push-the-image)
- [Run it: Replicate](#run-it-replicate) · [Run it: Modal](#run-it-modal)
- [Integrating with a host app (3DStreet)](#integrating-with-a-host-app-3dstreet)
- [Background: why run it off-Replicate](#background-why-run-it-off-replicate)
- [Build issues & fixes](#build-issues-encountered--fixes) · [Known risks](#known-risks) · [Architecture](#architecture-notes) · [Attribution](#attribution)

---

## The pipeline (platform-independent)

Default path: `extract_frames` (ffmpeg) → `hloc` features + `colmap`/`glomap` SfM → `gsplat` `simple_trainer.py mcmc` → `splat.ply`.

| File | Role |
| --- | --- |
| `cog.yaml` | Build recipe: CUDA 12.4, Python 3.10, COLMAP 3.10 built from source, GLOMAP + gsplat + hloc from pinned upstream submodules, slim worker deps. |
| `predict.py` | Cog `Predictor` wrapping `vid2scene_core/vid2scene.py::process_video_to_scene(...)`. The single orchestration entrypoint. |
| `modal_app.py` | Modal harness — reuses the image + `Predictor` for non-preemptible, scale-to-zero execution. See [Run it: Modal](#run-it-modal). |
| `LICENSE` | Apache 2.0 (this wrapper). Upstream vid2scene is also Apache 2.0. |

**One image, many consumers.** `cog.yaml` + `predict.py` build the image; everything else (Replicate, Modal) just *consumes* it. A fix here, rebuilt and pushed once, propagates to every platform.

```
  cog.yaml + predict.py   --cog build-->   r8.im/<owner>/vid2scene   (one image)
                                              /                 \
                                         cog push          Image.from_registry
                                            |                      |
                                        Replicate               Modal
```

## Inputs
| Input | Default | Notes |
| --- | --- | --- |
| `video` | — | A slow, steady **orbit of a static subject** works best. `.mp4`, `.mov` (incl. iPhone HEVC), etc. — ffmpeg extraction is format-agnostic. Aspect ratio is **preserved** (downscaled so the long edge ≤ 1920 px; never cropped). |
| `reconstruction_method` | `glomap` | `glomap` (global SfM) needs no model weights. `colmap` (incremental) is slower, sometimes more robust on hard captures. |
| `target_framecount` | `600` | Frames sampled from the video. **Primary driver of SfM (and total) cost** — see below. |
| `training_num_steps` | `30000` | gsplat training steps. Lower = faster + cheaper, slightly lower quality. |
| `training_max_num_gaussians` | `500000` | **Primary control on detail + `.ply` size** (~164 bytes/Gaussian → 500k ≈ 82 MB, 10M ≈ 1.6 GB). Range 100k–10M. |
| `equirectangular` | `false` | Treat input as 360°/equirectangular video (rig-based SfM + perspective split). |
| `use_background_sphere` | `true` | Add a background sphere for distant/sky content (helps outdoor / 360 captures). |
| `remove_background` | `false` | InSPyReNet background removal per frame — good for isolating a single object; leave **off** for scenes/environments. This may also add to processing time / cost. |
| `apriltag_size_meters` | `0` (off) | If an AprilTag of known physical size (meters) is visible, scales the reconstruction to **real-world units**. |

## Output
A single `.ply` Gaussian splat (`output_dir/ply/splat.ply` upstream). Drop straight into any splat viewer. The gsplat exporter writes position + SH-degree-2 color + opacity/scale/rotation (41 floats = 164 bytes per Gaussian).

---

## Quality vs. cost: the knobs that matter

A job run has **two phases with very different resource profiles**:

```
[ Structure-from-Motion ]      →   [ gsplat training ]
  hloc + glomap/colmap             mcmc densify + optimize
  CPU-bound, GPU ~idle             GPU-bound, low VRAM
  ~2/3 of wall-clock               ~1/3 of wall-clock
```

So **which knob you turn matters more than how far:**

| Knob | SfM phase (the big, CPU cost) | Training phase (GPU) | Detail / `.ply` size |
| --- | --- | --- | --- |
| `target_framecount` | **dominant cost driver** | — | indirect |
| (resolution) | strong | some | yes — *but not currently a user knob; auto-capped to 1920 px long edge* |
| `training_num_steps` | — | linear (15k ≈ half the training) | minor |
| `training_max_num_gaussians` | — | moderate | **yes** |

Takeaways:
- **`target_framecount` is the cost lever.** The CPU SfM phase dominates wall-clock, and it scales with frame count (feature extraction + matching + bundle adjustment). Halving frames meaningfully cuts cost — but very low frame counts can hurt reconstruction *reliability* on sparse captures, not just quality.
- **`training_max_num_gaussians` is the quality / file-size lever.** It barely touches cost (only the smaller GPU phase) but drives visible detail and `.ply` size.
- **`training_num_steps` is a minor speed tweak.**
- You **cannot** make a job dramatically cheaper by only dropping gaussians/steps — those touch only the smaller GPU phase. Real cost differences come from `target_framecount` (and resolution, if exposed).

### Example presets
Illustrative settings bundles (relative cost assumes a fixed GPU; **measure on your platform** — wall-time is very CPU-dependent):

| Preset | `target_framecount` | `training_num_steps` | `training_max_num_gaussians` | Relative cost | Result |
| --- | --- | --- | --- | --- | --- |
| Fast | ~300 | 15000 | 500000 | lowest (fewer frames ↓ SfM, fewer steps ↓ training) | quick, rougher |
| Default | 600 | 30000 | 500000 | baseline | the shipped default |
| High-detail | 600–900 | 30000 | 1–2M | ~2–4× (more frames ↑ SfM + more gaussians ↑ training & size) | visibly sharper, larger `.ply` |

---

## Hardware & cost (COGS)

Measured on a default job (500k gaussians / 30k steps / 600 frames) producing an 82 MB `.ply`:

| Fact | Value |
| --- | --- |
| **Peak VRAM** | **~2.5 GB** (of 46 GB on an L40S) at default settings |
| GPU needed | **L4 (24 GB) is ample; T4 (16 GB) also works.** VRAM is *not* the constraint at default/typical settings — only the 10M-gaussian extreme warrants a re-check. **Pick the GPU by cost, not VRAM.** |
| Wall-clock | **CPU-bound, and CPU-speed-dependent:** the same job ran **~17.6 min** on Replicate's L40S host vs **~41.5 min** on a slower dedicated box. The SfM phase (CPU) dominates. |
| Raw COGS | Replicate L40S: **~$1.00** for a default job (~17.6 min). An L4 on a pay-per-second platform is typically **~$0.50–0.80** depending on the host CPU. |

**GPU architecture lock.** Kernels are compiled for `TORCH_CUDA_ARCH_LIST=7.5;8.6;8.9`, so the image runs on **T4 (7.5), L4 (8.9), or L40S (8.9)** only. **A100 (8.0) and H100 (9.0) will NOT load the kernels** — add `8.0`/`9.0` (or `+PTX`) to the arch list and rebuild to use those.

**Latency note.** GLOMAP's bundle adjustment runs on CPU because apt `libceres-dev` has no CUDA — that's most of the idle-GPU SfM time. Building Ceres with CUDA + cuDSS and rebuilding GLOMAP against it would move bundle adjustment onto the (otherwise idle) GPU and shorten the dominant phase. Highest-leverage perf change; treat as an experiment (verify the speedup for your scene sizes first).

---

## Build & push the image

> **You do NOT need a GPU to build.** The CUDA toolkit cross-compiles for the arches in `TORCH_CUDA_ARCH_LIST` (`7.5;8.6;8.9`). A GPU is only needed to *run* (`cog predict`) or serve. The build is heavy (~31.8 GB image; compiles COLMAP/Ceres/GLOMAP): use **≥32 GB RAM and ≥100 GB free disk**. Clean build is ~20–25 min.

```bash
# Docker + cog on an x86_64 Linux host:
curl -fsSL https://get.docker.com | sh
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog
git clone https://github.com/3DStreet/vid2scene-cog.git && cd vid2scene-cog

cog build -t vid2scene-cog

# Push (see Build Issue #6 — docker login with the API token is the reliable path):
docker login r8.im -u <your-replicate-username>   # paste API token as the password
cog push r8.im/<owner>/vid2scene                  # create the model on replicate.com first
```

Local smoke test (needs a GPU): `cog predict -i video=@orbit.mp4 -i training_num_steps=7000`

---

## Run it: Replicate

Push as above. **Version resolution is automatic** if the host app resolves `latest_version` at runtime, so a re-push goes live without a code change.

```python
import replicate
output = replicate.run(
    "<owner>/vid2scene:<version>",
    input={"video": open("orbit.mp4", "rb"), "target_framecount": 300},
)
# `output` is a URL to the resulting .ply — download it.
```

Configure the GPU tier (L40S/L4/T4 — see [Hardware](#hardware--cost-cogs)) in the Replicate dashboard before `cog push`. **Caveat:** Replicate's *on-demand* tier preempts long private jobs — see [Background](#background-why-run-it-off-replicate). For reliable long-running execution, use Replicate **Deployments** (reserved capacity) or Modal.

## Run it: Modal

`modal_app.py` runs the **same image** on Modal — non-preemptible, scale-to-zero, pay-per-job GPU, with a 24 h time limit. It imports `Predictor` from the image and reuses `setup()` + `process_video_to_scene`, so there is no forked logic.

```bash
pip install modal && modal token new
# one-time: store r8.im pull creds so Modal can pull the private image
modal secret create r8im-pull REGISTRY_USERNAME=<user> REGISTRY_PASSWORD=<replicate-token>
# optional: GCS upload + completion webhook config
modal secret create vid2scene-io GCS_BUCKET=<bucket> WEBHOOK_URL=<callback> ...
modal deploy modal_app.py

# local test / calibration (synchronous):
modal run modal_app.py --video-url <url> --gaussians 500000 --steps 30000
```

Production invocation is async: `POST` to the `enqueue` web endpoint → it `.spawn()`s the job and returns a `call_id` → the job uploads the `.ply` to GCS and POSTs a completion webhook (or the caller polls `FunctionCall.from_id(call_id)`). Tunables (GPU, CPU cores, memory, timeout) are env-vars at the top of `modal_app.py`.

---

## Integrating with a host app (3DStreet)

[3DStreet](https://3dstreet.app) drives this through its existing job system — **the model is just a compute backend behind that queue**, not a new queue. To add a backend (e.g. Modal alongside Replicate), implement three responsibilities:

| Responsibility | Replicate backend | Modal backend |
| --- | --- | --- |
| **Submit** | `POST …/predictions` → prediction id | `POST` the Modal `enqueue` endpoint → `call_id` |
| **Track** | store prediction id on the job row | store `call_id` on the job row |
| **Complete** | poll status / webhook → fetch output URL | webhook from the job (or poll the `FunctionCall`) → read `.ply` from GCS |

Flow: host enqueues → backend runs (~tens of minutes) → `.ply` lands in storage → host streams it into the user's gallery, and any downstream optimization (e.g. 3DStreet's RAD/LOD step) runs as usual. **Nothing past "produce the `.ply`" changes.** Quality/cost presets (the [knobs above](#quality-vs-cost-the-knobs-that-matter)) are chosen host-side and passed as inputs at submit time.

---

## Background: why run it off-Replicate

These jobs are long (tens of minutes) and Replicate's **on-demand** tier **preempts** them — surfacing as `code: PA` / `Director … (E8765)`, with logs wiped on failure so it's indistinguishable from OOM via the API.

This was diagnosed with two tests:
- **Head-to-head on Replicate:** the last-known-good image *and* the current image both failed identically — ruling out a code regression.
- **Dedicated, non-preemptible GPU:** the **exact same image** ran to **success** (valid 500k-gaussian `.ply`), with **peak VRAM only ~2.5 GB**.

**Conclusion: the failures are platform preemption, not OOM or a bug.** A later Replicate retry succeeding (intermittently) is itself a preemption signature. The fix is non-preemptible execution — Replicate **Deployments** (reserved, always-on cost) or **Modal** (pay-per-job, scale-to-zero). Because peak VRAM is tiny, an **L4** is the right cost/availability target.

---

## Build Issues Encountered & Fixes

Here are various modifications Claude and I made to make this work. These include disabling features that may be available in the original repo for our use case.

### 1. Submodules live at the repo top level (FIXED)
GLOMAP, gsplat, Hierarchical-Localization, vggt, etc. are top-level submodules of vid2scene. `cog.yaml` initializes **only** the three the `.ply` path needs (`glomap gsplat Hierarchical-Localization`) to keep the image small.

### 2. COLMAP must be built from source (FIXED)
Upstream gets `colmap` from a conda env (`colmap=3.10`); Cog is pip-only. The pipeline shells out to the `colmap` CLI **and** GLOMAP's `find_package(COLMAP)` needs it, so `cog.yaml` builds **COLMAP 3.10 from source** with `-DGUI_ENABLED=OFF -DTESTS_ENABLED=OFF` (no Qt5) — but SiftGPU still needs **OpenGL + GLEW**, so those `-dev` libs stay.

### 3. The 57 GB image was auto-disabled by Replicate (FIXED — the big one)
The first build was ~57 GB and Replicate auto-disabled it: *"consistently fails to complete setup"* — it couldn't cold-boot an image that large. **Slimmed to ~31.8 GB** (dropped the `vggt` submodule + ~6.25 GB deps, and `open3d`/`depth-anything-3`/`torchaudio`/`decord`/`pycocotools`). Keep the image small.

### 4. `vggt` is a hard module-level import (FIXED)
`vid2scene.py` imports `vggt_to_colmap` at module top. Since `predict.py` only exposes `glomap`/`colmap`, `cog.yaml` `sed`-stubs that import to a clear error and skips the submodule + its deps.

### 5. `transparent-background` needs `flet==0.27.6` pinned (FIXED)
`transparent_background` 1.3.3's `__init__` imports a flet GUI module; only flet 0.27.x has the API it expects. So flet stays pinned even though its GUI is unused. The InSPyReNet "fast" weight is **pre-baked** into the image.

### 6. Authentication for push (gotcha)
`cog login --token-stdin` wants a **CLI auth token** (needs a browser). On a headless box, `docker login r8.im -u <username>` with the **API token** as the password works, and `cog push` reuses Docker's stored credentials.

### 7. `predict.py` paths (FIXED)
`setup()` adds the repo root to `sys.path` (so the `vggt` stub resolves), adds `/usr/local/bin` to `PATH` (colmap/glomap), and sets `GSPLAT_SCRIPT`. Output `.ply` resolves at `output_dir/ply/splat.ply`.

## Known Risks
- **GPU arch lock:** kernels compiled for `7.5;8.6;8.9` — A100/H100 need a recompile (see [Hardware](#hardware--cost-cogs)).
- **Upstream layout drift:** submodule paths / `Worker_Dockerfile` steps move occasionally — hence the pinned commit. Re-verify before bumping the pin.
- **Wheel availability:** `pycolmap-cuda12`, `pyceres`, etc. are pinned for **Python 3.10**; changing Python may force source builds.
- **CPU bundle adjustment:** apt `libceres-dev` has no CUDA, so GLOMAP's BA runs on CPU (works, just slower). Build Ceres with CUDA/cuDSS if SfM time becomes a bottleneck.
- **Dropped capabilities:** `vggt`/`sam3`/`quest` methods, open3d visualization, and depth-anything are intentionally not bundled.

## Architecture Notes
- `.ply` size is governed by Gaussian count (the mcmc densifier grows toward `training_max_num_gaussians`).
- The image is the unit of reproducibility: pinned upstream commit + pinned CUDA arches + pinned wheels. Rebuild + push once → all consumers (Replicate, Modal) update.

## Attribution
- **Pipeline, models, and tooling: [vid2scene](https://github.com/samuelm2/vid2scene) by Sam M² / [SamuSynth](https://samusynth.com)** — Apache 2.0. This is the project that does all the real work; please credit it. See the [wind-down notice](https://vid2scene.com/winddown/).
- Built on [GLOMAP](https://github.com/colmap/glomap) + [COLMAP](https://github.com/colmap/colmap), [hloc](https://github.com/cvg/Hierarchical-Localization), and [gsplat](https://github.com/nerfstudio-project/gsplat).
- Cog wrapper structure inspired by [`kfarr/sharp-ml-replicate`](https://github.com/kfarr/sharp-ml-replicate) and [`3DStreet/hy-world-2-replicate`](https://github.com/3DStreet/hy-world-2-replicate).
- Wrapper code in this repo: Apache 2.0 (see `LICENSE`).
