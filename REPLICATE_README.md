<!--
  This is the SHORT readme intended to be pasted into the Replicate model page
  (replicate.com/kfarr/vid2scene/settings -> Readme). Keep it short and link out
  to GitHub so there's a single source of truth (the repo) and minimal drift.
  The full docs live in README.md in the repo.
-->

# vid2scene — video → 3D Gaussian splat

Turns a short video (a slow orbit of a subject) into a single `.ply` Gaussian splat.

This is a [Cog](https://github.com/replicate/cog) packaging of **[vid2scene](https://github.com/samuelm2/vid2scene) by Sam M² / [SamuSynth](https://samusynth.com)** — frame extraction → [hloc](https://github.com/cvg/Hierarchical-Localization) + [GLOMAP](https://github.com/colmap/glomap) SfM → [gsplat](https://github.com/nerfstudio-project/gsplat) training. All the real work is theirs; vid2scene is open source under Apache 2.0 ([wind-down notice](https://vid2scene.com/winddown/)). This just makes the reconstruction pipeline callable as a model.

**Source, full docs, and build/deploy instructions:** https://github.com/3DStreet/vid2scene-cog

## Usage
Upload a video and (optionally) tune:
- `target_framecount` — frames sampled from the video
- `training_num_steps` — gsplat steps (quality vs. speed)
- `training_max_num_gaussians` — caps Gaussian count / `.ply` size & detail (up to 10M; ~164 B each)
- `reconstruction_method`, `equirectangular`, `use_background_sphere`, `remove_background`, `apriltag_size_meters` (real-world scale)

Output is one `.ply` — drop it into [3DStreet](https://3dstreet.org), [Spark](https://github.com/sparkjsdev/spark), or any splat viewer.

Runs on **L40S**. Built for **[3DStreet](https://3dstreet.org)** — the easiest way to turn video into splats without running this yourself. Provided as-is under Apache 2.0, with no warranty or guarantee of updates; fork and self-host if you'd rather run your own.
