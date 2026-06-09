"""
Replicate Cog predictor for the vid2scene video -> Gaussian-splat pipeline.

This is a thin wrapper around the upstream STANDALONE pipeline
(`vid2scene_core/vid2scene.py::process_video_to_scene`). It deliberately uses
*none* of the vid2scene SaaS (web, DB, queue, billing) — 3DStreet provides the
queue (generationJobs), storage (Firebase/GCS), auth, and tokens already. The
contract here is intentionally tiny:

    input : a video file (+ a few reconstruction knobs)
    output: a single .ply Gaussian splat

3DStreet's existing `generateReplicateSplat` flow streams that .ply into the
user's gallery, and the downstream RAD/LOD Cloud Run pipeline
(`onSplatAssetCreated`) optimizes it — so nothing past "produce the .ply" needs
to change. See docs/vid2scene-video-to-splat.md.

The upstream entrypoint signature (verified against samuelm2/vid2scene):

    process_video_to_scene(
        video_path=None, image_dir=None, output_dir=None, sfm_dir=None,
        target_framecount=600, preview_data_handler=None,
        remove_background_from_images=False, equirectangular=False,
        use_background_sphere=False, apply_pilgram_filter_name=None,
        training_max_num_gaussians=1_000_000, training_num_steps=30_000,
        kill_check=None, reconstruction_method='glomap',
        apriltag_size_meters=None, mock=False, quest_project_dir=None,
    ) -> "path to output .ply, or None if terminated"

NOTE: untested in this sandbox (no GPU/CUDA). Validate on a GPU build box; the
two things most likely to need a tweak after a real build are (a) the exact
location of the produced .ply and (b) PATH/sys.path so the compiled binaries
(glomap, etc.) and the sibling Python modules resolve. Both are isolated below.
"""

import os
import sys
import shutil
import tempfile
import subprocess

from cog import BasePredictor, Input, Path

# Where cog.yaml clones the upstream repo. Two paths must be importable:
#   - VID2SCENE_CORE: the pipeline package itself (its modules use sibling imports
#     like `import extract_frames`).
#   - VID2SCENE_REPO (repo root): so module-level `import vggt` (pulled in by
#     vid2scene.py -> vggt_to_colmap) resolves to the vendored vggt submodule.
# Both must also be on PATH so the compiled binaries the pipeline shells out to
# (colmap, glomap) are found.
VID2SCENE_REPO = os.environ.get("VID2SCENE_REPO", "/src/vid2scene")
VID2SCENE_CORE = os.path.join(VID2SCENE_REPO, "vid2scene_core")

# The gsplat trainer the pipeline launches as a subprocess (run_gsplat reads
# GSPLAT_SCRIPT). Matches the upstream Worker_Dockerfile's GSPLAT_SCRIPT env.
GSPLAT_SCRIPT = os.environ.get(
    "GSPLAT_SCRIPT", os.path.join(VID2SCENE_REPO, "gsplat", "examples", "simple_trainer.py")
)


class Predictor(BasePredictor):
    def setup(self):
        # repo root first so `import vggt` (vendored submodule) resolves, then the
        # core package for the sibling imports inside the pipeline.
        for p in (VID2SCENE_REPO, VID2SCENE_CORE):
            if p not in sys.path:
                sys.path.insert(0, p)
        # Make locally-installed binaries (colmap/glomap `ninja install` target)
        # discoverable; /usr/local/bin is the default prefix.
        os.environ["PATH"] = f"/usr/local/bin:{os.environ.get('PATH', '')}"
        # The pipeline requires GSPLAT_SCRIPT to be set (raises ValueError if not).
        os.environ["GSPLAT_SCRIPT"] = GSPLAT_SCRIPT
        # Import lazily so an import error surfaces clearly at predict time.
        from vid2scene import process_video_to_scene  # noqa: F401

        self._process = process_video_to_scene

    def predict(
        self,
        # NOTE: keep every `description` a SINGLE-LINE string. coglet captures the
        # raw source of multi-line implicitly-concatenated literals (quotes +
        # newlines + indentation) into the schema, which shows up garbled in the
        # Replicate UI. One line each = clean labels.
        video: Path = Input(
            description="Source video. A slow, steady orbit around a static subject works best.",
        ),
        reconstruction_method: str = Input(
            description="Structure-from-Motion method. glomap (default) needs no model weights.",
            choices=["glomap", "colmap"],
            default="glomap",
        ),
        target_framecount: int = Input(
            description="Target number of frames to sample from the video.",
            default=600,
            ge=30,
            le=2000,
        ),
        training_num_steps: int = Input(
            description="gsplat training steps. Fewer is faster but lower quality.",
            default=30000,
            ge=2000,
            le=30000,
        ),
        training_max_num_gaussians: int = Input(
            description="Cap on the number of Gaussians: the primary control on output detail and .ply size (~164 bytes/Gaussian, so 5M is ~820 MB). Higher means more detail and a larger file. The 500k default is a sensible mid-range; raise it for large or complex scenes. File-size gating per user tier is enforced by 3DStreet on save.",
            default=500000,
            ge=100000,
            le=10000000,
        ),
        equirectangular: bool = Input(
            description="Treat the input as 360/equirectangular video.",
            default=False,
        ),
        use_background_sphere: bool = Input(
            description="Add a background sphere for distant/sky content. Helps outdoor or 360 captures where the background is far away.",
            default=True,
        ),
        remove_background: bool = Input(
            description="Remove the background from each frame before reconstruction (InSPyReNet). Good for isolating a single object; leave off for scenes/environments where you want the surroundings.",
            default=False,
        ),
        apriltag_size_meters: float = Input(
            description="Optional. If an AprilTag of known physical size (in meters) is visible in the video, set it to scale the reconstruction to real-world units. Leave at 0 to skip AprilTag detection entirely.",
            default=0.0,
            ge=0.0,
            le=10.0,
        ),
    ) -> Path:
        work_dir = tempfile.mkdtemp(prefix="vid2scene_")
        out_dir = os.path.join(work_dir, "out")
        os.makedirs(out_dir, exist_ok=True)

        # The pipeline uses sibling imports + relative tooling, so run it with
        # vid2scene_core as the working directory.
        prev_cwd = os.getcwd()
        os.chdir(VID2SCENE_CORE)
        try:
            ply_path = self._process(
                video_path=str(video),
                output_dir=out_dir,
                target_framecount=target_framecount,
                equirectangular=equirectangular,
                use_background_sphere=use_background_sphere,
                remove_background_from_images=remove_background,
                # 0 (the optional default) means "no AprilTag" — pass None so the
                # pipeline skips detection (it treats None/<=0 as disabled).
                apriltag_size_meters=(apriltag_size_meters or None),
                training_max_num_gaussians=training_max_num_gaussians,
                training_num_steps=training_num_steps,
                reconstruction_method=reconstruction_method,
            )
        finally:
            os.chdir(prev_cwd)

        # The function returns the .ply path; fall back to the documented default
        # location if a build returns None but wrote the file anyway.
        candidate = ply_path or os.path.join(out_dir, "ply", "splat.ply")
        if not candidate or not os.path.exists(candidate):
            # Last resort: find any .ply under the output dir.
            found = subprocess.run(
                ["find", out_dir, "-name", "*.ply"],
                capture_output=True,
                text=True,
            ).stdout.split()
            if not found:
                raise RuntimeError(
                    "vid2scene produced no .ply — check the worker logs for SfM "
                    "or training failures."
                )
            candidate = found[0]

        # Copy the result out of the temp dir so Cog can return it after cleanup.
        result = Path(os.path.join(work_dir, "splat.ply"))
        if str(result) != candidate:
            shutil.copyfile(candidate, result)
        return result
