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

The upstream entrypoint signature (3DStreet/vid2scene fork, cog-phase2 — adds
max_resolution / stop_after_sfm / normalize_override and the direct .insv
dual-fisheye path over samuelm2 upstream):

    process_video_to_scene(
        video_path=None, image_dir=None, output_dir=None, sfm_dir=None,
        target_framecount=600, preview_data_handler=None,
        remove_background_from_images=False, equirectangular=False,
        insv_fisheye=False, insv_lens_fov=None, insv_calibration=None,
        insv_no_factory_calibration=False,
        use_background_sphere=False, apply_pilgram_filter_name=None,
        training_max_num_gaussians=1_000_000, training_num_steps=30_000,
        kill_check=None, reconstruction_method='glomap',
        apriltag_size_meters=None, mock=False, quest_project_dir=None,
        max_resolution=1920, stop_after_sfm=False, normalize_override=None,
    ) -> "path to output .ply (SfM dir if stop_after_sfm), or None if terminated"

NOTE: untested in this sandbox (no GPU/CUDA). Validate on a GPU build box; the
two things most likely to need a tweak after a real build are (a) the exact
location of the produced .ply and (b) PATH/sys.path so the compiled binaries
(glomap, etc.) and the sibling Python modules resolve. Both are isolated below.
"""

import os
import sys
import shutil
import tempfile
import threading
import time
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


# ---------------------------------------------------------------------------
# Debug instrumentation (the `debug` input). The container can be RAM-killed
# mid-run with no traceback (the suspected Replicate failure mode), so the
# strategy is to PRINT a compact memory/disk line every few seconds: log lines
# survive the kill and are retrievable via the predictions API afterwards.
# Memory is read from the cgroup (container-wide, so COLMAP/glomap/gsplat
# subprocesses are included), with /proc fallbacks.
# ---------------------------------------------------------------------------

def _read_first(*paths):
    for p in paths:
        try:
            with open(p) as f:
                return f.read().strip()
        except OSError:
            continue
    return None


def _gib(n_bytes):
    return f"{n_bytes / 2**30:.2f}"


def _cgroup_mem_bytes():
    """(current, limit) in bytes; either may be None (no cgroup / unlimited)."""
    cur = _read_first(
        "/sys/fs/cgroup/memory.current",  # v2
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",  # v1
    )
    lim = _read_first(
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    cur = int(cur) if cur and cur.isdigit() else None
    # v2 spells "no limit" as the literal string "max"; v1 as a huge number.
    lim = int(lim) if lim and lim.isdigit() and int(lim) < 2**60 else None
    return cur, lim


def _meminfo_kb(field):
    info = _read_first("/proc/meminfo") or ""
    for line in info.splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    return None


def _tmp_mount_line():
    """The /proc/mounts entry covering /tmp (longest matching mount point)."""
    mounts = _read_first("/proc/mounts") or ""
    best = None
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) >= 3 and ("/tmp" + "/").startswith(parts[1].rstrip("/") + "/"):
            if best is None or len(parts[1]) > len(best[1]):
                best = parts
    return " ".join(best[:4]) if best else "unknown"


def _debug_startup_dump():
    cur, lim = _cgroup_mem_bytes()
    total_kb = _meminfo_kb("MemTotal")
    du = shutil.disk_usage("/tmp")
    print(
        "[v2s-debug] startup"
        f" cgroup_mem_limit={_gib(lim) + 'GiB' if lim else 'none'}"
        f" host_mem_total={_gib(total_kb * 1024) + 'GiB' if total_kb else '?'}"
        f" cpus={os.cpu_count()}"
        f" tmp_disk={_gib(du.used)}/{_gib(du.total)}GiB"
        f" tmp_mount=[{_tmp_mount_line()}]",
        flush=True,
    )


def _start_debug_sampler(interval_s=15):
    """Print a memory/disk line every interval_s; returns a stop() callable."""
    _debug_startup_dump()
    stop_event = threading.Event()
    t0 = time.monotonic()

    def _sample():
        while not stop_event.wait(interval_s):
            cur, lim = _cgroup_mem_bytes()
            avail_kb = _meminfo_kb("MemAvailable")
            du = shutil.disk_usage("/tmp")
            print(
                f"[v2s-debug] t={time.monotonic() - t0:.0f}s"
                f" cgroup_mem={_gib(cur) if cur else '?'}/{_gib(lim) if lim else 'none'}GiB"
                f" mem_avail={_gib(avail_kb * 1024) if avail_kb else '?'}GiB"
                f" tmp_disk={_gib(du.used)}/{_gib(du.total)}GiB",
                flush=True,
            )

    thread = threading.Thread(target=_sample, daemon=True)
    thread.start()
    return stop_event.set


class Predictor(BasePredictor):
    def setup(self):
        # repo root first so `import vggt` (vendored submodule) resolves, then the
        # core package for the sibling imports inside the pipeline.
        for p in (VID2SCENE_REPO, VID2SCENE_CORE):
            if p not in sys.path:
                sys.path.insert(0, p)
        # Make locally-installed binaries (colmap/glomap `ninja install` target)
        # discoverable; /usr/local/bin is the default prefix. APPEND, don't prepend:
        # /usr/local/bin also contains a bare `python` that would shadow the pyenv
        # interpreter holding all the pip deps, so the gsplat subprocess (`python
        # simple_trainer.py`) would run under the wrong python (ModuleNotFoundError:
        # imageio). Appending keeps pyenv's python first while still finding colmap.
        os.environ["PATH"] = f"{os.environ.get('PATH', '')}:/usr/local/bin"
        # The pipeline requires GSPLAT_SCRIPT to be set (raises ValueError if not).
        os.environ["GSPLAT_SCRIPT"] = GSPLAT_SCRIPT
        # The gsplat trainer's stderr is now piped into the logs; without this
        # its tqdm bars would blow Replicate's 256 KiB log cap.
        os.environ["TQDM_DISABLE"] = "1"
        # Import lazily so an import error surfaces clearly at predict time.
        from vid2scene import process_video_to_scene  # noqa: F401

        self._process = process_video_to_scene

    # -- stage API -----------------------------------------------------------
    # predict() runs both stages back to back (the Replicate contract). The
    # Modal harness calls them individually so SfM (CPU-bound, GPU idle) and
    # training (GPU-bound, ~2 cores) can run on differently-shaped machines
    # with the SfM directory handed off via a shared volume.

    def run_sfm(
        self,
        *,
        video_path: str,
        out_dir: str,
        target_framecount: int = 600,
        resolution: int = 1920,
        equirectangular: bool = False,
        insv_fisheye: bool = False,
        use_background_sphere: bool = True,
        remove_background: bool = False,
        apriltag_size_meters: float | None = None,
        reconstruction_method: str = "glomap",
    ) -> str:
        """Frames -> SfM -> scene prep (sphere/filters/scale). Returns the SfM
        artifacts directory, ready to be consumed by run_train()."""
        return self._run_pipeline(
            video_path=video_path,
            output_dir=out_dir,
            target_framecount=target_framecount,
            max_resolution=resolution,
            equirectangular=equirectangular,
            insv_fisheye=insv_fisheye,
            use_background_sphere=use_background_sphere,
            remove_background_from_images=remove_background,
            apriltag_size_meters=apriltag_size_meters,
            reconstruction_method=reconstruction_method,
            stop_after_sfm=True,
        )

    def run_train(
        self,
        *,
        sfm_dir: str,
        out_dir: str,
        training_max_num_gaussians: int = 500000,
        training_num_steps: int = 30000,
        normalize: bool = True,
    ) -> str:
        """gsplat training against an existing SfM directory; returns the .ply
        path. Pass normalize=False iff the SfM stage applied AprilTag scaling
        (real-world units must be preserved)."""
        return self._run_pipeline(
            sfm_dir=sfm_dir,
            output_dir=out_dir,
            training_max_num_gaussians=training_max_num_gaussians,
            training_num_steps=training_num_steps,
            # The SfM stage already applied background sphere / bg removal /
            # AprilTag scale — passing those flags again would apply them twice.
            normalize_override=normalize,
        )

    def _run_pipeline(self, **kwargs):
        # The pipeline uses sibling imports + relative tooling, so run it with
        # vid2scene_core as the working directory.
        prev_cwd = os.getcwd()
        os.chdir(VID2SCENE_CORE)
        try:
            return self._process(**kwargs)
        finally:
            os.chdir(prev_cwd)

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
            description="Structure-from-Motion method. glomap (default): fast global SfM, recommended. colmap: slower incremental, sometimes more robust on difficult captures.",
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
            description="Cap on the number of Gaussians: the primary control on output detail and .ply size (~164 bytes/Gaussian, so 5M is ~820 MB). Higher means more detail and a larger file. The 500k default is a sensible mid-range; raise it for large or complex scenes. Warning: a high cap increases training time and may cause the job to time out depending on machine capacity.",
            default=500000,
            ge=100000,
            le=10000000,
        ),
        equirectangular: bool = Input(
            description="Treat the input as 360/equirectangular video.",
            default=False,
        ),
        insv_fisheye: bool = Input(
            description="Treat the input as a raw Insta360 .insv recording and reconstruct directly from its dual fisheye streams (better ground/sky detail than a pre-stitched equirectangular). Required for .insv through this wrapper: the staged input file loses its extension, so the pipeline's auto-detection never fires.",
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
        resolution: int = Input(
            description="Maximum long-edge resolution (pixels) for extracted frames; the video is downscaled to this, never upscaled. Lower is faster and cheaper at reduced detail.",
            default=1920,
            ge=512,
            le=3840,
        ),
        debug: bool = Input(
            description="Log container memory/disk usage every 15s plus resource limits at startup. For diagnosing failed or killed jobs; adds noise to logs.",
            default=False,
        ),
    ) -> Path:
        work_dir = tempfile.mkdtemp(prefix="vid2scene_")
        out_dir = os.path.join(work_dir, "out")
        os.makedirs(out_dir, exist_ok=True)

        stop_debug = _start_debug_sampler() if debug else None
        try:
            sfm_dir = self.run_sfm(
                video_path=str(video),
                out_dir=out_dir,
                target_framecount=target_framecount,
                resolution=resolution,
                equirectangular=equirectangular,
                insv_fisheye=insv_fisheye,
                use_background_sphere=use_background_sphere,
                remove_background=remove_background,
                # 0 (the optional default) means "no AprilTag" — pass None so the
                # pipeline skips detection (it treats None/<=0 as disabled).
                apriltag_size_meters=(apriltag_size_meters or None),
                reconstruction_method=reconstruction_method,
            )
            ply_path = self.run_train(
                sfm_dir=sfm_dir,
                out_dir=out_dir,
                training_max_num_gaussians=training_max_num_gaussians,
                training_num_steps=training_num_steps,
                # AprilTag scaling fixes real-world units; don't renormalize.
                normalize=not apriltag_size_meters,
            )
        finally:
            if stop_debug:
                stop_debug()

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
