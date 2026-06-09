"""
Modal deployment for the vid2scene video -> Gaussian-splat pipeline.

WHY THIS EXISTS
---------------
Replicate's on-demand tier often preempts these long (~40 min) private jobs — 
predictions may fail with `code: PA` / `Director ... E8765`. A controlled test
on a dedicated, non-preemptible L40S ran the *exact same image* to success
(peak VRAM only ~2.5 GB), proving the failures are platform preemption, not OOM
or a code bug. Modal gives us non-preemptible, scale-to-zero, pay-per-job GPU
execution with a 24h time limit (vs Cloud Run's hard 60-min cap, which the
10M-gaussian top tier would blow past). See README.md ("Background: why run
it off-Replicate").

THE KEY ARCHITECTURE: ONE IMAGE, TWO CONSUMERS
----------------------------------------------
The container image `r8.im/kfarr/vid2scene` (built by cog.yaml on Hetzner) is the
single source of truth — it bakes in the upstream pipeline, the CUDA binaries
(colmap/glomap/gsplat), the deps, and predict.py. Replicate consumes it via
`cog push`; Modal consumes the SAME image via `Image.from_registry(...)`. So any
fix you make in cog.yaml/predict.py — rebuild + push once — propagates to BOTH
platforms. This file is just the Modal-side harness (the analog of predict.py);
it imports and reuses the existing Predictor so there is no logic duplication.

    cog.yaml + predict.py  --cog build-->  r8.im/kfarr/vid2scene
                                              /                \
                                       cog push            from_registry
                                          |                      |
                                      Replicate               this file

DEPLOY
------
    pip install modal && modal token new
    # one-time: store the r8.im pull creds so Modal can pull the private image
    modal secret create r8im-pull REGISTRY_USERNAME=kfarr REGISTRY_PASSWORD=<replicate-token>
    # (optional) GCS creds for uploading the .ply, + a webhook secret
    modal secret create vid2scene-io GCS_BUCKET=<bucket> WEBHOOK_URL=<3dstreet-callback> ...
    modal deploy modal_app.py

INVOKE  (see "Integrating with a host app" in README.md)
------
Production path is async: 3DStreet's Cloud Task POSTs to the `enqueue` web
endpoint, which `.spawn()`s the job and returns a call id; the job uploads the
.ply to GCS and POSTs a webhook back to 3DStreet's reconciler on completion.
A synchronous `.remote()` path is included for local testing / calibration.
"""

import os
import sys
import tempfile
import urllib.request

import modal

# --- The image: the cog-built artifact, pulled as-is -------------------------
# Anything baked by cog.yaml (pipeline, CUDA binaries, weights, predict.py) is
# already here. We only add the few host-side libs THIS harness needs.
IMAGE_REF = os.environ.get("VID2SCENE_IMAGE", "r8.im/kfarr/vid2scene")

image = (
    modal.Image.from_registry(
        IMAGE_REF,
        # The cog image ships its own python 3.10 (pyenv). add_python lets Modal
        # inject a compatible interpreter for its client; keep it matched to the
        # image's 3.10 to avoid ABI surprises with the baked CUDA extensions.
        add_python="3.10",
        secret=modal.Secret.from_name("r8im-pull"),
    )
    # Host-side helpers for I/O glue (the pipeline itself needs none of these).
    # fastapi is required by the @modal.fastapi_endpoint `enqueue` route.
    .pip_install("google-cloud-storage", "requests", "fastapi[standard]")
    # predict.py lives at the cog working dir (/src) in the image; make it import-
    # able so we can reuse Predictor.setup() + the pipeline entrypoint verbatim.
    .env({"PYTHONPATH": "/src"})
)

app = modal.App("vid2scene")

# Tunables. SfM (the ~27-min CPU-bound stage) dominates wall-clock, so give it
# real cores; gsplat needs the GPU but barely any VRAM (2.5 GB measured).
GPU = os.environ.get("VID2SCENE_GPU", "L4")        # L4/24GB is ample; T4 also fits
CPU = float(os.environ.get("VID2SCENE_CPU", "16"))  # physical cores -> faster SfM
MEMORY_MB = int(os.environ.get("VID2SCENE_MEM_MB", "32768"))
TIMEOUT_S = int(os.environ.get("VID2SCENE_TIMEOUT_S", "7200"))  # 2h: covers max tier


@app.cls(
    image=image,
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MB,
    timeout=TIMEOUT_S,
    # Bursty consumer workload: scale to zero, but keep a short keep-warm window
    # so back-to-back jobs reuse a hot container (skips re-running setup()).
    scaledown_window=120,
    secrets=[modal.Secret.from_name("vid2scene-io")],
)
class Vid2Scene:
    @modal.enter()
    def setup(self):
        # Reuse the EXACT Replicate setup: sys.path/env wiring + lazy import of
        # process_video_to_scene. No logic forked from predict.py.
        if "/src" not in sys.path:
            sys.path.insert(0, "/src")
        from predict import Predictor

        self._predictor = Predictor()
        self._predictor.setup()
        self._process = self._predictor._process

    @modal.method()
    def run(
        self,
        video_url: str,
        *,
        job_id: str | None = None,
        reconstruction_method: str = "glomap",
        target_framecount: int = 600,
        training_num_steps: int = 30000,
        training_max_num_gaussians: int = 500000,
        equirectangular: bool = False,
        use_background_sphere: bool = True,
        remove_background: bool = False,
        apriltag_size_meters: float = 0.0,
    ) -> dict:
        """Run one job. Returns {job_id, gcs_uri|None, size_bytes}. Uploads the
        .ply to GCS and (if configured) POSTs a completion webhook."""
        work_dir = tempfile.mkdtemp(prefix="vid2scene_")
        out_dir = os.path.join(work_dir, "out")
        os.makedirs(out_dir, exist_ok=True)

        # 1. fetch the input video (3DStreet passes a signed GCS/Firebase URL)
        video_path = os.path.join(work_dir, "input")
        urllib.request.urlretrieve(video_url, video_path)

        # 2. run the pipeline (same entrypoint & cwd contract as predict.py)
        prev_cwd = os.getcwd()
        os.chdir("/src/vid2scene/vid2scene_core")
        try:
            ply_path = self._process(
                video_path=video_path,
                output_dir=out_dir,
                target_framecount=target_framecount,
                equirectangular=equirectangular,
                use_background_sphere=use_background_sphere,
                remove_background_from_images=remove_background,
                apriltag_size_meters=(apriltag_size_meters or None),
                training_max_num_gaussians=training_max_num_gaussians,
                training_num_steps=training_num_steps,
                reconstruction_method=reconstruction_method,
            )
        finally:
            os.chdir(prev_cwd)

        candidate = ply_path or os.path.join(out_dir, "ply", "splat.ply")
        if not candidate or not os.path.exists(candidate):
            import glob

            found = glob.glob(os.path.join(out_dir, "**", "*.ply"), recursive=True)
            if not found:
                raise RuntimeError("vid2scene produced no .ply — check job logs.")
            candidate = found[0]

        size = os.path.getsize(candidate)

        # 3. ship the result. Default: upload to GCS so 3DStreet streams it into
        #    the gallery (its onSplatAssetCreated RAD/LOD step takes it from there).
        gcs_uri = _upload_to_gcs(candidate, job_id) if os.environ.get("GCS_BUCKET") else None

        result = {"job_id": job_id, "gcs_uri": gcs_uri, "size_bytes": size}

        # 4. tell 3DStreet's reconciler we're done (alternative to it polling).
        _post_webhook({"status": "succeeded", **result})
        return result


def _upload_to_gcs(local_path: str, job_id: str | None) -> str:
    from google.cloud import storage

    bucket_name = os.environ["GCS_BUCKET"]
    prefix = os.environ.get("GCS_PREFIX", "vid2scene")
    name = f"{prefix}/{job_id or os.path.basename(local_path)}.ply"
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(name)
    blob.upload_from_filename(local_path, content_type="application/octet-stream")
    return f"gs://{bucket_name}/{name}"


def _post_webhook(payload: dict) -> None:
    url = os.environ.get("WEBHOOK_URL")
    if not url:
        return
    import json

    import requests

    headers = {"Content-Type": "application/json"}
    if os.environ.get("WEBHOOK_SECRET"):
        headers["Authorization"] = f"Bearer {os.environ['WEBHOOK_SECRET']}"
    try:
        requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
    except Exception as e:  # never fail the job because the callback flaked
        print(f"WARN: webhook POST failed: {e}")


# --- Async enqueue endpoint: this is what 3DStreet's Cloud Task hits ----------
# It spawns the job and returns immediately with a call id. 3DStreet stores that
# id on the generationJobs row; the reconciler either waits for the webhook above
# or polls modal.FunctionCall.from_id(call_id).get(timeout=0). This is the
# Modal-side of the "compute backend behind the host's queue" — see README.md.
@app.function(image=image, secrets=[modal.Secret.from_name("vid2scene-io")])
@modal.fastapi_endpoint(method="POST")
def enqueue(body: dict) -> dict:
    """POST {video_url, job_id, ...knobs} -> {call_id}. Fire-and-forget."""
    if os.environ.get("ENQUEUE_SECRET"):
        # NOTE: wire real request-auth here (shared secret / OIDC) before prod.
        pass
    video_url = body.pop("video_url")
    call = Vid2Scene().run.spawn(video_url, **body)
    return {"call_id": call.object_id}


# --- Local test / calibration: `modal run modal_app.py --video-url <URL>` -----
@app.local_entrypoint()
def main(video_url: str, gaussians: int = 500000, steps: int = 30000):
    res = Vid2Scene().run.remote(
        video_url,
        job_id="local-test",
        training_max_num_gaussians=gaussians,
        training_num_steps=steps,
    )
    print("result:", res)
