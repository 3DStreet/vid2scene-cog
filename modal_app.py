"""
Modal harness for the vid2scene video -> Gaussian-splat pipeline.

predict.py makes the cog image runnable on Replicate; this file makes the SAME
image runnable on Modal. Neither contains pipeline logic — both are thin,
host-agnostic adapters around the code baked into the image.

WHY THIS EXISTS
---------------
Replicate's on-demand tier preempts long (~tens of minutes) private jobs —
predictions may fail with `code: PA` / `Director ... E8765`. The exact same
image runs to success on dedicated hardware (peak VRAM only ~2.5 GB). Modal
gives non-preemptible, scale-to-zero, pay-per-job GPU execution with a 24h
limit (vs Cloud Run GPU's hard 60-min cap, which the largest presets would
blow past). See README.md ("Background: why run it off-Replicate").

THE KEY ARCHITECTURE: ONE IMAGE, TWO CONSUMERS
----------------------------------------------
The cog-built container image is the single source of truth — it bakes in the
upstream pipeline, the CUDA binaries (colmap/glomap/gsplat), the deps, and
predict.py. Replicate consumes it via `cog push`; Modal consumes the SAME
image via `Image.from_registry(...)`. Any fix in cog.yaml/predict.py —
rebuild + push once — propagates to BOTH platforms.

    cog.yaml + predict.py  --cog build-->  <registry>/<owner>/vid2scene
                                              /                \\
                                       cog push            from_registry
                                          |                      |
                                      Replicate               this file

DEPLOY
------
    pip install modal && modal token new
    # one-time: registry creds so Modal can pull the (private) image
    modal secret create r8im-pull REGISTRY_USERNAME=<user> REGISTRY_PASSWORD=<token>
    # I/O config — every key is optional; an unset key disables that feature:
    #   GCS_BUCKET, GCS_PREFIX   upload the finished .ply to your GCS bucket
    #   GCS_SA_JSON              service-account key JSON for the upload (Modal
    #                            containers have no ambient GCP credentials)
    #   WEBHOOK_URL              POST completion/failure to your backend
    #                            (a per-job "webhook_url" in the enqueue body wins)
    #   WEBHOOK_SECRET           sent back as "Authorization: Bearer <secret>"
    #   ENQUEUE_SECRET           require this token on enqueue/status requests
    modal secret create vid2scene-io GCS_BUCKET=<bucket> WEBHOOK_URL=<url> ...
    modal deploy modal_app.py

THE CONTRACT — how any host app plugs in (Firebase/GCP, or anything HTTP)
-------------------------------------------------------------------------
1. SUBMIT   POST https://<workspace>--vid2scene-enqueue.modal.run
              {"video_url": "<any fetchable URL — a signed Storage URL works>",
               "job_id": "<your id, optional>", "secret": "<ENQUEUE_SECRET>",
               "webhook_url": "<per-job callback URL, optional>",
               ...quality knobs (see `run_split` params)}
            -> {"call_id": "..."}  and the job runs asynchronously.
2. RESULT   receive the webhook on your backend (e.g. a Cloud Function):
              POST <webhook_url or WEBHOOK_URL>
              {"status": "succeeded"|"failed", "job_id": "...",
               "gcs_uri": "gs://...", "size_bytes": N}    (+ "error" on failure)
            ...or poll without a Modal SDK:
              GET https://<workspace>--vid2scene-status.modal.run
                  ?call_id=...&secret=<ENQUEUE_SECRET>
              -> {"status": "running"|"succeeded"|"failed"|"expired", ...}
3. FETCH    read the .ply from gcs_uri — it's your bucket.

A synchronous `modal run` path is included at the bottom for local testing.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import uuid

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
    # cog bakes the WHOLE repo at /src — including a build-time copy of THIS
    # file. With PYTHONPATH=/src below, that stale copy would shadow the
    # version Modal mounts at deploy time, so deployed functions silently run
    # old code. Remove it; predict.py is the only thing we import from /src.
    .run_commands("rm -rf /src/modal_app.py /src/__pycache__")
    # predict.py lives at the cog working dir (/src) in the image; make it import-
    # able so we can reuse Predictor.setup() + the pipeline entrypoint verbatim.
    # PYTHONUNBUFFERED is inherited by the gsplat trainer subprocess — without it
    # the trainer's piped output sits in a block buffer and the whole training
    # stage logs nothing until (unless) the process exits cleanly.
    .env({"PYTHONPATH": "/src", "PYTHONUNBUFFERED": "1"})
)

app = modal.App("vid2scene")

# Lightweight image for the control-plane functions (enqueue/status/run_split):
# they never touch the pipeline, and the multi-GB cog image gives web endpoints
# multi-minute cold starts on uncached workers. Slim cold-starts in ~1 s.
control_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "fastapi[standard]", "requests"
)

# Tunables. SfM (the ~27-min CPU-bound stage) dominates wall-clock, so give it
# real cores; gsplat needs the GPU but barely any VRAM (2.5 GB measured).
GPU = os.environ.get("VID2SCENE_GPU", "L4")        # L4/24GB is ample; T4 also fits
CPU = float(os.environ.get("VID2SCENE_CPU", "16"))  # physical cores -> faster SfM
MEMORY_MB = int(os.environ.get("VID2SCENE_MEM_MB", "32768"))
TIMEOUT_S = int(os.environ.get("VID2SCENE_TIMEOUT_S", "7200"))  # 2h: covers max tier

# Split-shape tunables (see "THE SPLIT SHAPE" below). The job is two opposite
# workloads, so each stage gets its own machine:
#   SfM:   CPU-hot, GPU only for hloc feature extraction/matching -> T4 + 16 cores
#   train: GPU-hot, ~2 cores busy                                 -> L4 + 4 cores
# Measured (default preset): split ≈ $0.94/job vs $1.36 single-function.
SFM_GPU = os.environ.get("VID2SCENE_SFM_GPU", "T4")
SFM_CPU = float(os.environ.get("VID2SCENE_SFM_CPU", "16"))
SFM_MEM_MB = int(os.environ.get("VID2SCENE_SFM_MEM_MB", "24576"))
TRAIN_GPU = os.environ.get("VID2SCENE_TRAIN_GPU", "L4")
TRAIN_CPU = float(os.environ.get("VID2SCENE_TRAIN_CPU", "4"))
TRAIN_MEM_MB = int(os.environ.get("VID2SCENE_TRAIN_MEM_MB", "16384"))

# Handoff volume for the split shape. Stages do their actual work on local disk
# (COLMAP's sqlite and gsplat's per-step image reads are too random for FUSE)
# and use the volume only to pass `sfm_output/` from one machine to the other;
# the train stage deletes the job's directory once the .ply is shipped.
work_volume = modal.Volume.from_name("vid2scene-work", create_if_missing=True)

# Jobs that die between stages leak /work/<job_id> (sfm_output is a few GB);
# completed jobs leave only a tiny result.json marker. Sweep both once they are
# old enough that no retry can still want them.
STALE_JOB_DAYS = 7


def _sweep_stale_jobs(keep_job: str) -> None:
    cutoff = time.time() - STALE_JOB_DAYS * 86400
    for name in os.listdir("/work"):
        path = os.path.join("/work", name)
        if name != keep_job and os.path.getmtime(path) < cutoff:
            shutil.rmtree(path, ignore_errors=True)


def _fetch_video(video_url: str, video_path: str) -> None:
    # A clean, picklable error: HTTPError carries a live socket reader, which
    # Modal can't serialize back to the coordinator — the host app would see an
    # exception-serialization artifact instead of the actual failure.
    try:
        urllib.request.urlretrieve(video_url, video_path)
    except Exception as e:
        raise RuntimeError(f"failed to download input video: {e}") from None


def _load_predictor():
    """Predictor from the baked predict.py — the same code Replicate runs."""
    if "/src" not in sys.path:
        sys.path.insert(0, "/src")
    from predict import Predictor

    predictor = Predictor()
    predictor.setup()
    return predictor


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
        self._predictor = _load_predictor()
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
        insv_fisheye: bool = False,
        use_background_sphere: bool = True,
        remove_background: bool = False,
        apriltag_size_meters: float = 0.0,
        webhook_url: str | None = None,
    ) -> dict:
        """Run one job on a single machine. Returns {job_id, gcs_uri|None,
        size_bytes}; uploads the .ply to GCS and POSTs a success webhook if
        configured. (Legacy shape — the split path is cheaper and also posts
        failure webhooks.)"""
        work_dir = tempfile.mkdtemp(prefix="vid2scene_")
        out_dir = os.path.join(work_dir, "out")
        os.makedirs(out_dir, exist_ok=True)

        # 1. fetch the input video (any URL the container can reach — signed
        #    GCS / Firebase Storage / S3 URLs are the usual choice)
        video_path = os.path.join(work_dir, "input")
        _fetch_video(video_url, video_path)

        # 2. run the pipeline (same entrypoint & cwd contract as predict.py)
        prev_cwd = os.getcwd()
        os.chdir("/src/vid2scene/vid2scene_core")
        try:
            ply_path = self._process(
                video_path=video_path,
                output_dir=out_dir,
                target_framecount=target_framecount,
                equirectangular=equirectangular,
                insv_fisheye=insv_fisheye,
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

        # 3. ship the result: upload to the configured GCS bucket (skipped if
        #    GCS_BUCKET is unset — the caller can still poll for this dict).
        gcs_uri = _upload_to_gcs(candidate, job_id) if os.environ.get("GCS_BUCKET") else None

        result = {"job_id": job_id, "gcs_uri": gcs_uri, "size_bytes": size}

        # 4. notify the host app (alternative to it polling the FunctionCall).
        _post_webhook({"status": "succeeded", **result}, url=webhook_url)
        return result


def _upload_to_gcs(local_path: str, job_id: str | None) -> str:
    from google.cloud import storage

    bucket_name = os.environ["GCS_BUCKET"]
    prefix = os.environ.get("GCS_PREFIX", "vid2scene")
    name = f"{prefix}/{job_id or os.path.basename(local_path)}.ply"
    # Modal containers carry no ambient GCP credentials — pass a service-account
    # key as the GCS_SA_JSON secret (the raw JSON). Default creds remain the
    # fallback for environments that do have them.
    sa_json = os.environ.get("GCS_SA_JSON")
    client = (
        storage.Client.from_service_account_info(json.loads(sa_json))
        if sa_json
        else storage.Client()
    )
    blob = client.bucket(bucket_name).blob(name)
    blob.upload_from_filename(local_path, content_type="application/octet-stream")
    return f"gs://{bucket_name}/{name}"


def _post_webhook(payload: dict, url: str | None = None) -> None:
    # Per-job URL (e.g. one carrying a signed token in its query string) wins;
    # the static WEBHOOK_URL secret is the fallback for hosts with one endpoint.
    url = url or os.environ.get("WEBHOOK_URL")
    if not url:
        return
    import requests

    headers = {"Content-Type": "application/json"}
    if os.environ.get("WEBHOOK_SECRET"):
        headers["Authorization"] = f"Bearer {os.environ['WEBHOOK_SECRET']}"
    try:
        requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
    except Exception as e:  # never fail the job because the callback flaked
        print(f"WARN: webhook POST failed: {e}")


# --- THE SPLIT SHAPE (requires the image with predict.py's stage API) ---------
# Two functions with opposite resource shapes, handing sfm_output/ off via the
# volume. Projected COGS for the default tier: ~$0.46 (SfM, T4+16cpu) + ~$0.48
# (train, L4+4cpu) ≈ $0.94 vs ~$1.36 measured on the single-function shape.


@app.cls(
    image=image,
    gpu=SFM_GPU,
    cpu=SFM_CPU,
    memory=SFM_MEM_MB,
    timeout=TIMEOUT_S,
    scaledown_window=120,
    volumes={"/work": work_volume},
)
class Vid2SceneSfM:
    @modal.enter()
    def setup(self):
        self._predictor = _load_predictor()

    @modal.method()
    def run_sfm(
        self,
        video_url: str,
        *,
        job_id: str,
        reconstruction_method: str = "glomap",
        target_framecount: int = 600,
        resolution: int = 1920,
        equirectangular: bool = False,
        insv_fisheye: bool = False,
        use_background_sphere: bool = True,
        remove_background: bool = False,
        apriltag_size_meters: float = 0.0,
    ) -> dict:
        """Stage A: video -> SfM artifacts, copied to /work/{job_id}/sfm_output."""
        handoff_dir = os.path.join("/work", job_id, "sfm_output")
        # Modal replays the same inputs when it retries a lost worker. If this
        # job's SfM already reached the volume, don't pay for the stage twice.
        work_volume.reload()
        if os.path.isdir(handoff_dir):
            return {"job_id": job_id, "normalize": not apriltag_size_meters}

        work_dir = tempfile.mkdtemp(prefix="vid2scene_")
        out_dir = os.path.join(work_dir, "out")
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(work_dir, "input")
        _fetch_video(video_url, video_path)

        sfm_dir = self._predictor.run_sfm(
            video_path=video_path,
            out_dir=out_dir,
            target_framecount=target_framecount,
            resolution=resolution,
            equirectangular=equirectangular,
            insv_fisheye=insv_fisheye,
            use_background_sphere=use_background_sphere,
            remove_background=remove_background,
            apriltag_size_meters=(apriltag_size_meters or None),
            reconstruction_method=reconstruction_method,
        )

        shutil.copytree(sfm_dir, handoff_dir, dirs_exist_ok=True)
        work_volume.commit()
        shutil.rmtree(work_dir, ignore_errors=True)
        # AprilTag scaling fixes real-world units; training must not renormalize.
        return {"job_id": job_id, "normalize": not apriltag_size_meters}


@app.cls(
    image=image,
    gpu=TRAIN_GPU,
    cpu=TRAIN_CPU,
    memory=TRAIN_MEM_MB,
    timeout=TIMEOUT_S,
    scaledown_window=120,
    volumes={"/work": work_volume},
    secrets=[modal.Secret.from_name("vid2scene-io")],
)
class Vid2SceneTrain:
    @modal.enter()
    def setup(self):
        self._predictor = _load_predictor()

    @modal.method()
    def run_train(
        self,
        *,
        job_id: str,
        training_num_steps: int = 30000,
        training_max_num_gaussians: int = 500000,
        normalize: bool = True,
        webhook_url: str | None = None,
    ) -> dict:
        """Stage B: /work/{job_id}/sfm_output -> .ply -> GCS + webhook."""
        work_volume.reload()
        job_dir = os.path.join("/work", job_id)
        handoff_dir = os.path.join(job_dir, "sfm_output")
        marker_path = os.path.join(job_dir, "result.json")
        # Retry of a job that already shipped (worker lost between our return and
        # the caller receiving it): hand back the recorded result, train nothing.
        # Re-post the webhook too — the loss may have raced the original POST,
        # and hosts must treat terminal callbacks as idempotent anyway.
        if os.path.exists(marker_path):
            with open(marker_path) as f:
                result = json.load(f)
            _post_webhook({"status": "succeeded", **result}, url=webhook_url)
            return result
        if not os.path.isdir(handoff_dir):
            raise RuntimeError(f"no SfM handoff at {handoff_dir} — did run_sfm succeed?")

        work_dir = tempfile.mkdtemp(prefix="vid2scene_")
        out_dir = os.path.join(work_dir, "out")
        os.makedirs(out_dir, exist_ok=True)
        local_sfm = os.path.join(out_dir, "sfm_output")
        shutil.copytree(handoff_dir, local_sfm)

        ply_path = self._predictor.run_train(
            sfm_dir=local_sfm,
            out_dir=out_dir,
            training_max_num_gaussians=training_max_num_gaussians,
            training_num_steps=training_num_steps,
            normalize=normalize,
        )

        candidate = ply_path or os.path.join(out_dir, "ply", "splat.ply")
        if not candidate or not os.path.exists(candidate):
            import glob

            found = glob.glob(os.path.join(out_dir, "**", "*.ply"), recursive=True)
            if not found:
                raise RuntimeError("vid2scene produced no .ply — check job logs.")
            candidate = found[0]

        size = os.path.getsize(candidate)
        gcs_uri = _upload_to_gcs(candidate, job_id) if os.environ.get("GCS_BUCKET") else None
        result = {"job_id": job_id, "gcs_uri": gcs_uri, "size_bytes": size}

        # Drop the bulky handoff but leave a marker behind: retries short-circuit
        # on it instead of failing with "no SfM handoff" after a successful run.
        with open(marker_path, "w") as f:
            json.dump(result, f)
        shutil.rmtree(handoff_dir, ignore_errors=True)
        _sweep_stale_jobs(keep_job=job_id)
        work_volume.commit()
        shutil.rmtree(work_dir, ignore_errors=True)

        _post_webhook({"status": "succeeded", **result}, url=webhook_url)
        return result


# Thin chaining function: ~zero-cost container (default CPU share) that waits on
# stage A then launches stage B. Lets enqueue stay fire-and-forget with ONE call
# id covering the whole job.
@app.function(
    image=control_image,
    timeout=2 * TIMEOUT_S,
    secrets=[modal.Secret.from_name("vid2scene-io")],
)
def run_split(
    video_url: str,
    job_id: str | None = None,
    reconstruction_method: str = "glomap",
    target_framecount: int = 600,
    resolution: int = 1920,
    training_num_steps: int = 30000,
    training_max_num_gaussians: int = 500000,
    equirectangular: bool = False,
    insv_fisheye: bool = False,
    use_background_sphere: bool = True,
    remove_background: bool = False,
    apriltag_size_meters: float = 0.0,
    webhook_url: str | None = None,
) -> dict:
    job_id = job_id or uuid.uuid4().hex
    try:
        sfm_result = Vid2SceneSfM().run_sfm.remote(
            video_url,
            job_id=job_id,
            reconstruction_method=reconstruction_method,
            target_framecount=target_framecount,
            resolution=resolution,
            equirectangular=equirectangular,
            insv_fisheye=insv_fisheye,
            use_background_sphere=use_background_sphere,
            remove_background=remove_background,
            apriltag_size_meters=apriltag_size_meters,
        )
        return Vid2SceneTrain().run_train.remote(
            job_id=job_id,
            training_num_steps=training_num_steps,
            training_max_num_gaussians=training_max_num_gaussians,
            normalize=sfm_result["normalize"],
            webhook_url=webhook_url,
        )
    except Exception as e:
        # Host apps need failure callbacks too (e.g. to refund a charge or
        # surface an error state) — the success webhook fires in run_train.
        _post_webhook(
            {"status": "failed", "job_id": job_id, "error": str(e)[:1000]},
            url=webhook_url,
        )
        raise


# --- Async enqueue endpoint: the HTTP front door for host apps ----------------
# A host app's queue/task system POSTs here; the job is spawned and a call id
# returned immediately. The host then either waits for the webhook (see
# _post_webhook) or polls modal.FunctionCall.from_id(call_id).get(timeout=0).
# The model stays a stateless compute backend behind the HOST's queue — this
# endpoint adds no queue of its own. See README.md "Integrating with a host app".
@app.function(image=control_image, secrets=[modal.Secret.from_name("vid2scene-io")])
@modal.fastapi_endpoint(method="POST")
def enqueue(body: dict) -> dict:
    """POST {video_url, job_id?, secret?, ...knobs} -> {call_id}. Fire-and-forget.
    Pass "split": false to use the legacy single-machine shape (success webhook
    only; the default split shape also posts failure webhooks)."""
    # Shared-token auth: set ENQUEUE_SECRET in the vid2scene-io secret and have
    # callers include it as "secret" in the body. For header-based auth instead,
    # Modal's built-in proxy auth (requires_proxy_auth=True above) works without
    # any code here.
    expected = os.environ.get("ENQUEUE_SECRET")
    if expected and body.pop("secret", None) != expected:
        return {"error": "unauthorized"}
    body.pop("secret", None)
    video_url = body.pop("video_url")
    # Pin the job_id at enqueue time: Modal replays a lost run_split with the
    # same inputs, and the idempotent stages key their prior work off this id.
    body.setdefault("job_id", uuid.uuid4().hex)
    if body.pop("split", True):
        call = run_split.spawn(video_url, **body)
    else:
        call = Vid2Scene().run.spawn(video_url, **body)
    return {"call_id": call.object_id}


# Poll companion to `enqueue`: hosts whose backends have no Modal SDK (e.g. a
# Node reconciler) GET this with the call_id they stored at submit time. This
# is the safety net for dropped webhooks, not the primary completion signal.
@app.function(image=control_image, secrets=[modal.Secret.from_name("vid2scene-io")])
@modal.fastapi_endpoint(method="GET")
def status(call_id: str, secret: str = "") -> dict:
    """GET ?call_id=...&secret=... ->
    {"status": "running"|"succeeded"|"failed"|"expired", "result"?, "error"?}"""
    expected = os.environ.get("ENQUEUE_SECRET")
    if expected and secret != expected:
        return {"error": "unauthorized"}
    try:
        call = modal.FunctionCall.from_id(call_id)
        result = call.get(timeout=0)
        return {"status": "succeeded", "result": result}
    except (TimeoutError, modal.exception.TimeoutError):
        return {"status": "running"}
    except modal.exception.OutputExpiredError:
        # Results are only retained ~7 days; an expired job is long since
        # terminal — the host's give-up timeout should have fired well before.
        return {"status": "expired"}
    except Exception as e:
        # .get() re-raises whatever the job raised -> the job failed.
        return {"status": "failed", "error": str(e)[:1000]}


# --- Local test / calibration -------------------------------------------------
#   single shape: modal run --detach modal_app.py --video-url <URL>
#   split shape:  modal run --detach modal_app.py --video-url <URL> --split
# (--detach matters for long jobs: ephemeral apps die with the client otherwise)
@app.local_entrypoint()
def main(
    video_url: str,
    gaussians: int = 500000,
    steps: int = 30000,
    frames: int = 600,
    resolution: int = 1920,
    split: bool = False,
    insv: bool = False,
    job_id: str = "",
):
    if split:
        res = run_split.remote(
            video_url,
            # Reusing a previous run's job_id resumes it: stages whose artifacts
            # already reached the volume are skipped.
            job_id=job_id or f"local-test-{uuid.uuid4().hex[:8]}",
            target_framecount=frames,
            resolution=resolution,
            insv_fisheye=insv,
            training_max_num_gaussians=gaussians,
            training_num_steps=steps,
        )
    else:
        res = Vid2Scene().run.remote(
            video_url,
            job_id="local-test",
            target_framecount=frames,
            insv_fisheye=insv,
            training_max_num_gaussians=gaussians,
            training_num_steps=steps,
        )
    print("result:", res)
