# MuseTalk 1.5 lip-sync worker for RunPod Serverless

A single Docker image that turns an audio file plus a short talking-head
source video into a lip-synced video, running as a RunPod Serverless endpoint.
One endpoint, three modes: `render`, `prepare`, `probe`. The worker writes its
output directly to S3-compatible object storage through a presigned PUT URL, so
it never holds storage credentials.

## What is in this directory

| File | Purpose |
|---|---|
| `handler.py` | RunPod entry point: input validation and safety fence, downloads, sha256, avatar cache, quality gate, presigned upload, structured errors |
| `musetalk_runner.py` | Model loading (at import time), `prepare` (DWPose + S3FD + face parsing + VAE latents), `synthesize` (whisper → UNet → blend → ffmpeg stdin) |
| `quality_gate.py` | Deterministic output gate via ffprobe: container, duration, frame count, resolution, face ratio |
| `score.py` | Optional metrics (`SCORE_ENABLED=1`): Laplacian sharpness ratio, LSE when SyncNet is present |
| `Dockerfile` | Stages `weights` → `runtime` → `test` |
| `requirements.txt` | Trimmed, version-pinned dependencies |
| `download_weights.sh` | Weight download plus sha256 verification (`--pin` generates the list) |
| `weights.sha256` | Produced by the first pinned build and committed (see Build) |
| `NOTICE` | Third-party license notices (copied into the image root) |
| `test_input.json` | Sample request for `python handler.py --test_input` (no GPU needed) |
| `endpoint.example.json` | RunPod endpoint settings |
| `smoke-test.sh` | Post-deploy probe / prepare / render check (curl + jq) |
| `tests/` | CPU unit tests: input fence and quality gate |
| `SETUP.md` | Step by step: GitHub Actions build, GHCR, RunPod endpoint |

## API contract

Requests go to `POST https://api.runpod.ai/v2/{endpointId}/run`, results are
polled with `GET /status/{id}` (every 5 s), and the caller can abort with
`POST /cancel/{id}` when its own deadline passes.

### `render`

Input: `audio_url`, `source_url`, `output_put_url` (presigned PUT),
`output_key`, optional `source_hash`, `prep_url`, `fps` (25), `options`.

Output: `{ ok, key, duration_seconds, frames, fps, width, height, bytes, model,
prep_cache_hit, prep_source, timings_ms, gpu_seconds, quality, worker }`.

Handled errors still complete the job (`COMPLETED`) with
`{ ok:false, code, error, retryable }`. Codes: `INPUT_REJECTED`,
`DOWNLOAD_FAILED`, `FACE_NOT_FOUND`, `PREP_FAILED`, `RENDER_FAILED`, `OOM`,
`QUALITY_GATE`, `UPLOAD_FAILED`. `retryable` is true only for
`DOWNLOAD_FAILED`, `UPLOAD_FAILED` and `OOM`.

### `prepare`

Input: `source_url`, `prep_put_url`, `prep_key`
(`avatars/{slug}/prep-{hash12}.tar`), optional `expected_source_hash`, `norms`.

Output: `source_hash`, `frames`, `face_frames_ratio`, `face_box_px`, `bytes`,
`timings_ms`. The package contains `latents.pt`, `coords.pkl`,
`mask_coords.pkl`, `masks.npz`, `meta.json`. Source frames are not stored.

### `probe`

Output: `{ ok, image, musetalk_commit, gpu, vram_free_mb, cache_avatars }`.

## Source video requirements

25 fps, up to 1080p, 15 to 90 seconds, a single frontal face, face box roughly
250 to 450 px. Re-encode if needed:

```bash
ffmpeg -i in.mp4 -vf fps=25 -c:v libx264 -crf 18 -pix_fmt yuv420p -an out.mp4
```

## Build and push (no GPU needed)

The recommended path is GitHub Actions, described in `SETUP.md`. Manual
equivalent:

1. Pin the MuseTalk commit and the ffmpeg static build hash:

   ```bash
   git ls-remote https://github.com/TMElyralab/MuseTalk.git HEAD
   curl -fsSL -o /tmp/ffmpeg.tar.xz https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-01-13-13/ffmpeg-n8.1.2-50-g1a748fe2cd-linux64-gpl-8.1.tar.xz
   sha256sum /tmp/ffmpeg.tar.xz
   ```

2. First build only: generate the weight list and commit it:

   ```bash
   docker buildx build --platform linux/amd64 --target weights \
     --build-arg WEIGHTS_PIN=--pin -t lipsync-weights --load .
   docker run --rm lipsync-weights cat /tools/weights.sha256 > weights.sha256
   ```

   Later builds run in verification mode (no `WEIGHTS_PIN`) and STOP if any
   weight does not match the committed list.

3. Runtime image:

   ```bash
   docker buildx build --platform linux/amd64 --target runtime \
     --build-arg MUSETALK_COMMIT=<sha> \
     --build-arg FFMPEG_SHA256=<sha256> \
     --build-arg IMAGE_VERSION=1.0.0 \
     -t ghcr.io/<owner>/sales-lipsync-musetalk:1.0.0 --push .
   docker buildx imagetools inspect ghcr.io/<owner>/sales-lipsync-musetalk:1.0.0
   ```

   Takes 30 to 60 minutes (mmcv compilation plus about 4.5 GB of weights).
   Image size 12 to 14 GB. Never tag `latest`; the RunPod template pins the
   digest.

4. CPU tests (independent of the image build, seconds):

   ```bash
   docker buildx build --target test -t lipsync-test --load . && docker run --rm lipsync-test
   # or locally:
   LIPSYNC_TEST_MODE=1 python3 -m unittest discover -s tests -v
   python3 -m py_compile *.py
   ```

## RunPod endpoint settings

Apply `endpoint.example.json` in the console or through the API.

| Setting | Value |
|---|---|
| Worker type | Flex, min 0, max 2 |
| GPU priority | 1) L4 / RTX A5000 / RTX 3090 · 2) RTX 4090 · 3) RTX A4000 / A4500 |
| Blackwell (5090, RTX PRO, B200) | OFF (CUDA 11.8 image does not run there) |
| Allowed CUDA versions | 11.8 to 12.6 (this is what actually excludes Blackwell hosts and their MIG slices; the GPU list alone is not enough) |
| Idle timeout | 30 s |
| Execution timeout | 900 s (requests lower it via `policy`: render 600 s, prepare 900 s) |
| FlashBoot | on |
| Scaler | Queue delay, 4 s |
| Container disk | 30 GB |
| Network volume | none |
| Data center | unrestricted |

Endpoint env: `ALLOWED_INPUT_HOSTS=<your media host>`,
`ALLOWED_UPLOAD_HOST_SUFFIX=.r2.cloudflarestorage.com`, `MAX_AUDIO_SECONDS=70`,
`MAX_SOURCE_SECONDS=90`, `MAX_HEIGHT=1080`, `MAX_DOWNLOAD_MB=300`,
`CACHE_MAX_AVATARS=10`, `SCORE_ENABLED=0`, `LOG_LEVEL=info`.

Checklist: no storage credentials in the endpoint env; the RunPod API key is
restricted to this endpoint; a spending limit and 2FA are enabled on the
account.

## Cost

On the 24 GB standard tier expect roughly 0.03 to 0.05 USD per minute of
output (generation at about 2.5x real time, plus a 30 s idle window and a
share of cold starts). Idle cost is zero. Very short clips (under 5 s) inflate
the per-minute cost 5 to 10 times, so callers should keep scenes at 8 s or
longer.

## Security

- The worker never receives long-lived storage credentials. Write access is a
  single-key, PUT-only presigned URL with a 30 minute lifetime, generated by
  the caller.
- Input fence: https only, host allowlist, no redirects, size and duration
  caps, `output_key` must match the presigned URL path, `..` is rejected.
  Presigned URLs are masked as `X-Amz-***` in logs and error messages.
- Temporary files are removed in `finally`; only the derived avatar cache
  (latents, coordinates, masks) survives between jobs.
- Image: pinned commit, digest-pinned tag, zero downloads at runtime
  (`HF_HUB_OFFLINE=1`), sha256-verified ffmpeg, non-root user, no demo or
  test data, no SyncNet weight. `NOTICE` is copied to the image root.

## Versioning

`IMAGE_VERSION` follows semver (1.0.0). Model, weight or contract changes bump
minor; handler-only fixes bump patch. The RunPod template is moved to the new
digest on every release; old images stay in the registry for rollback.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `mmcv` compilation takes hours or fails | `mim install mmcv==2.0.1` pulls a prebuilt wheel for cu118 + torch 2.0.1; without a wheel it compiles from source. Retry with `MMCV_WITH_OPS=1 FORCE_CUDA=1`, or reuse the last working image |
| Worker `FAILED`, log says `CUDA error: no kernel image` | A Blackwell GPU (5090 / RTX PRO / B200) was selected. Remove it from the endpoint GPU list |
| `OOM` (retryable) | Batch 8 can be too much on the 16 GB tier; use `options.batch_size=4`. If it repeats, drop the 16 GB tier |
| `FACE_NOT_FOUND` | A frame without a face (profile turn, out of frame). Trim or regenerate the clip, run `prepare` again |
| `INPUT_REJECTED: frame rate` | Source is not 25 fps; re-encode with the ffmpeg command above |
| `QUALITY_GATE: duration drift` | Audio file broken or truncated; check the audio encoding chain. Nothing is uploaded until the gate passes |
| First scene very slow, `prep_source: computed` | `prepare` was never run for that avatar |
| `UPLOAD_FAILED` HTTP 403 | Presigned URL expired (30 min) or `output_key` does not match the URL path; resubmitting the job creates a fresh signature |
| `S3FD` tries to download at runtime | `torch-hub/checkpoints/s3fd-619a316812.pth` is missing from the image; check `download_weights.sh` output and `TORCH_HOME` |

## Acceptance tests that need a live GPU

These are not unit tests; run them with `smoke-test.sh` after deployment.

- Probe returns the image version and GPU name; `prepare` writes a tar to
  storage and returns the hash; a clip that violates the norms is rejected
  with `INPUT_REJECTED`.
- A 10 s scene: cold start at most 90 s, warm at most 30 s; a 60 s scene warm
  at most 180 s; the second call reports `prep_cache_hit=true`; output is
  25 fps at source resolution and within ±250 ms of the audio length.
- Measured cost at most 0.08 USD per output minute on the 24 GB standard tier.
- Presigned URLs return 403 after 30 minutes; the endpoint env holds no storage
  keys; `X-Amz-Signature` never appears in worker logs.
