# Setup: build the image with GitHub Actions and deploy it to RunPod

This guide takes the worker from source to a running RunPod Serverless
endpoint without building anything on your own machine.

## Why this path

| Topic | Decision | Reason |
|---|---|---|
| Build location | GitHub Actions (ubuntu runner, 4 vCPU / 16 GB RAM) | Off your machine, free for public repositories, reproducible |
| Registry | GHCR (ghcr.io), public package | Public packages have free storage and transfer; the workflow's own `GITHUB_TOKEN` can push, no extra token needed |
| Repository visibility | Public | The image holds no secrets (MIT code, open weights, handler); all keys live in the RunPod endpoint env. A private package would exceed the free storage quota and RunPod would need registry credentials |
| RunPod's own GitHub build | Not used | Its Docker build step is capped at 30 minutes; this image needs 30 to 60 |

## Costs

| Item | Amount |
|---|---|
| GitHub Actions (public repo) | 0 |
| GHCR public package storage and transfer | 0 |
| RunPod Serverless, idle | 0 |
| RunPod Serverless, while rendering | about 0.04 to 0.05 USD per output minute (24 GB standard GPU, cold starts included) |

## Prerequisites

1. A RunPod account with 2FA enabled and a spending limit set (Billing →
   Spending limit). Create an API key and, if possible, restrict it to
   Serverless.
2. A GitHub account; `gh` logged in (`gh auth status`).
3. Optional: the RunPod Claude Code plugin (`claude plugin marketplace add
   runpod/runpod-plugins-official`, `claude plugin install runpod@runpod`) and a
   sign-in via `/mcp` → runpod, so the endpoint can be created from the agent.

## Step 1 · Create the repository

```bash
cd sales-lipsync-worker          # this directory, copied out of the main project
git init -b main
git add -A
git commit -m "MuseTalk 1.5 lip-sync worker for RunPod Serverless"
gh repo create <owner>/sales-lipsync-worker --public --source=. --push \
  --description "MuseTalk 1.5 lip-sync worker (RunPod Serverless)"
```

The repository contains the worker files, `.github/workflows/build.yml`,
`.gitignore` and `PINS.md`.

## Step 2 · Version pins

- `MUSETALK_COMMIT`: set in the workflow env (see `PINS.md`).
- `FFMPEG_SHA256`: the workflow downloads the ffmpeg static build, hashes it
  and passes the hash as a build argument; the value is printed in the job
  summary.
- `weights.sha256`: the first run is a "pin" build that generates the list and
  uploads it as an artifact. Commit that file to the repository; every later
  build verifies against it and stops on a mismatch.

## Step 3 · Run the workflow

Actions → `build-worker-image` → Run workflow → `image_version=1.0.0`,
`pin_weights=true`. Or from a terminal:

```bash
gh workflow run build.yml -f image_version=1.0.0 -f pin_weights=true
gh run watch
```

What the job does: frees runner disk space (the default 14 GB is not enough),
sets up buildx, logs in to GHCR with `GITHUB_TOKEN`, computes the ffmpeg hash,
runs the CPU unit tests, generates `weights.sha256` (pin mode), builds and
pushes `ghcr.io/<owner>/sales-lipsync-musetalk:<version>` and writes the image
digest to the job summary. Expect 45 to 60 minutes.

After the first run:

```bash
gh run download --name weights-sha256      # puts weights.sha256 in the working dir
git add weights.sha256 && git commit -m "Pin model weights" && git push
```

Subsequent releases: bump the version and push a tag (`git tag v1.0.1 && git
push --tags`); the workflow runs in verification mode automatically.

## Step 4 · Package visibility

GitHub → your profile → Packages → `sales-lipsync-musetalk` → Package
settings → Change visibility → Public. A package linked to a public repository
is usually public already; verify it, otherwise RunPod cannot pull the image.

## Step 5 · Create the RunPod endpoint

Use the values from `endpoint.example.json` (console: Serverless → New
Endpoint → Custom image, or the RunPod API / agent plugin):

| Setting | Value |
|---|---|
| Name | sales-lipsync-musetalk |
| Image | `ghcr.io/<owner>/sales-lipsync-musetalk@sha256:<digest from the job summary>` |
| Workers | Flex, min 0, max 2 |
| GPU priority | 1) L4 / RTX A5000 / RTX 3090 · 2) RTX 4090 · 3) RTX A4000 / A4500; Blackwell off |
| Idle timeout | 30 s |
| Execution timeout | 900 s |
| FlashBoot | on |
| Scaler | Queue delay, 4 s |
| Container disk | 30 GB |
| Env | `ALLOWED_INPUT_HOSTS=<your media host>`, `ALLOWED_UPLOAD_HOST_SUFFIX=.r2.cloudflarestorage.com`, `MAX_AUDIO_SECONDS=70`, `MAX_SOURCE_SECONDS=90`, `MAX_HEIGHT=1080`, `MAX_DOWNLOAD_MB=300`, `CACHE_MAX_AVATARS=10`, `SCORE_ENABLED=0`, `LOG_LEVEL=info` |
| Never | any storage credential in the endpoint env (the worker writes through presigned URLs) |

RunPod assigns an endpoint ID (visible on the endpoint page and in its URL).
The calling application needs two settings: `RUNPOD_API_KEY` and
`RUNPOD_LIPSYNC_ENDPOINT_ID`.

## Step 6 · Verify

1. Probe: `smoke-test.sh` sends `{ "input": { "mode": "probe" } }` and expects
   `COMPLETED` (first call is a cold start, 60 to 90 s).
2. Prepare one avatar, then render one scene cold and one warm; check
   `prep_cache_hit=true` on the second call and compare timings with the
   acceptance thresholds in `README.md`.
3. Watch a couple of lip-synced clips for visual quality before enabling the
   endpoint in production.

## Releasing a new version

1. Change the worker source in the main project, run the CPU tests.
2. Copy the directory into this repository, bump `IMAGE_VERSION`, push a
   `v1.x.y` tag; Actions builds and pushes the image.
3. Point the RunPod template at the new digest. Old workers drain on idle and
   the new image takes over.
