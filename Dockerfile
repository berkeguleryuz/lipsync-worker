# syntax=docker/dockerfile:1.7
# Sales Tarafı · MuseTalk 1.5 lip-sync worker'ı (RunPod Serverless)
#
# Aşamalar:
#   weights  · ağırlıkları huggingface.co'dan indirir, sha256 doğrular (CPU)
#   runtime  · CUDA 11.8 + torch 2.0.1 + mmcv 2.0.1 + MuseTalk sabit commit
#   test     · CPU'da handler şema/allowlist ve kalite kapısı testleri
#
# Build (VPS ya da lokal, GPU gerekmez):
#   docker buildx build --platform linux/amd64 --target runtime \
#     --build-arg MUSETALK_COMMIT=<sha> --build-arg FFMPEG_SHA256=<sha256> \
#     --build-arg IMAGE_VERSION=1.0.0 \
#     -t ghcr.io/<org>/sales-lipsync-musetalk:1.0.0 --push .
#   docker buildx build --target test -t sales-lipsync-test . && docker run --rm sales-lipsync-test
#
# Kırmızı çizgiler: imajda demo/test verisi, .git, syncnet ağırlığı, R2
# anahtarı YOK. Etiket asla `latest` değil; RunPod şablonunda digest ile sabitlenir.

ARG PYTHON_VERSION=3.10

# ── weights ──────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS weights
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "huggingface_hub[cli]==0.30.2"
WORKDIR /weights
COPY download_weights.sh weights.sha256* /tools/
# Sabit liste varsa doğrular; yoksa build durur (ilk build'de --pin ile üretilir).
ARG WEIGHTS_PIN=""
RUN bash /tools/download_weights.sh /weights ${WEIGHTS_PIN}

# ── runtime ──────────────────────────────────────────────────────────────
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS runtime
ARG PYTHON_VERSION
ARG MUSETALK_COMMIT
ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-01-13-13/ffmpeg-n8.1.2-50-g1a748fe2cd-linux64-gpl-8.1.tar.xz
ARG FFMPEG_SHA256
ARG IMAGE_VERSION=dev
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TORCH_HOME=/app/torch-home \
    MUSETALK_DIR=/app/MuseTalk \
    IMAGE_VERSION=${IMAGE_VERSION} \
    MUSETALK_COMMIT=${MUSETALK_COMMIT} \
    CACHE_DIR=/workspace/cache \
    JOB_TMP_DIR=/workspace/tmp

RUN test -n "${MUSETALK_COMMIT}" || (echo "MUSETALK_COMMIT build arg zorunlu (sabit commit)" && exit 1)
RUN test -n "${FFMPEG_SHA256}" || (echo "FFMPEG_SHA256 build arg zorunlu (statik ffmpeg doğrulaması)" && exit 1)

RUN apt-get update && apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python3-pip \
      git curl ca-certificates xz-utils libgl1 libglib2.0-0 libsndfile1 \
  && rm -rf /var/lib/apt/lists/* \
  && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
  && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3

# ffmpeg statik build (ayrı süreç olarak çağrılır, link edilmez), sha256 doğrulamalı.
RUN curl -fsSL -o /tmp/ffmpeg.tar.xz "${FFMPEG_URL}" \
  && echo "${FFMPEG_SHA256}  /tmp/ffmpeg.tar.xz" | sha256sum -c - \
  && mkdir -p /tmp/ffmpeg && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1 \
  && install -m 0755 /tmp/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg \
  && install -m 0755 /tmp/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe \
  && rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz \
  && ffmpeg -version | head -1

# torch 2.0.1 cu118 (MuseTalk sabitlemesi; Blackwell kademesi bu imajla kapalı).
RUN python -m pip install --upgrade "pip<24.1" \
  && python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
       --index-url https://download.pytorch.org/whl/cu118

# mm* zinciri: mmcv derlemesi uzun sürer (30-60 dk build'in büyük kısmı).
RUN python -m pip install -U openmim \
  && mim install "mmengine==0.10.4" \
  && mim install "mmcv==2.0.1" \
  && mim install "mmdet==3.1.0" \
  && mim install "mmpose==1.1.0"

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

# MuseTalk sabit commit; demo/test verisi, sonuçlar ve .git imaja girmez.
WORKDIR /app
RUN git clone --filter=blob:none https://github.com/TMElyralab/MuseTalk.git MuseTalk \
  && cd MuseTalk && git checkout --detach "${MUSETALK_COMMIT}" \
  && rm -rf .git data results assets docs *.md \
  && find . -name "*.mp4" -delete -o -name "*.wav" -delete -o -name "*.png" -delete \
  && python -c "import ast,sys; [ast.parse(open(p).read()) for p in sys.argv[1:]]" \
       musetalk/utils/utils.py musetalk/utils/preprocessing.py musetalk/utils/blending.py \
       musetalk/utils/audio_processor.py musetalk/utils/face_parsing/__init__.py

# Ağırlıklar (weights aşamasından) ve S3FD torch hub önbelleği.
COPY --from=weights /weights/musetalkV15 /app/MuseTalk/models/musetalkV15
COPY --from=weights /weights/dwpose /app/MuseTalk/models/dwpose
COPY --from=weights /weights/face-parse-bisent /app/MuseTalk/models/face-parse-bisent
COPY --from=weights /weights/sd-vae /app/MuseTalk/models/sd-vae
COPY --from=weights /weights/whisper /app/MuseTalk/models/whisper
COPY --from=weights /weights/torch-hub /app/torch-home/hub

# Worker kodu ve lisans bildirimleri.
COPY handler.py musetalk_runner.py quality_gate.py score.py /app/
COPY NOTICE /NOTICE
COPY NOTICE /app/NOTICE

# Non-root kullanıcı; /workspace yazılabilir (önbellek + geçici iş dizini).
RUN useradd --create-home --uid 1000 worker \
  && mkdir -p /workspace/cache /workspace/tmp \
  && chown -R worker:worker /workspace /app/torch-home
USER worker
WORKDIR /app/MuseTalk

# Varsayılan güvenlik çiti; RunPod endpoint env'i üzerine yazar.
ENV ALLOWED_INPUT_HOSTS=media.clodron.com \
    ALLOWED_UPLOAD_HOST_SUFFIX=.r2.cloudflarestorage.com \
    MAX_AUDIO_SECONDS=70 \
    MAX_SOURCE_SECONDS=90 \
    MAX_HEIGHT=1080 \
    MAX_DOWNLOAD_MB=300 \
    CACHE_MAX_AVATARS=10 \
    SCORE_ENABLED=0 \
    LOG_LEVEL=info \
    PYTHONPATH=/app:/app/MuseTalk

CMD ["python", "-u", "/app/handler.py"]

# ── test (CPU) ───────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS test
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
  && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir requests==2.32.3
WORKDIR /app
COPY handler.py musetalk_runner.py quality_gate.py score.py test_input.json /app/
COPY tests /app/tests
ENV LIPSYNC_TEST_MODE=1 \
    ALLOWED_INPUT_HOSTS=media.clodron.com \
    CACHE_DIR=/tmp/cache
RUN python -m py_compile handler.py musetalk_runner.py quality_gate.py score.py
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
