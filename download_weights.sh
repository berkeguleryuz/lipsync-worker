#!/usr/bin/env bash
# MuseTalk 1.5 ağırlıklarını build sırasında indirir ve sha256 ile doğrular.
# Çalışma zamanında hiçbir indirme yok (HF_HUB_OFFLINE=1).
#
# Kullanım:
#   ./download_weights.sh <hedef_dizin>          # indir + weights.sha256 ile doğrula
#   ./download_weights.sh <hedef_dizin> --pin    # indir + weights.sha256 dosyasını (yeniden) yaz
#
# İlk build'de --pin ile liste üretilir ve depoya eklenir; sonraki build'ler
# listeyi doğrular. Liste yoksa ve --pin verilmediyse build DURUR: doğrulanmamış
# ağırlıkla imaj üretmek tedarik zinciri kuralına aykırı (tasarım bölüm 5).
#
# İmaja GİRMEYENLER: syncnet/latentsync_syncnet.pt (yalnız eğitim lisansı),
# musetalk/ v1.0 ağırlıkları, InsightFace paketleri, Wav2Lip ağırlıkları.
set -euo pipefail

HEDEF="${1:?hedef dizin gerekli}"
PIN="${2:-}"
BURASI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LISTE="$BURASI/weights.sha256"
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$HEDEF"
cd "$HEDEF"

echo "==> MuseTalk 1.5 (unet + config)"
huggingface-cli download TMElyralab/MuseTalk \
  --include "musetalkV15/*" \
  --local-dir . --local-dir-use-symlinks False

# DWPose ve face-parse ağırlıkları TMElyralab/MuseTalk deposunda YOK; DWPose
# resmi yayıncıdan (yzd-v), face-parse ise sha256 listesiyle sabitlenmiş HF
# aynasından (orijinal Google Drive'da; iki bağımsız ayna aynı hash'i veriyor).
echo "==> DWPose (yzd-v/DWPose, Apache 2.0)"
mkdir -p dwpose
huggingface-cli download yzd-v/DWPose dw-ll_ucoco_384.pth \
  --local-dir dwpose --local-dir-use-symlinks False

echo "==> face-parse-bisent 79999_iter.pth (zllrunning, MIT; HF aynası camenduru/MuseTalk)"
mkdir -p face-parse-bisent
huggingface-cli download camenduru/MuseTalk face-parse-bisent/79999_iter.pth \
  --local-dir . --local-dir-use-symlinks False

echo "==> sd-vae-ft-mse"
mkdir -p sd-vae
huggingface-cli download stabilityai/sd-vae-ft-mse \
  --include "config.json" "diffusion_pytorch_model.bin" \
  --local-dir sd-vae --local-dir-use-symlinks False

echo "==> whisper-tiny (çok dilli; tiny.en DEĞİL)"
mkdir -p whisper
huggingface-cli download openai/whisper-tiny \
  --include "config.json" "model.safetensors" "preprocessor_config.json" \
  --local-dir whisper --local-dir-use-symlinks False

# face-parse-bisent'in resnet18 omurgası HF paketinde yoksa PyTorch'tan.
if [ ! -f face-parse-bisent/resnet18-5c106cde.pth ]; then
  echo "==> resnet18 omurgası"
  mkdir -p face-parse-bisent
  curl -fsSL -o face-parse-bisent/resnet18-5c106cde.pth \
    https://download.pytorch.org/models/resnet18-5c106cde.pth
fi

# S3FD yüz dedektörü (face_alignment kopyası, BSD-3): preprocessing modülü
# import anında torch hub önbelleğinden yükler; build'de oraya konur ki
# çalışma zamanında indirme olmasın.
echo "==> S3FD dedektörü (torch hub önbelleği)"
mkdir -p torch-hub/checkpoints
if [ ! -f torch-hub/checkpoints/s3fd-619a316812.pth ]; then
  curl -fsSL -o torch-hub/checkpoints/s3fd-619a316812.pth \
    https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
fi

# Yasak / gereksiz dosyalar hiçbir koşulda kalmasın.
rm -rf syncnet musetalk .cache
find . -name "*.md" -delete || true

echo "==> sha256"
if [ "$PIN" = "--pin" ]; then
  find . -type f \( -name "*.pth" -o -name "*.bin" -o -name "*.safetensors" -o -name "*.json" \) \
    | sort | xargs sha256sum > "$LISTE"
  echo "weights.sha256 yazıldı: $LISTE ($(wc -l < "$LISTE") dosya). Depoya ekleyin."
else
  if [ ! -f "$LISTE" ]; then
    echo "HATA: $LISTE yok. İlk build'de '--pin' ile üretip depoya ekleyin." >&2
    exit 2
  fi
  sha256sum -c --strict "$LISTE"
  echo "ağırlıklar doğrulandı."
fi

du -sh . | sed 's/^/toplam: /'
