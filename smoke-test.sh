#!/usr/bin/env bash
# Post-deploy smoke check (rough version of acceptance tests K11 / K12):
# probe -> (optional prepare) -> render (cold) -> render (warm).
# Prints timings and prep_cache_hit. Presigned PUT URLs require object storage
# credentials, so this script reads them from ENV; the full measurement
# (TTS + presign + table) lives in the application that calls this worker.
#
# Required env: RUNPOD_API_KEY, RUNPOD_LIPSYNC_ENDPOINT_ID
# For render: AUDIO_URL, SOURCE_URL, OUTPUT_KEY, OUTPUT_PUT_URL (optional PREP_URL, SOURCE_HASH)
# For prepare: SOURCE_URL, PREP_KEY, PREP_PUT_URL
# Dependencies: curl, jq
set -euo pipefail

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY gerekli}"
: "${RUNPOD_LIPSYNC_ENDPOINT_ID:?RUNPOD_LIPSYNC_ENDPOINT_ID gerekli}"
TABAN="https://api.runpod.ai/v2/${RUNPOD_LIPSYNC_ENDPOINT_ID}"
POLL_SN="${POLL_SN:-5}"
MAKS_POLL="${MAKS_POLL:-312}"

# İmzalı URL'leri log'a yazmamak için jq ile maskele.
maskele() { sed -E 's/X-Amz-[^"&]*/X-Amz-***/g'; }

is_kostur() { # $1 = input json, $2 = policy json
  local govde yanit id durum i
  govde=$(jq -cn --argjson input "$1" --argjson policy "$2" '{input: $input, policy: $policy}')
  yanit=$(curl -sS -X POST "$TABAN/run" -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "$govde")
  id=$(echo "$yanit" | jq -r '.id // empty')
  if [ -z "$id" ]; then echo "run başarısız: $(echo "$yanit" | maskele)" >&2; return 1; fi
  for ((i = 0; i < MAKS_POLL; i++)); do
    sleep "$POLL_SN"
    yanit=$(curl -sS "$TABAN/status/$id" -H "Authorization: Bearer $RUNPOD_API_KEY")
    durum=$(echo "$yanit" | jq -r '.status // empty')
    case "$durum" in
      COMPLETED) echo "$yanit" | jq -c '.output'; return 0 ;;
      FAILED|TIMED_OUT|CANCELLED) echo "iş $durum: $(echo "$yanit" | maskele)" >&2; return 1 ;;
      *) ;;
    esac
  done
  curl -sS -X POST "$TABAN/cancel/$id" -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null || true
  echo "zaman aşımı, iş iptal edildi ($id)" >&2
  return 1
}

echo "== probe"
PROBE=$(is_kostur '{"mode":"probe"}' '{"executionTimeout":60000}')
echo "$PROBE" | jq '{image, musetalk_commit, gpu, vram_free_mb, cache_avatars}'

if [ -n "${PREP_PUT_URL:-}" ]; then
  echo "== prepare"
  GIRDI=$(jq -cn --arg s "$SOURCE_URL" --arg p "$PREP_PUT_URL" --arg k "$PREP_KEY" \
    '{mode:"prepare", source_url:$s, prep_put_url:$p, prep_key:$k, fps:25}')
  SONUC=$(is_kostur "$GIRDI" '{"executionTimeout":900000,"ttl":900000}')
  echo "$SONUC" | jq '{ok, code, error, source_hash, frames, face_frames_ratio, face_box_px, bytes, timings_ms}'
fi

if [ -n "${OUTPUT_PUT_URL:-}" ]; then
  render() {
    local girdi
    girdi=$(jq -cn --arg a "$AUDIO_URL" --arg s "$SOURCE_URL" --arg o "$OUTPUT_PUT_URL" --arg k "$OUTPUT_KEY" \
      --arg p "${PREP_URL:-}" --arg h "${SOURCE_HASH:-}" \
      '{mode:"render", audio_url:$a, source_url:$s, output_put_url:$o, output_key:$k, fps:25}
       + (if $p != "" then {prep_url:$p} else {} end)
       + (if $h != "" then {source_hash:$h} else {} end)')
    is_kostur "$girdi" '{"executionTimeout":600000,"ttl":900000}'
  }
  echo "== render (soğuk)"
  T0=$(date +%s)
  SOGUK=$(render)
  echo "duvar saati: $(( $(date +%s) - T0 )) sn"
  echo "$SOGUK" | jq '{ok, code, error, duration_seconds, frames, prep_cache_hit, prep_source, gpu_seconds, timings_ms, quality, worker}'

  echo "== render (sıcak, aynı sahne)"
  T0=$(date +%s)
  SICAK=$(render)
  echo "duvar saati: $(( $(date +%s) - T0 )) sn"
  echo "$SICAK" | jq '{ok, prep_cache_hit, prep_source, gpu_seconds, timings_ms}'
  echo "K12 beklenti: 10 sn sahnede soğuk ≤ 90 sn, sıcak ≤ 30 sn; ikinci çağrıda prep_cache_hit=true."
fi
