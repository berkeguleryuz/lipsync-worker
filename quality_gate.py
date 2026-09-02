# -*- coding: utf-8 -*-
"""Deterministik kalite kapısı (tasarım bölüm 3.6).

Her render'da ~0,5 sn: ffprobe ile çıktı dosyasının biçim ve süre tutarlılığı
kontrol edilir. Kapıdan geçemeyen çıktı R2'ye YÜKLENMEZ.

| Kontrol                                            | Sonuç          |
| Dosya var, boyut > 50 KB                           | RENDER_FAILED  |
| video + ses akışı, h264 / aac, yuv420p             | RENDER_FAILED  |
| video süresi - ses süresi                          | ≤ 250 ms, yoksa QUALITY_GATE |
| kare sayısı - round(ses_sn × fps)                  | ≤ 2, yoksa QUALITY_GATE |
| çıktı genişlik/yükseklik == kaynak                 | QUALITY_GATE   |
| face_frames_ratio (hazırlık metadata'sı)            | == 1.0, yoksa FACE_NOT_FOUND |

`probe_fn` enjekte edilebilir: CPU testleri ffprobe çalıştırmadan kapıyı
sınar. SyncNet skoru burada değil (score.py, yalnız ölçüm).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

MIN_DOSYA_BAYT = 50 * 1024
SURE_TOLERANSI_MS = 250
KARE_TOLERANSI = 2


def ffprobe(yol: Path) -> Dict[str, Any]:
    """`ffprobe -show_streams -show_format` çıktısını JSON olarak döner."""
    cmd = [
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-count_packets", "-of", "json", str(yol),
    ]
    sonuc = subprocess.run(cmd, capture_output=True, text=True)
    if sonuc.returncode != 0:
        raise RuntimeError(f"ffprobe başarısız: {sonuc.stderr.strip()[:300]}")
    return json.loads(sonuc.stdout or "{}")


def _kesir(metin: Any) -> Optional[float]:
    """'25/1' ya da '30000/1001' gibi kesirleri float'a çevirir."""
    if metin is None:
        return None
    try:
        if isinstance(metin, (int, float)):
            return float(metin)
        pay, _, payda = str(metin).partition("/")
        p = float(pay)
        q = float(payda) if payda else 1.0
        return p / q if q else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _akis(probe: Dict[str, Any], tur: str) -> Optional[Dict[str, Any]]:
    for akis in probe.get("streams", []) or []:
        if akis.get("codec_type") == tur:
            return akis
    return None


def _sure(probe: Dict[str, Any], akis: Optional[Dict[str, Any]]) -> Optional[float]:
    for kaynak in (akis or {}, probe.get("format", {}) or {}):
        deger = kaynak.get("duration")
        try:
            if deger is not None:
                return float(deger)
        except (TypeError, ValueError):
            continue
    return None


def video_bilgisi(probe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Kaynak/çıktı video akışının boyut, süre, fps ve kare sayısı."""
    akis = _akis(probe, "video")
    if akis is None:
        return None
    sure = _sure(probe, akis)
    fps = _kesir(akis.get("avg_frame_rate")) or _kesir(akis.get("r_frame_rate"))
    kare = None
    for alan in ("nb_frames", "nb_read_packets"):
        try:
            if akis.get(alan) not in (None, "N/A"):
                kare = int(akis[alan])
                break
        except (TypeError, ValueError):
            continue
    if kare is None and sure is not None and fps:
        kare = int(round(sure * fps))
    return {
        "width": int(akis.get("width") or 0),
        "height": int(akis.get("height") or 0),
        "duration": float(sure or 0.0),
        "fps": fps,
        "frames": kare,
        "codec": akis.get("codec_name"),
        "pix_fmt": akis.get("pix_fmt"),
    }


def ses_bilgisi(probe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    akis = _akis(probe, "audio")
    if akis is None:
        return None
    sure = _sure(probe, akis)
    return {
        "duration": float(sure or 0.0),
        "codec": akis.get("codec_name"),
        "sample_rate": int(akis.get("sample_rate") or 0),
    }


@dataclass
class KapiSonucu:
    ok: bool
    code: Optional[str] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


def kontrol(
    cikti: Path,
    audio_seconds: float,
    source_width: int,
    source_height: int,
    fps: int = 25,
    face_frames_ratio: Optional[float] = None,
    probe_fn: Callable[[Path], Dict[str, Any]] = ffprobe,
    dosya_boyutu: Optional[int] = None,
) -> KapiSonucu:
    """Kapı kontrolü. Sıra önemli: biçim hataları RENDER_FAILED, tutarlılık
    hataları QUALITY_GATE; yüz oranı FACE_NOT_FOUND."""
    cikti = Path(cikti)
    if dosya_boyutu is None:
        if not cikti.is_file():
            return KapiSonucu(False, "RENDER_FAILED", "çıktı dosyası yok")
        dosya_boyutu = cikti.stat().st_size
    if dosya_boyutu < MIN_DOSYA_BAYT:
        return KapiSonucu(False, "RENDER_FAILED", f"çıktı çok küçük ({dosya_boyutu} bayt)")

    try:
        probe = probe_fn(cikti)
    except Exception as hata:
        return KapiSonucu(False, "RENDER_FAILED", f"çıktı okunamadı: {hata}")

    video = video_bilgisi(probe)
    ses = ses_bilgisi(probe)
    if video is None or ses is None:
        return KapiSonucu(False, "RENDER_FAILED", "çıktıda video ve ses akışı birlikte olmalı")
    if video["codec"] != "h264" or ses["codec"] != "aac" or video["pix_fmt"] != "yuv420p":
        return KapiSonucu(
            False,
            "RENDER_FAILED",
            f"biçim beklenen değil: video={video['codec']}/{video['pix_fmt']} ses={ses['codec']}",
        )

    beklenen_kare = int(round(audio_seconds * fps))
    sure_farki_ms = int(round((video["duration"] - audio_seconds) * 1000))
    kare_farki = (video["frames"] - beklenen_kare) if video["frames"] is not None else None
    metrikler = {
        "output_duration": round(video["duration"], 3),
        "output_frames": video["frames"],
        "expected_frames": beklenen_kare,
        "duration_delta_ms": sure_farki_ms,
        "frame_delta": kare_farki,
        "face_frames_ratio": face_frames_ratio,
    }

    if face_frames_ratio is not None and face_frames_ratio < 1.0:
        return KapiSonucu(False, "FACE_NOT_FOUND", f"yüz oranı {face_frames_ratio:.3f} < 1.0", metrikler)
    if abs(sure_farki_ms) > SURE_TOLERANSI_MS:
        return KapiSonucu(
            False,
            "QUALITY_GATE",
            f"çıktı süresi ses süresinden {sure_farki_ms:+d} ms sapıyor (tolerans ±{SURE_TOLERANSI_MS} ms)",
            metrikler,
        )
    if kare_farki is not None and abs(kare_farki) > KARE_TOLERANSI:
        return KapiSonucu(
            False,
            "QUALITY_GATE",
            f"kare sayısı {video['frames']}, beklenen {beklenen_kare} (tolerans ±{KARE_TOLERANSI})",
            metrikler,
        )
    if video["width"] != source_width or video["height"] != source_height:
        return KapiSonucu(
            False,
            "QUALITY_GATE",
            f"çıktı {video['width']}x{video['height']}, kaynak {source_width}x{source_height}",
            metrikler,
        )
    return KapiSonucu(True, None, None, metrikler)
