# -*- coding: utf-8 -*-
"""İsteğe bağlı ölçüm katmanı (tasarım 3.6): yalnız SCORE_ENABLED=1 imajında ve
`options.score=true` isteğinde çalışır. v1'de YÖNLENDİRME YOK: değerler
`quality` içine yazılır, eşik uygulanmaz. Dağılım ilk 100 sahnede toplanır,
v2 kararı (lse_c eşiği ile latentsync kademesi) veriyle alınır.

- sharpness_ratio · ağız bölgesi Laplacian varyansı, çıktı / kaynak. 1'e
  yakın = keskinlik korunmuş; belirgin düşüş 256 px üretim bölgesinin
  yumuşatması demek.
- lse_c / lse_d · SyncNet (joonson/syncnet_python, MIT). Modül imajda
  yoksa None döner; syncnet ağırlıkları YALNIZ bu skor modu imajına girer,
  ticari çıktıya değil (yalnız ölçüm).
"""
from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, Optional


def _orta_kare(video_yolu: str, oran: float = 0.5):
    """Videonun ortasındaki kareyi bgr numpy olarak döner."""
    import cv2  # noqa: WPS433

    cap = cv2.VideoCapture(video_yolu)
    try:
        toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(toplam * oran) - 1))
        ok, kare = cap.read()
        return kare if ok else None
    finally:
        cap.release()


def _alt_yuz_kesiti(kare):
    """Kaba ağız bölgesi: karenin orta-alt kesiti (dedektör gerektirmez)."""
    h, w = kare.shape[:2]
    return kare[int(h * 0.55) : int(h * 0.85), int(w * 0.3) : int(w * 0.7)]


def keskinlik_orani(cikti_yolu: str, kaynak_yolu: str) -> Optional[float]:
    import cv2  # noqa: WPS433

    c = _orta_kare(cikti_yolu)
    k = _orta_kare(kaynak_yolu)
    if c is None or k is None:
        return None
    lap = lambda kare: float(cv2.Laplacian(cv2.cvtColor(_alt_yuz_kesiti(kare), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())  # noqa: E731
    payda = lap(k)
    if payda <= 0:
        return None
    return round(lap(c) / payda, 4)


def syncnet_skoru(cikti_yolu: str) -> Dict[str, Optional[float]]:
    """SyncNet LSE-C / LSE-D. `syncnet_python` yoksa None."""
    try:
        import syncnet_python  # type: ignore # noqa: F401, WPS433
    except Exception:
        return {"lse_c": None, "lse_d": None}
    try:
        sonuc = subprocess.run(
            ["python", "-m", "syncnet_python.score", "--video", cikti_yolu, "--json"],
            capture_output=True, text=True, timeout=120,
        )
        veri = json.loads(sonuc.stdout or "{}")
        return {"lse_c": veri.get("lse_c"), "lse_d": veri.get("lse_d")}
    except Exception:
        return {"lse_c": None, "lse_d": None}


def hesapla(cikti_yolu: str, kaynak_yolu: str, ses_wav: str) -> Dict[str, Any]:
    sonuc: Dict[str, Any] = {"sharpness_ratio": keskinlik_orani(cikti_yolu, kaynak_yolu)}
    sonuc.update(syncnet_skoru(cikti_yolu))
    return sonuc
