# -*- coding: utf-8 -*-
"""RunPod Serverless handler: MuseTalk 1.5 lip-sync worker'ı.

Tek endpoint, üç mod (tasarım bölüm 3.3):
  render  · sahne üretimi: ses + kaynak klip → lip-sync mp4, presigned PUT ile R2'ye
  prepare · avatar hazırlık paketi (tar) üretimi, seed sonrası bir kez
  probe   · sağlık ve sürüm

Güvenlik çiti (tasarım bölüm 5): yalnız https, girdi host allowlist'i, yükleme
host son eki, output_key ile yol eşleşmesi, boyut/süre tavanları, yönlendirme
takip edilmez. Presigned URL'ler loglara ve hata metinlerine asla yazılmaz.

Model yükleme import anında (`handler()` dışında) yapılır ki FlashBoot anlık
görüntüsü modeli VRAM'de yakalasın. LIPSYNC_TEST_MODE=1 ile (Docker `test`
hedefi ve CPU birim testleri) model yüklenmez; yalnız doğrulama, kapı ve
yardımcılar çalışır.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import unquote, urlsplit

import quality_gate

IMAGE_VERSION = os.environ.get("IMAGE_VERSION", "dev")
MUSETALK_COMMIT = os.environ.get("MUSETALK_COMMIT", "unknown")
MODEL_ADI = "musetalk-1.5"
TEST_MODU = os.environ.get("LIPSYNC_TEST_MODE") == "1"

log = logging.getLogger("lipsync")
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "info").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)

# ── Hata modeli ──────────────────────────────────────────────────────────

# Sözleşmedeki kodlar. retryable yalnız DOWNLOAD_FAILED, UPLOAD_FAILED, OOM.
HATA_KODLARI = {
    "INPUT_REJECTED",
    "DOWNLOAD_FAILED",
    "FACE_NOT_FOUND",
    "PREP_FAILED",
    "RENDER_FAILED",
    "OOM",
    "QUALITY_GATE",
    "UPLOAD_FAILED",
}
GECICI_HATA_KODLARI = {"DOWNLOAD_FAILED", "UPLOAD_FAILED", "OOM"}


class IsHatasi(Exception):
    """Ele alınan, yapılandırılmış hata. Handler bunu {ok:false,...} olarak döner."""

    def __init__(self, code: str, message: str, retryable: Optional[bool] = None):
        if code not in HATA_KODLARI:
            raise ValueError(f"bilinmeyen hata kodu: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = code in GECICI_HATA_KODLARI if retryable is None else retryable

    def sozluk(self) -> Dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "error": url_maskele(self.message),
            "retryable": self.retryable,
        }


# ── Yapılandırma ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Ayarlar:
    allowed_input_hosts: frozenset = field(default_factory=frozenset)
    allowed_upload_host_suffix: str = ".r2.cloudflarestorage.com"
    max_audio_seconds: float = 70.0
    max_source_seconds: float = 90.0
    max_height: int = 1080
    max_download_mb: int = 300
    cache_max_avatars: int = 10
    score_enabled: bool = False
    cache_dir: str = "/workspace/cache"
    download_timeout_s: int = 60
    upload_timeout_s: int = 300


def ayarlari_oku(env: Optional[Dict[str, str]] = None) -> Ayarlar:
    e = os.environ if env is None else env
    hostlar = frozenset(
        h.strip().lower() for h in e.get("ALLOWED_INPUT_HOSTS", "").split(",") if h.strip()
    )
    return Ayarlar(
        allowed_input_hosts=hostlar,
        allowed_upload_host_suffix=e.get("ALLOWED_UPLOAD_HOST_SUFFIX", ".r2.cloudflarestorage.com").strip().lower(),
        max_audio_seconds=float(e.get("MAX_AUDIO_SECONDS", "70")),
        max_source_seconds=float(e.get("MAX_SOURCE_SECONDS", "90")),
        max_height=int(e.get("MAX_HEIGHT", "1080")),
        max_download_mb=int(e.get("MAX_DOWNLOAD_MB", "300")),
        cache_max_avatars=int(e.get("CACHE_MAX_AVATARS", "10")),
        score_enabled=e.get("SCORE_ENABLED", "0") == "1",
        cache_dir=e.get("CACHE_DIR", "/workspace/cache"),
        download_timeout_s=int(e.get("DOWNLOAD_TIMEOUT_S", "60")),
        upload_timeout_s=int(e.get("UPLOAD_TIMEOUT_S", "300")),
    )


# ── URL hijyeni ──────────────────────────────────────────────────────────

_IMZA_DESENI = re.compile(r"([?&])X-Amz-[^\s]*")
_HASH_DESENI = re.compile(r"^sha256:[0-9a-f]{64}$")


def url_maskele(metin: str) -> str:
    """Log ve hata metinlerinde imzalı sorgu dizisini gizler."""
    return _IMZA_DESENI.sub(r"\1X-Amz-***", metin)


def _https_host(url: str, alan: str) -> str:
    try:
        parca = urlsplit(url)
    except ValueError as hata:
        raise IsHatasi("INPUT_REJECTED", f"{alan}: geçersiz URL ({hata})")
    if parca.scheme != "https":
        raise IsHatasi("INPUT_REJECTED", f"{alan}: yalnız https kabul edilir")
    if not parca.hostname:
        raise IsHatasi("INPUT_REJECTED", f"{alan}: host yok")
    if parca.username or parca.password:
        raise IsHatasi("INPUT_REJECTED", f"{alan}: URL'de kimlik bilgisi olamaz")
    return parca.hostname.lower()


def girdi_url_dogrula(url: Any, alan: str, ayar: Ayarlar) -> str:
    if not isinstance(url, str) or not url:
        raise IsHatasi("INPUT_REJECTED", f"{alan} zorunlu")
    host = _https_host(url, alan)
    if host not in ayar.allowed_input_hosts:
        raise IsHatasi("INPUT_REJECTED", f"{alan}: host izin listesinde değil ({host})")
    return url


def anahtar_dogrula(key: Any, alan: str) -> str:
    if not isinstance(key, str) or not key:
        raise IsHatasi("INPUT_REJECTED", f"{alan} zorunlu")
    if key.startswith("/") or "//" in key or ".." in key.split("/") or "\\" in key:
        raise IsHatasi("INPUT_REJECTED", f"{alan}: geçersiz yol")
    if not (key.startswith("sales/") or key.startswith("avatars/")):
        raise IsHatasi("INPUT_REJECTED", f"{alan}: yalnız sales/ veya avatars/ altına yazılabilir")
    return key


def yukleme_url_dogrula(url: Any, alan: str, key: str, ayar: Ayarlar) -> str:
    if not isinstance(url, str) or not url:
        raise IsHatasi("INPUT_REJECTED", f"{alan} zorunlu")
    host = _https_host(url, alan)
    if not host.endswith(ayar.allowed_upload_host_suffix):
        raise IsHatasi("INPUT_REJECTED", f"{alan}: yükleme host'u beklenen son ekle bitmiyor")
    yol = unquote(urlsplit(url).path)
    if not yol.endswith("/" + key):
        raise IsHatasi("INPUT_REJECTED", f"{alan}: yol {alan.replace('_put_url', '_key')} ile eşleşmiyor")
    return url


def _tam_sayi(deger: Any, alan: str, varsayilan: int, alt: int, ust: int) -> int:
    if deger is None:
        return varsayilan
    if isinstance(deger, bool) or not isinstance(deger, (int, float)) or int(deger) != deger:
        raise IsHatasi("INPUT_REJECTED", f"{alan}: tam sayı olmalı")
    d = int(deger)
    if d < alt or d > ust:
        raise IsHatasi("INPUT_REJECTED", f"{alan}: {alt}-{ust} aralığında olmalı")
    return d


VARSAYILAN_SECENEKLER: Dict[str, Any] = {
    "batch_size": 8,
    "extra_margin": 10,
    "parsing_mode": "jaw",
    "left_cheek_width": 90,
    "right_cheek_width": 90,
    "score": False,
}

VARSAYILAN_NORMLAR: Dict[str, Any] = {
    "min_face_px": 250,
    "max_face_px": 450,
    "max_height": 1080,
    "min_seconds": 15,
    "max_seconds": 90,
}


def secenekleri_dogrula(ham: Any) -> Dict[str, Any]:
    if ham is None:
        return dict(VARSAYILAN_SECENEKLER)
    if not isinstance(ham, dict):
        raise IsHatasi("INPUT_REJECTED", "options: nesne olmalı")
    s = dict(VARSAYILAN_SECENEKLER)
    s["batch_size"] = _tam_sayi(ham.get("batch_size"), "options.batch_size", 8, 1, 32)
    s["extra_margin"] = _tam_sayi(ham.get("extra_margin"), "options.extra_margin", 10, 0, 40)
    s["left_cheek_width"] = _tam_sayi(ham.get("left_cheek_width"), "options.left_cheek_width", 90, 20, 160)
    s["right_cheek_width"] = _tam_sayi(ham.get("right_cheek_width"), "options.right_cheek_width", 90, 20, 160)
    mod = ham.get("parsing_mode", "jaw")
    if mod not in ("jaw", "raw"):
        raise IsHatasi("INPUT_REJECTED", "options.parsing_mode: jaw veya raw")
    s["parsing_mode"] = mod
    s["score"] = bool(ham.get("score", False))
    return s


def girdiyi_dogrula(job_input: Any, ayar: Ayarlar) -> Dict[str, Any]:
    """İstek girdisini şemaya ve güvenlik çitine göre normalize eder.

    Dönen sözlük yalnız doğrulanmış alanları içerir; bilinmeyen alanlar atılır.
    Hata: IsHatasi(INPUT_REJECTED).
    """
    if not isinstance(job_input, dict):
        raise IsHatasi("INPUT_REJECTED", "input nesne olmalı")
    mod = job_input.get("mode", "render")
    if mod not in ("render", "prepare", "probe"):
        raise IsHatasi("INPUT_REJECTED", f"mode tanınmadı: {mod!r}")
    if mod == "probe":
        return {"mode": "probe"}

    fps = _tam_sayi(job_input.get("fps"), "fps", 25, 1, 60)
    if mod == "render":
        key = anahtar_dogrula(job_input.get("output_key"), "output_key")
        cikti: Dict[str, Any] = {
            "mode": "render",
            "audio_url": girdi_url_dogrula(job_input.get("audio_url"), "audio_url", ayar),
            "source_url": girdi_url_dogrula(job_input.get("source_url"), "source_url", ayar),
            "output_put_url": yukleme_url_dogrula(job_input.get("output_put_url"), "output_put_url", key, ayar),
            "output_key": key,
            "fps": fps,
            "options": secenekleri_dogrula(job_input.get("options")),
        }
        prep_url = job_input.get("prep_url")
        if prep_url:
            cikti["prep_url"] = girdi_url_dogrula(prep_url, "prep_url", ayar)
        source_hash = job_input.get("source_hash")
        if source_hash:
            if not isinstance(source_hash, str) or not _HASH_DESENI.match(source_hash):
                raise IsHatasi("INPUT_REJECTED", "source_hash: 'sha256:<64 hex>' biçiminde olmalı")
            cikti["source_hash"] = source_hash
        return cikti

    # prepare
    key = anahtar_dogrula(job_input.get("prep_key"), "prep_key")
    if not key.startswith("avatars/"):
        raise IsHatasi("INPUT_REJECTED", "prep_key: avatars/ altında olmalı")
    normlar = dict(VARSAYILAN_NORMLAR)
    ham_norm = job_input.get("norms")
    if ham_norm is not None:
        if not isinstance(ham_norm, dict):
            raise IsHatasi("INPUT_REJECTED", "norms: nesne olmalı")
        for ad in normlar:
            if ad in ham_norm:
                normlar[ad] = _tam_sayi(ham_norm[ad], f"norms.{ad}", normlar[ad], 1, 10000)
    cikti = {
        "mode": "prepare",
        "source_url": girdi_url_dogrula(job_input.get("source_url"), "source_url", ayar),
        "prep_put_url": yukleme_url_dogrula(job_input.get("prep_put_url"), "prep_put_url", key, ayar),
        "prep_key": key,
        "fps": fps,
        "norms": normlar,
    }
    beklenen = job_input.get("expected_source_hash")
    if beklenen:
        if not isinstance(beklenen, str) or not _HASH_DESENI.match(beklenen):
            raise IsHatasi("INPUT_REJECTED", "expected_source_hash: 'sha256:<64 hex>' biçiminde olmalı")
        cikti["expected_source_hash"] = beklenen
    return cikti


# ── Ağ: indirme ve yükleme (requests tembel import; testler ağ kullanmaz) ─


def indir(url: str, hedef: Path, ayar: Ayarlar) -> Dict[str, Any]:
    """URL'yi hedefe indirir; boyut tavanı ve Content-Length doğrulaması yapar,
    akış sırasında sha256 hesaplar. Yönlendirme takip edilmez."""
    import requests  # noqa: WPS433

    tavan = ayar.max_download_mb * 1024 * 1024
    ozet = hashlib.sha256()
    toplam = 0
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(10, ayar.download_timeout_s),
            allow_redirects=False,
            headers={"User-Agent": f"sales-lipsync/{IMAGE_VERSION}"},
        ) as yanit:
            if yanit.status_code != 200:
                raise IsHatasi("DOWNLOAD_FAILED", f"indirme HTTP {yanit.status_code}: {url_maskele(url)}")
            beyan = yanit.headers.get("Content-Length")
            if beyan is not None and int(beyan) > tavan:
                raise IsHatasi("INPUT_REJECTED", f"dosya {ayar.max_download_mb} MB tavanını aşıyor")
            with open(hedef, "wb") as f:
                for parca in yanit.iter_content(chunk_size=1024 * 1024):
                    if not parca:
                        continue
                    toplam += len(parca)
                    if toplam > tavan:
                        raise IsHatasi("INPUT_REJECTED", f"dosya {ayar.max_download_mb} MB tavanını aşıyor")
                    ozet.update(parca)
                    f.write(parca)
            if beyan is not None and int(beyan) != toplam:
                raise IsHatasi("DOWNLOAD_FAILED", "Content-Length ile indirilen boyut uyuşmuyor")
    except IsHatasi:
        raise
    except Exception as hata:  # ağ, zaman aşımı, disk
        raise IsHatasi("DOWNLOAD_FAILED", f"indirme hatası: {url_maskele(str(hata))}")
    if toplam == 0:
        raise IsHatasi("DOWNLOAD_FAILED", "boş dosya indirildi")
    return {"bytes": toplam, "sha256": "sha256:" + ozet.hexdigest()}


def yukle(put_url: str, kaynak: Path, content_type: str, ayar: Ayarlar) -> int:
    """Presigned PUT ile yükler. Content-Type ve Content-Length worker'dan
    gider (imzada yalnız host var, UNSIGNED-PAYLOAD)."""
    import requests  # noqa: WPS433

    boyut = kaynak.stat().st_size
    try:
        with open(kaynak, "rb") as f:
            yanit = requests.put(
                put_url,
                data=f,
                headers={"Content-Type": content_type, "Content-Length": str(boyut)},
                timeout=(10, ayar.upload_timeout_s),
                allow_redirects=False,
            )
    except Exception as hata:
        raise IsHatasi("UPLOAD_FAILED", f"yükleme hatası: {url_maskele(str(hata))}")
    if yanit.status_code not in (200, 201, 204):
        # 403: imza süresi dolmuş ya da yol uyuşmuyor; tekrar da başarısız olur
        # ama üst katman aynı sahne için yeni imza üretebilir, o yüzden geçici.
        raise IsHatasi("UPLOAD_FAILED", f"yükleme HTTP {yanit.status_code}")
    return boyut


# ── ffmpeg yardımcıları ──────────────────────────────────────────────────


def ffprobe(yol: Path) -> Dict[str, Any]:
    return quality_gate.ffprobe(yol)


def ses_wav_yap(girdi: Path, cikti: Path) -> None:
    """Sesi 16 kHz mono PCM wav'a çevirir (whisper girişi)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(girdi), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(cikti),
    ]
    sonuc = subprocess.run(cmd, capture_output=True, text=True)
    if sonuc.returncode != 0:
        raise IsHatasi("INPUT_REJECTED", f"ses çözülemedi: {sonuc.stderr.strip()[:300]}")


# ── Avatar önbelleği (disk LRU) ──────────────────────────────────────────


def onbellek_yolu(ayar: Ayarlar, source_hash: str) -> Path:
    return Path(ayar.cache_dir) / "avatars" / source_hash.replace("sha256:", "")


def onbellekte_var_mi(ayar: Ayarlar, source_hash: str) -> bool:
    d = onbellek_yolu(ayar, source_hash)
    return (d / "meta.json").is_file() and (d / "latents.pt").is_file()


def onbellek_dokun(ayar: Ayarlar, source_hash: str) -> None:
    d = onbellek_yolu(ayar, source_hash)
    d.mkdir(parents=True, exist_ok=True)
    (d / ".last_used").write_text(str(time.time()))


def onbellek_kirp(ayar: Ayarlar) -> int:
    """En eski kullanılan avatarları siler; CACHE_MAX_AVATARS kadar kalır."""
    kok = Path(ayar.cache_dir) / "avatars"
    if not kok.is_dir():
        return 0
    girdiler = []
    for d in kok.iterdir():
        if not d.is_dir():
            continue
        try:
            zaman = float((d / ".last_used").read_text())
        except Exception:
            zaman = 0.0
        girdiler.append((zaman, d))
    girdiler.sort()
    silinen = 0
    while len(girdiler) > ayar.cache_max_avatars:
        _, d = girdiler.pop(0)
        shutil.rmtree(d, ignore_errors=True)
        silinen += 1
    return silinen


def onbellek_avatarlari(ayar: Ayarlar) -> list:
    kok = Path(ayar.cache_dir) / "avatars"
    if not kok.is_dir():
        return []
    return sorted(d.name for d in kok.iterdir() if (d / "meta.json").is_file())


def paketi_ac(tar_yolu: Path, hedef: Path, beklenen_hash: str) -> None:
    """prep tar'ını güvenli açar (yol kaçışı yok) ve meta.json hash'ini doğrular."""
    hedef.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_yolu, "r:*") as tar:
            for uye in tar.getmembers():
                ad = Path(uye.name)
                if ad.is_absolute() or ".." in ad.parts or not (uye.isfile() or uye.isdir()):
                    raise IsHatasi("PREP_FAILED", f"paket içinde güvensiz yol: {uye.name}")
            tar.extractall(hedef)
        meta = json.loads((hedef / "meta.json").read_text())
    except IsHatasi:
        raise
    except Exception as hata:
        shutil.rmtree(hedef, ignore_errors=True)
        raise IsHatasi("PREP_FAILED", f"hazırlık paketi açılamadı: {hata}")
    if meta.get("source_hash") != beklenen_hash:
        shutil.rmtree(hedef, ignore_errors=True)
        raise IsHatasi("PREP_FAILED", "hazırlık paketi hash'i kaynak klip ile uyuşmuyor")


def paketi_yaz(prep_dir: Path, tar_yolu: Path) -> int:
    with tarfile.open(tar_yolu, "w") as tar:
        for dosya in sorted(prep_dir.iterdir()):
            if dosya.name.startswith("."):
                continue
            tar.add(dosya, arcname=dosya.name)
    return tar_yolu.stat().st_size


# ── Model (import anında; test modunda atlanır) ─────────────────────────

RUNNER = None
if not TEST_MODU:
    import musetalk_runner  # noqa: E402

    RUNNER = musetalk_runner.load_models()
    log.info("model yüklendi: %s commit=%s imaj=%s", MODEL_ADI, MUSETALK_COMMIT, IMAGE_VERSION)


def _ilerleme(job: Dict[str, Any], mesaj: str) -> None:
    """RunPod progress_update; kütüphane yoksa (test) sessiz."""
    try:
        import runpod  # noqa: WPS433

        runpod.serverless.progress_update(job, url_maskele(mesaj))
    except Exception:
        pass


def _gpu_bilgisi() -> Dict[str, Any]:
    try:
        import torch  # noqa: WPS433

        if not torch.cuda.is_available():
            return {"gpu": None, "vram_free_mb": None}
        bos, _toplam = torch.cuda.mem_get_info()
        return {"gpu": torch.cuda.get_device_name(0), "vram_free_mb": int(bos / (1024 * 1024))}
    except Exception:
        return {"gpu": None, "vram_free_mb": None}


def _oom_mu(hata: BaseException) -> bool:
    metin = str(hata).lower()
    return "out of memory" in metin or type(hata).__name__ == "OutOfMemoryError"


# ── Modlar ───────────────────────────────────────────────────────────────


def probe_modu(ayar: Ayarlar) -> Dict[str, Any]:
    bilgi = _gpu_bilgisi()
    return {
        "ok": True,
        "image": IMAGE_VERSION,
        "model": MODEL_ADI,
        "musetalk_commit": MUSETALK_COMMIT,
        "gpu": bilgi["gpu"],
        "vram_free_mb": bilgi["vram_free_mb"],
        "cache_avatars": onbellek_avatarlari(ayar),
        "score_enabled": ayar.score_enabled,
        "test_mode": TEST_MODU,
    }


def _kaynak_bilgisi(yol: Path, ayar: Ayarlar) -> Dict[str, Any]:
    bilgi = quality_gate.video_bilgisi(ffprobe(yol))
    if bilgi is None:
        raise IsHatasi("INPUT_REJECTED", "kaynak klipte video akışı yok")
    if bilgi["duration"] > ayar.max_source_seconds:
        raise IsHatasi("INPUT_REJECTED", f"kaynak klip {ayar.max_source_seconds:.0f} sn tavanını aşıyor")
    if bilgi["height"] > ayar.max_height:
        raise IsHatasi("INPUT_REJECTED", f"kaynak yükseklik {ayar.max_height} px tavanını aşıyor")
    if bilgi["width"] % 2 or bilgi["height"] % 2:
        raise IsHatasi("INPUT_REJECTED", "kaynak genişlik ve yükseklik çift sayı olmalı (yuv420p)")
    return bilgi


def _hazirligi_getir(
    job: Dict[str, Any],
    girdi: Dict[str, Any],
    ayar: Ayarlar,
    source_hash: str,
    kaynak_yolu: Path,
    is_dizini: Path,
    zaman: Dict[str, int],
) -> Dict[str, Any]:
    """Arama sırası: disk → prep_url → kendisi hesapla (yalnız diske yazar)."""
    t = time.time()
    if onbellekte_var_mi(ayar, source_hash):
        onbellek_dokun(ayar, source_hash)
        zaman["prep"] = 0
        return {"dir": onbellek_yolu(ayar, source_hash), "hit": True, "source": "disk"}

    if girdi.get("prep_url"):
        _ilerleme(job, "prep paketi indiriliyor")
        tar_yolu = is_dizini / "prep.tar"
        try:
            indir(girdi["prep_url"], tar_yolu, ayar)
            paketi_ac(tar_yolu, onbellek_yolu(ayar, source_hash), source_hash)
            onbellek_dokun(ayar, source_hash)
            onbellek_kirp(ayar)
            zaman["prep"] = int((time.time() - t) * 1000)
            return {"dir": onbellek_yolu(ayar, source_hash), "hit": True, "source": "r2"}
        except IsHatasi as hata:
            # Paket bozuk ya da inmedi: yerelde hesaplamaya düş, ama logla.
            log.warning("prep paketi kullanılamadı, yerelde hesaplanacak: %s", hata.message)

    _ilerleme(job, "avatar hazırlığı hesaplanıyor")
    hedef = onbellek_yolu(ayar, source_hash)
    gecici = Path(str(hedef) + ".tmp")
    shutil.rmtree(gecici, ignore_errors=True)
    gecici.mkdir(parents=True, exist_ok=True)
    try:
        RUNNER.prepare(
            source_path=str(kaynak_yolu),
            out_dir=str(gecici),
            fps=girdi["fps"],
            source_hash=source_hash,
            options=girdi["options"],
            meta_ek={"musetalk_commit": MUSETALK_COMMIT, "image": IMAGE_VERSION},
        )
    except IsHatasi:
        shutil.rmtree(gecici, ignore_errors=True)
        raise
    except Exception as hata:
        shutil.rmtree(gecici, ignore_errors=True)
        if _oom_mu(hata):
            raise IsHatasi("OOM", f"hazırlıkta bellek yetmedi: {hata}")
        raise IsHatasi("PREP_FAILED", f"hazırlık hesaplanamadı: {hata}")
    shutil.rmtree(hedef, ignore_errors=True)
    gecici.rename(hedef)
    onbellek_dokun(ayar, source_hash)
    onbellek_kirp(ayar)
    zaman["prep"] = int((time.time() - t) * 1000)
    return {"dir": hedef, "hit": False, "source": "computed"}


def render_modu(job: Dict[str, Any], girdi: Dict[str, Any], ayar: Ayarlar) -> Dict[str, Any]:
    if RUNNER is None:
        raise RuntimeError("model yüklü değil (LIPSYNC_TEST_MODE)")
    t0 = time.time()
    zaman: Dict[str, int] = {}
    is_dizini = Path(tempfile.mkdtemp(prefix="render-", dir=os.environ.get("JOB_TMP_DIR")))
    try:
        # 1) İndirme
        _ilerleme(job, "girdiler indiriliyor")
        t = time.time()
        ses_ham = is_dizini / "audio.bin"
        kaynak = is_dizini / "source.mp4"
        indir(girdi["audio_url"], ses_ham, ayar)
        kaynak_bilgi = indir(girdi["source_url"], kaynak, ayar)
        zaman["download"] = int((time.time() - t) * 1000)

        source_hash = kaynak_bilgi["sha256"]
        if girdi.get("source_hash") and girdi["source_hash"] != source_hash:
            raise IsHatasi("INPUT_REJECTED", "source_hash kaynak klip ile uyuşmuyor")

        # 2) Doğrulama
        ses_bilgi = quality_gate.ses_bilgisi(ffprobe(ses_ham))
        if ses_bilgi is None:
            raise IsHatasi("INPUT_REJECTED", "seste ses akışı yok")
        if ses_bilgi["duration"] > ayar.max_audio_seconds:
            raise IsHatasi("INPUT_REJECTED", f"ses {ayar.max_audio_seconds:.0f} sn tavanını aşıyor")
        video_bilgi = _kaynak_bilgisi(kaynak, ayar)
        wav = is_dizini / "audio.wav"
        ses_wav_yap(ses_ham, wav)

        # 3) Hazırlık paketi
        hazirlik = _hazirligi_getir(job, girdi, ayar, source_hash, kaynak, is_dizini, zaman)

        # 4) Üretim + kodlama
        _ilerleme(job, "lip-sync üretiliyor")
        cikti = is_dizini / "out.mp4"
        try:
            uretim = RUNNER.synthesize(
                prep_dir=str(hazirlik["dir"]),
                source_path=str(kaynak),
                audio_wav=str(wav),
                out_mp4=str(cikti),
                fps=girdi["fps"],
                options=girdi["options"],
            )
        except IsHatasi:
            raise
        except Exception as hata:
            if _oom_mu(hata):
                raise IsHatasi("OOM", f"üretimde bellek yetmedi: {hata}")
            raise IsHatasi("RENDER_FAILED", f"üretim hatası: {hata}")
        zaman.update({k: int(v) for k, v in uretim.get("timings_ms", {}).items()})

        # 5) Kalite kapısı (deterministik)
        _ilerleme(job, "kalite kapısı")
        t = time.time()
        meta = json.loads((Path(hazirlik["dir"]) / "meta.json").read_text())
        kapi = quality_gate.kontrol(
            cikti,
            audio_seconds=ses_bilgi["duration"],
            source_width=video_bilgi["width"],
            source_height=video_bilgi["height"],
            fps=girdi["fps"],
            face_frames_ratio=meta.get("face_frames_ratio"),
        )
        zaman["gate"] = int((time.time() - t) * 1000)
        if not kapi.ok:
            raise IsHatasi(kapi.code, kapi.error)

        kalite: Dict[str, Any] = dict(kapi.metrics)
        kalite.update({"gate": "pass", "lse_c": None, "lse_d": None, "sharpness_ratio": None})
        if ayar.score_enabled and girdi["options"].get("score"):
            try:
                import score  # noqa: WPS433

                kalite.update(score.hesapla(str(cikti), str(kaynak), str(wav)))
            except Exception as hata:  # ölçüm katmanı hiçbir zaman render'ı düşürmez
                log.warning("skor hesaplanamadı: %s", hata)

        # 6) Yükleme
        _ilerleme(job, "R2'ye yükleniyor")
        t = time.time()
        boyut = yukle(girdi["output_put_url"], cikti, "video/mp4", ayar)
        zaman["upload"] = int((time.time() - t) * 1000)
        zaman["total"] = int((time.time() - t0) * 1000)

        bilgi = _gpu_bilgisi()
        return {
            "ok": True,
            "key": girdi["output_key"],
            "duration_seconds": kapi.metrics["output_duration"],
            "frames": kapi.metrics["output_frames"],
            "fps": girdi["fps"],
            "width": video_bilgi["width"],
            "height": video_bilgi["height"],
            "bytes": boyut,
            "model": MODEL_ADI,
            "prep_cache_hit": hazirlik["hit"],
            "prep_source": hazirlik["source"],
            "timings_ms": zaman,
            "gpu_seconds": round(zaman["total"] / 1000.0, 2),
            "quality": {
                "gate": "pass",
                "duration_delta_ms": kalite.get("duration_delta_ms"),
                "frame_delta": kalite.get("frame_delta"),
                "face_frames_ratio": meta.get("face_frames_ratio"),
                "lse_c": kalite.get("lse_c"),
                "lse_d": kalite.get("lse_d"),
                "sharpness_ratio": kalite.get("sharpness_ratio"),
            },
            "worker": {"gpu": bilgi["gpu"], "image": IMAGE_VERSION},
        }
    finally:
        shutil.rmtree(is_dizini, ignore_errors=True)


def prepare_modu(job: Dict[str, Any], girdi: Dict[str, Any], ayar: Ayarlar) -> Dict[str, Any]:
    if RUNNER is None:
        raise RuntimeError("model yüklü değil (LIPSYNC_TEST_MODE)")
    t0 = time.time()
    zaman: Dict[str, int] = {}
    is_dizini = Path(tempfile.mkdtemp(prefix="prepare-", dir=os.environ.get("JOB_TMP_DIR")))
    try:
        _ilerleme(job, "kaynak klip indiriliyor")
        t = time.time()
        kaynak = is_dizini / "source.mp4"
        kaynak_bilgi = indir(girdi["source_url"], kaynak, ayar)
        zaman["download"] = int((time.time() - t) * 1000)
        source_hash = kaynak_bilgi["sha256"]
        if girdi.get("expected_source_hash") and girdi["expected_source_hash"] != source_hash:
            raise IsHatasi("INPUT_REJECTED", "expected_source_hash kaynak klip ile uyuşmuyor")

        normlar = girdi["norms"]
        bilgi = _kaynak_bilgisi(kaynak, ayar)
        if bilgi["duration"] < normlar["min_seconds"] or bilgi["duration"] > normlar["max_seconds"]:
            raise IsHatasi(
                "INPUT_REJECTED",
                f"kaynak süre {bilgi['duration']:.1f} sn, norm {normlar['min_seconds']}-{normlar['max_seconds']} sn",
            )
        if bilgi["height"] > normlar["max_height"]:
            raise IsHatasi("INPUT_REJECTED", f"kaynak yükseklik {bilgi['height']} > {normlar['max_height']}")
        if bilgi["fps"] and abs(bilgi["fps"] - girdi["fps"]) > 0.01:
            raise IsHatasi(
                "INPUT_REJECTED",
                f"kaynak kare hızı {bilgi['fps']:.3f}, beklenen {girdi['fps']} (önce ffmpeg ile normalize edin)",
            )

        _ilerleme(job, "avatar hazırlığı hesaplanıyor")
        t = time.time()
        prep_dir = is_dizini / "prep"
        prep_dir.mkdir()
        try:
            ozet = RUNNER.prepare(
                source_path=str(kaynak),
                out_dir=str(prep_dir),
                fps=girdi["fps"],
                source_hash=source_hash,
                options=dict(VARSAYILAN_SECENEKLER),
                meta_ek={"musetalk_commit": MUSETALK_COMMIT, "image": IMAGE_VERSION},
            )
        except IsHatasi:
            raise
        except Exception as hata:
            if _oom_mu(hata):
                raise IsHatasi("OOM", f"hazırlıkta bellek yetmedi: {hata}")
            raise IsHatasi("PREP_FAILED", f"hazırlık hesaplanamadı: {hata}")
        zaman["prep"] = int((time.time() - t) * 1000)

        if ozet["face_frames_ratio"] < 1.0:
            raise IsHatasi(
                "FACE_NOT_FOUND",
                f"Kaynak klipte {ozet['frames'] - ozet['face_frames']} karede yüz bulunamadı",
            )
        kutu = ozet["face_box_px"]
        if kutu["min"] < normlar["min_face_px"] or kutu["max"] > normlar["max_face_px"]:
            raise IsHatasi(
                "INPUT_REJECTED",
                f"yüz kutusu yüksekliği {kutu['min']}-{kutu['max']} px, norm {normlar['min_face_px']}-{normlar['max_face_px']} px",
            )

        # Sıcak worker bir sonraki render'da diskten okusun.
        hedef = onbellek_yolu(ayar, source_hash)
        shutil.rmtree(hedef, ignore_errors=True)
        hedef.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(prep_dir, hedef)
        onbellek_dokun(ayar, source_hash)
        onbellek_kirp(ayar)

        _ilerleme(job, "paket R2'ye yükleniyor")
        t = time.time()
        tar_yolu = is_dizini / "prep.tar"
        paketi_yaz(prep_dir, tar_yolu)
        boyut = yukle(girdi["prep_put_url"], tar_yolu, "application/x-tar", ayar)
        zaman["upload"] = int((time.time() - t) * 1000)
        zaman["total"] = int((time.time() - t0) * 1000)
        return {
            "ok": True,
            "source_hash": source_hash,
            "key": girdi["prep_key"],
            "frames": ozet["frames"],
            "width": bilgi["width"],
            "height": bilgi["height"],
            "fps": girdi["fps"],
            "duration_seconds": bilgi["duration"],
            "face_frames_ratio": ozet["face_frames_ratio"],
            "face_box_px": kutu,
            "bytes": boyut,
            "timings_ms": zaman,
        }
    finally:
        shutil.rmtree(is_dizini, ignore_errors=True)


# ── Handler ──────────────────────────────────────────────────────────────


def handler(job: Dict[str, Any], ayar: Optional[Ayarlar] = None) -> Dict[str, Any]:
    """RunPod giriş noktası. Ele alınan hatalar {ok:false,...} olarak döner;
    beklenmedik istisna fırlatılır (RunPod job'ı FAILED işaretler)."""
    ayar = ayar or ayarlari_oku()
    job_id = job.get("id", "?")
    try:
        girdi = girdiyi_dogrula(job.get("input"), ayar)
    except IsHatasi as hata:
        log.warning("job %s girdi reddedildi: %s", job_id, hata.message)
        return hata.sozluk()

    mod = girdi["mode"]
    log.info("job %s mode=%s", job_id, mod)
    try:
        if mod == "probe":
            return probe_modu(ayar)
        if mod == "render":
            sonuc = render_modu(job, girdi, ayar)
        else:
            sonuc = prepare_modu(job, girdi, ayar)
        log.info("job %s tamam: %s", job_id, json.dumps({k: sonuc.get(k) for k in ("key", "duration_seconds", "prep_cache_hit", "timings_ms")}))
        return sonuc
    except IsHatasi as hata:
        log.warning("job %s hata %s (retryable=%s): %s", job_id, hata.code, hata.retryable, hata.message)
        return hata.sozluk()
    finally:
        # Job'lar arası VRAM parçalanmasını sınırlamak için.
        try:
            import torch  # noqa: WPS433

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    import runpod  # noqa: WPS433

    runpod.serverless.start({"handler": handler})
