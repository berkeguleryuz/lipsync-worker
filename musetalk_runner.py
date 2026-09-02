# -*- coding: utf-8 -*-
"""MuseTalk 1.5 çalıştırıcısı: model yükleme, avatar hazırlığı, üretim.

`scripts/realtime_inference.py` (sabit commit) akışını iki aşamaya böler:

  prepare(source_path, out_dir)
      Kaynak klibin her karesi için yüz kutusu (DWPose işaretleri + S3FD
      dedektörü), yüz-ayrıştırma maskesi ve VAE latent'i hesaplar; hazırlık
      paketini `out_dir` altına yazar: latents.pt, coords.pkl,
      mask_coords.pkl, masks.npz, meta.json. Kaynak KARELER pakete girmez.

  synthesize(prep_dir, source_path, audio_wav, out_mp4)
      Whisper özelliklerinden UNet ile 256x256 ağız bölgesini üretir, kaynak
      kareye yapıştırır (blend) ve kareleri ffmpeg stdin'e rawvideo olarak
      basar: tek geçişte H.264 yuv420p CRF 18 + AAC 128k. Blend/encode ayrı
      iş parçacığında.

Bellek: kaynak kareler RAM'de JPEG (kalite 95) olarak tutulur ve blend
sırasında açılır. 60 sn 1080p kaynak ham olarak ~5 GB tutar, JPEG ~300 MB.
Kalite 95 çıktıdaki CRF 18 H.264'ün altında kalır; FRAME_STORE_JPEG_QUALITY
ile ayarlanır (100 = kayıpsız değil, en az kayıp).

Tüm ağır import'lar `load_models()` içinde: modül CPU'da (test) import
edilebilir. Çalışma dizini MuseTalk deposu olmalı; MuseTalk'ın yardımcı
modülleri model yollarını `./models/...` olarak sabit kodluyor.
"""
from __future__ import annotations

import json
import os
import pickle
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MUSETALK_DIR = os.environ.get("MUSETALK_DIR", "/app/MuseTalk")
JPEG_KALITE = int(os.environ.get("FRAME_STORE_JPEG_QUALITY", "95"))
LATENT_YAN = 256


class YuzBulunamadi(Exception):
    """Hazırlıkta en az bir karede yüz yok; handler FACE_NOT_FOUND'a çevirir."""


# ── Kare deposu ──────────────────────────────────────────────────────────


class KareDeposu:
    """Kareleri JPEG bayt dizisi olarak tutar, istenince açar."""

    def __init__(self, kalite: int = JPEG_KALITE):
        import cv2  # noqa: WPS433

        self._cv2 = cv2
        self._param = [int(cv2.IMWRITE_JPEG_QUALITY), int(kalite)]
        self._kareler: List[bytes] = []
        self.width = 0
        self.height = 0

    def ekle(self, kare) -> None:
        if not self._kareler:
            self.height, self.width = kare.shape[:2]
        ok, buf = self._cv2.imencode(".jpg", kare, self._param)
        if not ok:
            raise RuntimeError("kare JPEG'e kodlanamadı")
        self._kareler.append(buf.tobytes())

    def __len__(self) -> int:
        return len(self._kareler)

    def al(self, i: int):
        import numpy as np  # noqa: WPS433

        return self._cv2.imdecode(np.frombuffer(self._kareler[i], dtype=np.uint8), self._cv2.IMREAD_COLOR)


def kareleri_oku(source_path: str, fps: int, en_fazla: Optional[int] = None) -> KareDeposu:
    """ffmpeg ile kaynağı `fps`'e normalize ederek bgr24 rawvideo okur."""
    import numpy as np  # noqa: WPS433

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", source_path],
        capture_output=True, text=True, check=True,
    )
    akis = json.loads(probe.stdout)["streams"][0]
    w, h = int(akis["width"]), int(akis["height"])
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", source_path,
        "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1",
    ]
    depo = KareDeposu()
    kare_bayt = w * h * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=kare_bayt * 4) as p:
        assert p.stdout is not None
        while True:
            if en_fazla is not None and len(depo) >= en_fazla:
                p.terminate()
                break
            ham = p.stdout.read(kare_bayt)
            if len(ham) < kare_bayt:
                break
            depo.ekle(np.frombuffer(ham, dtype=np.uint8).reshape(h, w, 3))
    if len(depo) == 0:
        raise RuntimeError("kaynaktan hiç kare okunamadı")
    return depo


# ── Model ────────────────────────────────────────────────────────────────


class Runner:
    def __init__(self) -> None:
        import torch  # noqa: WPS433

        os.chdir(MUSETALK_DIR)
        import sys

        if MUSETALK_DIR not in sys.path:
            sys.path.insert(0, MUSETALK_DIR)

        from musetalk.utils.utils import load_all_model  # noqa: WPS433
        from musetalk.utils.audio_processor import AudioProcessor  # noqa: WPS433
        from musetalk.utils.face_parsing import FaceParsing  # noqa: WPS433
        from transformers import WhisperModel  # noqa: WPS433

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae, unet, pe = load_all_model(
            unet_model_path="./models/musetalkV15/unet.pth",
            vae_type="sd-vae",
            unet_config="./models/musetalkV15/musetalk.json",
            device=self.device,
        )
        self.timesteps = torch.tensor([0], device=self.device)
        # fp16: 24 GB kademede rahat, 16 GB yedek (tasarım 1.2).
        self.pe = pe.half().to(self.device)
        self.vae = vae
        self.vae.vae = vae.vae.half().to(self.device)
        self.unet = unet
        self.unet.model = unet.model.half().to(self.device)
        self.weight_dtype = self.unet.model.dtype
        self.audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
        self.whisper = WhisperModel.from_pretrained("./models/whisper")
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)
        self._face_parsing_cls = FaceParsing
        self._fp_cache: Dict[Tuple[int, int], Any] = {}
        # DWPose ve S3FD modülleri import anında ağırlık yükler (musetalk.utils.preprocessing).
        import musetalk.utils.preprocessing as preprocessing  # noqa: WPS433

        self.preprocessing = preprocessing

    def _face_parser(self, left: int, right: int):
        anahtar = (left, right)
        if anahtar not in self._fp_cache:
            self._fp_cache[anahtar] = self._face_parsing_cls(left_cheek_width=left, right_cheek_width=right)
        return self._fp_cache[anahtar]

    # ── Yüz kutusu (realtime_inference / preprocessing.get_landmark_and_bbox eşdeğeri) ──

    def _yuz_kutusu(self, kare) -> Optional[Tuple[int, int, int, int]]:
        """DWPose işaretleriyle daraltılmış kutu; dedektör yüz bulamazsa None.

        Üst sınır burun köprüsü (işaret 29) çevresinden, alt ve yanlar
        işaretlerin uç noktalarından. İşaret kutusu geçersizse S3FD kutusu."""
        import numpy as np  # noqa: WPS433
        from mmpose.apis import inference_topdown  # noqa: WPS433
        from mmpose.structures import merge_data_samples  # noqa: WPS433

        pre = self.preprocessing
        sonuclar = inference_topdown(pre.model, kare)
        sonuclar = merge_data_samples(sonuclar)
        keypoints = sonuclar.pred_instances.keypoints
        if keypoints is None or len(keypoints) == 0:
            return None
        yuz = keypoints[0][23:91].astype(np.int32)
        tespit = pre.fa.get_detections_for_batch(np.asarray([kare]))
        f = tespit[0] if tespit else None
        if f is None:
            return None
        yarim_yuz = yuz[29].copy()
        yarim_mesafe = int(np.max(yuz[:, 1]) - yarim_yuz[1])
        ust = int(yarim_yuz[1] - yarim_mesafe)
        x1, y1, x2, y2 = int(np.min(yuz[:, 0])), ust, int(np.max(yuz[:, 0])), int(np.max(yuz[:, 1]))
        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            return tuple(int(v) for v in f)  # type: ignore[return-value]
        return (x1, y1, x2, y2)

    # ── prepare ──

    def prepare(
        self,
        source_path: str,
        out_dir: str,
        fps: int,
        source_hash: str,
        options: Dict[str, Any],
        meta_ek: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import cv2  # noqa: WPS433
        import numpy as np  # noqa: WPS433
        from musetalk.utils.blending import get_image_prepare_material  # noqa: WPS433

        t0 = time.time()
        depo = kareleri_oku(source_path, fps)
        extra_margin = int(options.get("extra_margin", 10))
        parsing_mode = options.get("parsing_mode", "jaw")
        fp = self._face_parser(int(options.get("left_cheek_width", 90)), int(options.get("right_cheek_width", 90)))

        coords: List[Tuple[int, int, int, int]] = []
        latents = []
        mask_coords = []
        masks: Dict[str, Any] = {}
        yuz_yukseklikleri: List[int] = []
        yuz_yok = 0

        with self.torch.no_grad():
            for i in range(len(depo)):
                kare = depo.al(i)
                kutu = self._yuz_kutusu(kare)
                if kutu is None:
                    yuz_yok += 1
                    coords.append((0, 0, 0, 0))
                    continue
                x1, y1, x2, y2 = kutu
                y2 = min(y2 + extra_margin, kare.shape[0])
                coords.append((x1, y1, x2, y2))
                yuz_yukseklikleri.append(y2 - y1)
                crop = cv2.resize(kare[y1:y2, x1:x2], (LATENT_YAN, LATENT_YAN), interpolation=cv2.INTER_LANCZOS4)
                latents.append(self.vae.get_latents_for_unet(crop).cpu())
                mask, crop_box = get_image_prepare_material(kare, [x1, y1, x2, y2], fp=fp, mode=parsing_mode)
                masks[f"m{i}"] = mask.astype(np.uint8)
                mask_coords.append(tuple(int(v) for v in crop_box))

        kare_sayisi = len(depo)
        yuzlu = kare_sayisi - yuz_yok
        if yuzlu == 0:
            raise YuzBulunamadi("hiçbir karede yüz bulunamadı")

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if latents:
            self.torch.save(self.torch.cat(latents, dim=0), out / "latents.pt")
        with open(out / "coords.pkl", "wb") as f:
            pickle.dump(coords, f)
        with open(out / "mask_coords.pkl", "wb") as f:
            pickle.dump(mask_coords, f)
        np.savez_compressed(out / "masks.npz", **masks)
        yuz_oran = yuzlu / kare_sayisi
        kutu_ozeti = {
            "min": int(min(yuz_yukseklikleri)) if yuz_yukseklikleri else 0,
            "median": int(np.median(yuz_yukseklikleri)) if yuz_yukseklikleri else 0,
            "max": int(max(yuz_yukseklikleri)) if yuz_yukseklikleri else 0,
        }
        meta = {
            "source_hash": source_hash,
            "fps": fps,
            "width": depo.width,
            "height": depo.height,
            "frames": kare_sayisi,
            "face_frames": yuzlu,
            "face_frames_ratio": round(yuz_oran, 4),
            "face_box_px": kutu_ozeti,
            "extra_margin": extra_margin,
            "parsing_mode": parsing_mode,
            "left_cheek_width": int(options.get("left_cheek_width", 90)),
            "right_cheek_width": int(options.get("right_cheek_width", 90)),
            "prep_ms": int((time.time() - t0) * 1000),
            **(meta_ek or {}),
        }
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        return {
            "frames": kare_sayisi,
            "face_frames": yuzlu,
            "face_frames_ratio": yuz_oran,
            "face_box_px": kutu_ozeti,
        }

    # ── synthesize ──

    def synthesize(
        self,
        prep_dir: str,
        source_path: str,
        audio_wav: str,
        out_mp4: str,
        fps: int,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        import cv2  # noqa: WPS433
        import numpy as np  # noqa: WPS433
        from musetalk.utils.blending import get_image_blending  # noqa: WPS433
        from musetalk.utils.utils import datagen  # noqa: WPS433

        zaman: Dict[str, int] = {}
        prep = Path(prep_dir)
        meta = json.loads((prep / "meta.json").read_text())
        if meta.get("face_frames_ratio", 0) < 1.0:
            raise YuzBulunamadi("hazırlık paketinde yüzsüz kare var")

        # 1) Ses özellikleri
        t = time.time()
        with self.torch.no_grad():
            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
                audio_wav, weight_dtype=self.weight_dtype,
            )
            whisper_chunks = self.audio_processor.get_whisper_chunk(
                whisper_input_features, self.device, self.weight_dtype, self.whisper, librosa_length,
                fps=fps, audio_padding_length_left=2, audio_padding_length_right=2,
            )
        cikti_kare = len(whisper_chunks)
        zaman["audio_features"] = int((time.time() - t) * 1000)

        # 2) Hazırlık paketi ve kaynak kareler (yalnız gerekli sayıda)
        t = time.time()
        latents = self.torch.load(prep / "latents.pt", map_location="cpu")
        with open(prep / "coords.pkl", "rb") as f:
            coords: List[Tuple[int, int, int, int]] = pickle.load(f)
        with open(prep / "mask_coords.pkl", "rb") as f:
            mask_coords: List[Tuple[int, int, int, int]] = pickle.load(f)
        masks_npz = np.load(prep / "masks.npz")
        kaynak_kare = int(meta["frames"])
        # Palindrom döngü: 0..S-1, S-1..0. Kaynaktan en fazla S kare gerekir.
        gerekli = min(cikti_kare, kaynak_kare)
        depo = kareleri_oku(source_path, fps, en_fazla=gerekli)
        if len(depo) < gerekli:
            raise RuntimeError(f"kaynaktan {gerekli} kare bekleniyordu, {len(depo)} okundu")
        dongu = list(range(kaynak_kare)) + list(range(kaynak_kare - 1, -1, -1))
        latent_dongu = [latents[i : i + 1] for i in range(kaynak_kare)]
        latent_dongu = latent_dongu + latent_dongu[::-1]
        zaman["load_prep"] = int((time.time() - t) * 1000)

        # 3) Kodlayıcı (ffmpeg stdin rawvideo) ve blend iş parçacığı
        w, h = int(meta["width"]), int(meta["height"])
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
            "-i", audio_wav,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart",
            out_mp4,
        ]
        kodlayici = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        kuyruk: "queue.Queue[Any]" = queue.Queue(maxsize=64)
        hata_kutusu: List[BaseException] = []
        blend_ms = [0.0]

        def blend_ve_yaz() -> None:
            idx = 0
            try:
                while True:
                    res = kuyruk.get()
                    if res is None:
                        break
                    t1 = time.time()
                    kaynak_idx = dongu[idx % len(dongu)]
                    x1, y1, x2, y2 = coords[kaynak_idx]
                    kare = depo.al(kaynak_idx)
                    res_kare = cv2.resize(res.astype(np.uint8), (x2 - x1, y2 - y1))
                    mask = masks_npz[f"m{kaynak_idx}"]
                    birlesik = get_image_blending(kare, res_kare, [x1, y1, x2, y2], mask, mask_coords[kaynak_idx])
                    kodlayici.stdin.write(np.ascontiguousarray(birlesik).tobytes())  # type: ignore[union-attr]
                    blend_ms[0] += (time.time() - t1) * 1000
                    idx += 1
            except BaseException as hata:  # noqa: BLE001
                hata_kutusu.append(hata)

        yazici = threading.Thread(target=blend_ve_yaz, daemon=True)
        yazici.start()

        # 4) UNet üretimi
        t = time.time()
        batch = int(options.get("batch_size", 8))
        try:
            with self.torch.no_grad():
                gen = datagen(whisper_chunks, latent_dongu, batch, delay_frame=0, device=self.device)
                for whisper_batch, latent_batch in gen:
                    if hata_kutusu:
                        break
                    audio_feature_batch = self.pe(whisper_batch.to(self.device))
                    latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
                    pred_latents = self.unet.model(
                        latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch,
                    ).sample
                    pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                    recon = self.vae.decode_latents(pred_latents)
                    for res_frame in recon:
                        kuyruk.put(res_frame)
        finally:
            kuyruk.put(None)
            yazici.join()
        zaman["generate"] = int((time.time() - t) * 1000)
        zaman["blend"] = int(blend_ms[0])

        t = time.time()
        # stdin BURADA KAPATILMAZ: CPython'ın communicate() uygulaması stdin'i
        # önce flush edip sonra kendisi kapatır; elle kapatılmış bir stdin
        # "ValueError: flush of closed file" fırlatır ve gerçek sonucu maskeler
        # (canlıda her render bu hatayla düşüyordu). Blend iş parçacığı yazarken
        # ffmpeg ölmüşse BrokenPipe hata_kutusu'na düşer, aşağıda raporlanır.
        try:
            _, stderr = kodlayici.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            kodlayici.kill()
            _, stderr = kodlayici.communicate()
            raise RuntimeError("ffmpeg kodlama zaman aşımı (600 sn)")
        except ValueError as hata:
            # Beklenmedik bir yolda stdin yine kapalıysa süreci bekle, hatayı sakla.
            kodlayici.wait()
            stderr = b""
            hata_kutusu.append(hata)
        zaman["encode"] = int((time.time() - t) * 1000)
        if hata_kutusu:
            raise RuntimeError(
                f"blend/encode iş parçacığı hatası: {hata_kutusu[0]}"
                + (f" · ffmpeg: {stderr.decode('utf-8', 'ignore').strip()[:200]}" if stderr else "")
            )
        if kodlayici.returncode != 0:
            raise RuntimeError(f"ffmpeg kodlama hatası: {stderr.decode('utf-8', 'ignore').strip()[:300]}")
        return {"frames": cikti_kare, "timings_ms": zaman}


def load_models() -> Runner:
    return Runner()
