# -*- coding: utf-8 -*-
"""K10: kalite kapısı. Sahte ffprobe çıktısıyla CPU'da; ffmpeg varsa ayrıca
gerçek üretilmiş mp4 ile."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import quality_gate as qg  # noqa: E402


def sahte_probe(sure=10.0, kare=250, w=1080, h=1080, vcodec="h264", acodec="aac", pix="yuv420p", ses_sure=None):
    return {
        "streams": [
            {
                "codec_type": "video", "codec_name": vcodec, "pix_fmt": pix, "width": w, "height": h,
                "duration": str(sure), "nb_frames": str(kare), "avg_frame_rate": "25/1",
            },
            {"codec_type": "audio", "codec_name": acodec, "duration": str(ses_sure if ses_sure is not None else sure), "sample_rate": "48000"},
        ],
        "format": {"duration": str(sure)},
    }


class Kapi(unittest.TestCase):
    def kontrol(self, probe, **ek):
        args = dict(audio_seconds=10.0, source_width=1080, source_height=1080, fps=25, dosya_boyutu=1_000_000)
        args.update(ek)
        return qg.kontrol(Path("/tmp/yok.mp4"), probe_fn=lambda _p: probe, **args)

    def test_uyumlu_gecer(self):
        s = self.kontrol(sahte_probe())
        self.assertTrue(s.ok, s.error)
        self.assertEqual(s.metrics["duration_delta_ms"], 0)
        self.assertEqual(s.metrics["frame_delta"], 0)
        self.assertEqual(s.metrics["output_frames"], 250)

    def test_tolerans_icinde_gecer(self):
        s = self.kontrol(sahte_probe(sure=10.2, kare=252))
        self.assertTrue(s.ok, s.error)

    def test_400ms_sapma_kalite_kapisi(self):
        s = self.kontrol(sahte_probe(sure=10.4, kare=260))
        self.assertFalse(s.ok)
        self.assertEqual(s.code, "QUALITY_GATE")
        self.assertIn("400", s.error)

    def test_kare_sapmasi(self):
        s = self.kontrol(sahte_probe(sure=10.0, kare=246))
        self.assertEqual(s.code, "QUALITY_GATE")

    def test_cozunurluk_farki(self):
        s = self.kontrol(sahte_probe(w=720, h=720))
        self.assertEqual(s.code, "QUALITY_GATE")

    def test_bicim_hatasi_render_failed(self):
        self.assertEqual(self.kontrol(sahte_probe(vcodec="hevc")).code, "RENDER_FAILED")
        self.assertEqual(self.kontrol(sahte_probe(pix="yuv444p")).code, "RENDER_FAILED")
        self.assertEqual(self.kontrol(sahte_probe(acodec="mp3")).code, "RENDER_FAILED")

    def test_ses_akisi_yok(self):
        probe = sahte_probe()
        probe["streams"] = probe["streams"][:1]
        self.assertEqual(self.kontrol(probe).code, "RENDER_FAILED")

    def test_kucuk_dosya(self):
        s = self.kontrol(sahte_probe(), dosya_boyutu=10)
        self.assertEqual(s.code, "RENDER_FAILED")

    def test_yuz_orani(self):
        s = self.kontrol(sahte_probe(), face_frames_ratio=0.98)
        self.assertEqual(s.code, "FACE_NOT_FOUND")
        self.assertTrue(self.kontrol(sahte_probe(), face_frames_ratio=1.0).ok)

    def test_probe_hatasi(self):
        def patlak(_p):
            raise RuntimeError("ffprobe yok")

        s = qg.kontrol(Path("/tmp/yok.mp4"), 10.0, 1080, 1080, probe_fn=patlak, dosya_boyutu=1_000_000)
        self.assertEqual(s.code, "RENDER_FAILED")

    def test_kesir_ve_bilgi(self):
        self.assertEqual(qg._kesir("30000/1001"), 30000 / 1001)
        self.assertIsNone(qg._kesir("0/0"))
        v = qg.video_bilgisi(sahte_probe())
        self.assertEqual(v["fps"], 25.0)
        self.assertEqual(qg.ses_bilgisi(sahte_probe())["sample_rate"], 48000)
        self.assertIsNone(qg.video_bilgisi({"streams": []}))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg yok")
class GercekDosya(unittest.TestCase):
    """ffmpeg lavfi ile üretilen 4 sn test klibi: uyumlu ses geçer, 400 ms
    kısa ses QUALITY_GATE."""

    def setUp(self):
        self.dizin = Path(tempfile.mkdtemp())
        self.mp4 = self.dizin / "out.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=4",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-shortest",
                str(self.mp4),
            ],
            check=True,
        )

    def tearDown(self):
        shutil.rmtree(self.dizin, ignore_errors=True)

    def test_gercek_mp4(self):
        boyut = self.mp4.stat().st_size
        gecer = qg.kontrol(self.mp4, audio_seconds=4.0, source_width=320, source_height=240, fps=25, dosya_boyutu=max(boyut, qg.MIN_DOSYA_BAYT))
        self.assertTrue(gecer.ok, gecer.error)
        self.assertLessEqual(abs(gecer.metrics["frame_delta"] or 0), 2)
        sapan = qg.kontrol(self.mp4, audio_seconds=3.6, source_width=320, source_height=240, fps=25, dosya_boyutu=max(boyut, qg.MIN_DOSYA_BAYT))
        self.assertEqual(sapan.code, "QUALITY_GATE")


if __name__ == "__main__":
    unittest.main()
