# -*- coding: utf-8 -*-
"""K9: handler girdi doğrulama ve güvenlik çiti (CPU, ağ yok, model yok)."""
import json
import os
import sys
import unittest
from pathlib import Path

os.environ["LIPSYNC_TEST_MODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import handler  # noqa: E402

AYAR = handler.ayarlari_oku(
    {
        "ALLOWED_INPUT_HOSTS": "media.example.com",
        "ALLOWED_UPLOAD_HOST_SUFFIX": ".r2.cloudflarestorage.com",
        "MAX_AUDIO_SECONDS": "70",
        "CACHE_DIR": "/tmp/lipsync-test-cache",
    }
)
KEY = "sales/p1/clips/s1-abcdef123456.mp4"
PUT = f"https://acct.r2.cloudflarestorage.com/bucket/{KEY}?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=1800&X-Amz-Signature=deadbeef"


def gecerli_render(**ek):
    g = {
        "mode": "render",
        "audio_url": "https://media.example.com/sales/p1/audio/s1.mp3",
        "source_url": "https://media.example.com/avatars/deniz/speaking-source.mp4",
        "output_put_url": PUT,
        "output_key": KEY,
        "fps": 25,
        "options": {"batch_size": 8, "parsing_mode": "jaw"},
    }
    g.update(ek)
    return g


class GirdiDogrulama(unittest.TestCase):
    def kod(self, girdi):
        with self.assertRaises(handler.IsHatasi) as ctx:
            handler.girdiyi_dogrula(girdi, AYAR)
        return ctx.exception

    def test_gecerli_render_normalize_edilir(self):
        s = handler.girdiyi_dogrula(gecerli_render(prep_url="https://media.example.com/avatars/deniz/prep-a.tar", source_hash="sha256:" + "a" * 64, bilinmeyen="x"), AYAR)
        self.assertEqual(s["mode"], "render")
        self.assertEqual(s["output_key"], KEY)
        self.assertEqual(s["options"]["extra_margin"], 10)
        self.assertNotIn("bilinmeyen", s)
        self.assertEqual(s["source_hash"], "sha256:" + "a" * 64)

    def test_allowlist_disi_host(self):
        h = self.kod(gecerli_render(audio_url="https://evil.example/s1.mp3"))
        self.assertEqual(h.code, "INPUT_REJECTED")
        self.assertFalse(h.retryable)

    def test_http_reddedilir(self):
        h = self.kod(gecerli_render(source_url="http://media.example.com/avatars/deniz/speaking-source.mp4"))
        self.assertEqual(h.code, "INPUT_REJECTED")
        self.assertIn("https", h.message)

    def test_output_key_nokta_nokta(self):
        kotu = "sales/../avatars/x.mp4"
        h = self.kod(gecerli_render(output_key=kotu, output_put_url=f"https://acct.r2.cloudflarestorage.com/bucket/{kotu}"))
        self.assertEqual(h.code, "INPUT_REJECTED")

    def test_output_key_yanlis_onek(self):
        kotu = "users/p1/x.mp4"
        h = self.kod(gecerli_render(output_key=kotu, output_put_url=f"https://acct.r2.cloudflarestorage.com/bucket/{kotu}"))
        self.assertEqual(h.code, "INPUT_REJECTED")

    def test_yanlis_upload_host(self):
        h = self.kod(gecerli_render(output_put_url=f"https://evil.example/bucket/{KEY}?X-Amz-Signature=x"))
        self.assertEqual(h.code, "INPUT_REJECTED")

    def test_upload_yolu_key_ile_eslesmeli(self):
        h = self.kod(gecerli_render(output_put_url="https://acct.r2.cloudflarestorage.com/bucket/sales/p1/clips/baska.mp4?X-Amz-Signature=x"))
        self.assertEqual(h.code, "INPUT_REJECTED")

    def test_kimlik_bilgili_url(self):
        h = self.kod(gecerli_render(audio_url="https://user:pw@media.example.com/s1.mp3"))
        self.assertEqual(h.code, "INPUT_REJECTED")

    def test_bozuk_source_hash(self):
        h = self.kod(gecerli_render(source_hash="md5:abc"))
        self.assertEqual(h.code, "INPUT_REJECTED")

    def test_secenek_araliklari(self):
        h = self.kod(gecerli_render(options={"batch_size": 999}))
        self.assertEqual(h.code, "INPUT_REJECTED")
        h = self.kod(gecerli_render(options={"parsing_mode": "full"}))
        self.assertEqual(h.code, "INPUT_REJECTED")
        h = self.kod(gecerli_render(fps=0))
        self.assertEqual(h.code, "INPUT_REJECTED")

    def test_taninmayan_mod(self):
        h = self.kod({"mode": "train"})
        self.assertEqual(h.code, "INPUT_REJECTED")

    def test_probe(self):
        self.assertEqual(handler.girdiyi_dogrula({"mode": "probe"}, AYAR), {"mode": "probe"})

    def test_prepare_gecerli_ve_norm(self):
        key = "avatars/deniz/prep-abc.tar"
        s = handler.girdiyi_dogrula(
            {
                "mode": "prepare",
                "source_url": "https://media.example.com/avatars/deniz/speaking-source.mp4",
                "prep_put_url": f"https://acct.r2.cloudflarestorage.com/bucket/{key}?X-Amz-Signature=x",
                "prep_key": key,
                "norms": {"min_face_px": 200},
            },
            AYAR,
        )
        self.assertEqual(s["norms"]["min_face_px"], 200)
        self.assertEqual(s["norms"]["max_face_px"], 450)
        # prep_key sales/ altında olamaz
        h = self.kod(
            {
                "mode": "prepare",
                "source_url": "https://media.example.com/a.mp4",
                "prep_put_url": "https://acct.r2.cloudflarestorage.com/bucket/sales/x.tar",
                "prep_key": "sales/x.tar",
            }
        )
        self.assertEqual(h.code, "INPUT_REJECTED")


class HandlerCevabi(unittest.TestCase):
    def test_girdi_reddi_yapilandirilmis_cevap(self):
        cevap = handler.handler({"id": "j1", "input": gecerli_render(audio_url="http://x/s.mp3")}, AYAR)
        self.assertEqual(cevap["ok"], False)
        self.assertEqual(cevap["code"], "INPUT_REJECTED")
        self.assertEqual(cevap["retryable"], False)
        # Hata metni imzalı URL içermez.
        self.assertNotIn("X-Amz-Signature=deadbeef", json.dumps(cevap))

    def test_probe_test_modunda(self):
        cevap = handler.handler({"id": "j2", "input": {"mode": "probe"}}, AYAR)
        self.assertTrue(cevap["ok"])
        self.assertTrue(cevap["test_mode"])
        self.assertIn("musetalk_commit", cevap)

    def test_retryable_kodlar(self):
        self.assertTrue(handler.IsHatasi("DOWNLOAD_FAILED", "x").retryable)
        self.assertTrue(handler.IsHatasi("UPLOAD_FAILED", "x").retryable)
        self.assertTrue(handler.IsHatasi("OOM", "x").retryable)
        for kod in ("INPUT_REJECTED", "FACE_NOT_FOUND", "PREP_FAILED", "RENDER_FAILED", "QUALITY_GATE"):
            self.assertFalse(handler.IsHatasi(kod, "x").retryable, kod)
        with self.assertRaises(ValueError):
            handler.IsHatasi("BILINMEYEN", "x")

    def test_url_maskele(self):
        self.assertEqual(handler.url_maskele(f"PUT {PUT} 403"), "PUT https://acct.r2.cloudflarestorage.com/bucket/sales/p1/clips/s1-abcdef123456.mp4?X-Amz-*** 403")

    def test_max_audio_env_okunur(self):
        a = handler.ayarlari_oku({"MAX_AUDIO_SECONDS": "12.5", "ALLOWED_INPUT_HOSTS": " A.example , b.example"})
        self.assertEqual(a.max_audio_seconds, 12.5)
        self.assertEqual(a.allowed_input_hosts, frozenset({"a.example", "b.example"}))


class Onbellek(unittest.TestCase):
    def test_lru_kirpma(self):
        import shutil
        import tempfile
        import time

        kok = tempfile.mkdtemp()
        try:
            ayar = handler.ayarlari_oku({"CACHE_DIR": kok, "CACHE_MAX_AVATARS": "2"})
            for i, h in enumerate(("sha256:" + str(i) * 64 for i in range(3))):
                d = handler.onbellek_yolu(ayar, h)
                d.mkdir(parents=True)
                (d / "meta.json").write_text("{}")
                (d / "latents.pt").write_bytes(b"x")
                (d / ".last_used").write_text(str(1000 + i))
            self.assertEqual(handler.onbellek_kirp(ayar), 1)
            kalan = handler.onbellek_avatarlari(ayar)
            self.assertEqual(len(kalan), 2)
            self.assertNotIn("0" * 64, kalan)
            self.assertTrue(handler.onbellekte_var_mi(ayar, "sha256:" + "1" * 64))
        finally:
            shutil.rmtree(kok, ignore_errors=True)

    def test_paket_yol_kacisi_reddedilir(self):
        import io
        import shutil
        import tarfile
        import tempfile

        kok = Path(tempfile.mkdtemp())
        try:
            tar_yolu = kok / "kotu.tar"
            with tarfile.open(tar_yolu, "w") as tar:
                veri = b"{}"
                bilgi = tarfile.TarInfo("../../etc/cron.d/x")
                bilgi.size = len(veri)
                tar.addfile(bilgi, io.BytesIO(veri))
            with self.assertRaises(handler.IsHatasi) as ctx:
                handler.paketi_ac(tar_yolu, kok / "acilan", "sha256:" + "a" * 64)
            self.assertEqual(ctx.exception.code, "PREP_FAILED")

            iyi = kok / "iyi.tar"
            with tarfile.open(iyi, "w") as tar:
                veri = json.dumps({"source_hash": "sha256:" + "b" * 64}).encode()
                bilgi = tarfile.TarInfo("meta.json")
                bilgi.size = len(veri)
                tar.addfile(bilgi, io.BytesIO(veri))
            with self.assertRaises(handler.IsHatasi):
                handler.paketi_ac(iyi, kok / "acilan2", "sha256:" + "a" * 64)  # hash uyuşmaz
            handler.paketi_ac(iyi, kok / "acilan3", "sha256:" + "b" * 64)
            self.assertTrue((kok / "acilan3" / "meta.json").is_file())
        finally:
            shutil.rmtree(kok, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
