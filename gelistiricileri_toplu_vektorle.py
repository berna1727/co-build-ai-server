"""
gelistiricileri_toplu_vektorle.py
============================================================================
Co-Build AI — Mevcut yazılımcı profillerini toplu (geriye dönük) vektörler
============================================================================

Ne işe yarar?
-------------
`developer_embeddings` tablosu kurulduğunda içi boştur (yeni tablo). Bu
script, `profiles` tablosundaki TÜM yazılımcıları (`user_type = 'developer'`)
tek tek `matchmaking_engine.gelistirici_profilini_vektorle` ile vektörleyip
`developer_embeddings` tablosuna yazar.

Bu script sadece demo verisi için değil — gerçek kayıt olan yazılımcılar da
`profiles` tablosuna düştüğü için, ileride toplu bir yeniden-vektörleme
gerekirse (örn. embedding modelini değiştirirseniz) aynı script tekrar
çalıştırılabilir. Zaten vektörlenmiş bir geliştirici tekrar işlenirse
`gelistirici_profilini_vektorle` upsert yaptığı için sorun çıkmaz, sadece
günceller.

Neden isimden değil `profiles` tablosundan okuyoruz?
-----------------------------------------------------
`demo_kullanici_uret.py` çalışırken oluşturduğu Supabase Auth UUID'lerini
hiçbir yere kaydetmiyor (sadece ekrana yazdırıyor) — yani "Ahmet Yılmaz'ın
developer_id'si şudur" bilgisi elimizde yok. Ama zaten `profiles` tablosuna
`id` (o UUID), `full_name`, `bio`, `skills` olarak yazılmış durumda. En
güvenilir ve genel yöntem, ismi değil doğrudan `profiles` tablosunu
kaynak almak.

Çalıştırma:
    python gelistiricileri_toplu_vektorle.py

Gereksinim:
    - Bu dosya `matchmaking_engine.py` ile AYNI klasörde olmalı.
    - .env dosyasında SUPABASE_URL ve SUPABASE_SERVICE_KEY tanımlı olmalı
      (demo_kullanici_uret.py için zaten kullandığınız değerlerin aynısı).
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from supabase import create_client

from matchmaking_engine import (
    EslestirmeMotoruHatasi,
    GelistiriciProfili,
    gelistirici_profilini_vektorle,
)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    sys.exit(
        "HATA: SUPABASE_URL / SUPABASE_SERVICE_KEY .env dosyasında tanımlı değil.\n"
        "Çözüm: '.env.example' dosyasını referans alarak '.env' dosyanızı kontrol edin."
    )

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def yazilimcilari_getir() -> list[dict]:
    """profiles tablosundan sadece yazılımcı (developer) tipindeki kayıtları çeker."""
    yanit = (
        supabase.table("profiles")
        .select("id, full_name, bio, skills")
        .eq("user_type", "developer")
        .execute()
    )
    return yanit.data or []


def main() -> None:
    yazilimcilar = yazilimcilari_getir()
    if not yazilimcilar:
        print("profiles tablosunda 'developer' tipinde kayıt bulunamadı — vektörlenecek bir şey yok.")
        return

    print(f"{len(yazilimcilar)} yazılımcı bulundu, vektörleniyor...\n")

    basarili, atlanan = 0, 0
    for satir in yazilimcilar:
        profil = GelistiriciProfili(
            developer_id=satir["id"],
            ad_soyad=satir.get("full_name") or "İsimsiz Yazılımcı",
            # Not: profiles.skills tek bir liste olarak tutuluyor (uzmanlık alanı /
            # bilinen dil ayrımı veritabanında yapılmıyor), bu yüzden hepsini
            # bildigi_diller alanına koyuyoruz — embedding metnine katkısı aynı.
            bildigi_diller=satir.get("skills") or [],
            bio=satir.get("bio") or "",
        )
        try:
            gelistirici_profilini_vektorle(profil)
            print(f"  ✓ {profil.ad_soyad} vektörlendi")
            basarili += 1
        except EslestirmeMotoruHatasi as exc:
            # Örn: hem skills hem bio boşsa embedding'e çevrilecek bir şey kalmaz.
            print(f"  ✗ {profil.ad_soyad} atlandı: {exc}")
            atlanan += 1

    print(f"\nTamamlandı. Vektörlenen: {basarili}, Atlanan: {atlanan}, Toplam: {len(yazilimcilar)}")


if __name__ == "__main__":
    main()
