"""Supabase'e demo kullanici ve proje verisi yukler.

DIKKAT: Bu script SUPABASE_SERVICE_KEY kullanir. Bu anahtar Row Level
Security'yi bypass eder ve tam admin yetkisine sahiptir. Yalnizca gelistirme /
demo projeleri uzerinde calistirin.
"""

import json
import os
import random
import sys

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# --- ORTAM DEGISKENLERI ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
DEMO_PASSWORD = os.getenv("DEMO_USER_PASSWORD", "DemoSifre123!")

_eksik = [
    isim
    for isim, deger in (
        ("SUPABASE_URL", SUPABASE_URL),
        ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
    )
    if not deger
]
if _eksik:
    sys.exit(
        f"HATA: eksik ortam degiskeni: {', '.join(_eksik)}\n"
        "Cozum: '.env.example' dosyasini '.env' olarak kopyalayip degerleri doldurun."
    )

# Sablon degerin kazara kullanilmasini yakala. Kalibi burada literal olarak
# yazmiyoruz; aksi halde pre-commit sir taramasi bu dosyayi yanlis pozitif isaretler.
_SABLON_ISARETLERI = ("replace_me", "your-service-key", "buraya", "changeme")
if any(isaret in SUPABASE_SERVICE_KEY.lower() for isaret in _SABLON_ISARETLERI):
    sys.exit("HATA: SUPABASE_SERVICE_KEY hala sablon degerinde. .env dosyasini doldurun.")


# --- SIR MASKELEME ---
# Supabase istemcisinden gelen hata mesajlari istek baglamini icerebilir.
# Ekrana/log'a basilan hicbir metinde sir gorunmemesini garanti ediyoruz.
_GIZLI_DEGERLER = tuple(
    deger for deger in (SUPABASE_SERVICE_KEY, DEMO_PASSWORD) if deger and len(deger) >= 8
)


def maskele(metin: object) -> str:
    """Verilen metindeki tum sir degerlerini '***GIZLI***' ile degistirir."""
    sonuc = str(metin)
    for sir in _GIZLI_DEGERLER:
        sonuc = sonuc.replace(sir, "***GIZLI***")
    return sonuc


# --- GUVENLIK ONAYI ---
# Yanlislikla canli bir projeye yazmayi onlemek icin acik onay istenir.
def onay_al() -> None:
    if os.getenv("DEMO_SEED_ONAY", "").strip().lower() in ("1", "true", "evet", "yes"):
        return
    print("Bu islem asagidaki Supabase projesine ADMIN yetkisiyle yazacak:")
    print(f"  {SUPABASE_URL}")
    print("  (service key yuklendi, degeri gosterilmiyor)")
    try:
        cevap = input("Devam etmek icin 'EVET' yazin: ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nIptal edildi (onay alinamadi).")
    if cevap != "EVET":
        sys.exit("Iptal edildi.")


onay_al()

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# --- VERIYI DOSYADAN OKU ---
with open("demo_veri.json", "r", encoding="utf-8") as f:
    data = json.load(f)

developers = data["developers"]
founders = data["founders"]
projects = data["projects"]

print(f"\n{len(developers)} yazılımcı, {len(founders)} fikir sahibi, {len(projects)} proje yüklenecek.\n")

# --- YAZILIMCI HESAPLARI OLUŞTUR ---
print("Yazılımcı hesapları oluşturuluyor...")
developer_ids = []
for i, dev in enumerate(developers):
    email = f"demo.yazilimci{i}@cobuildai-demo.com"
    try:
        auth_response = supabase.auth.admin.create_user({
            "email": email,
            "password": DEMO_PASSWORD,
            "email_confirm": True,
        })
        user_id = auth_response.user.id
        developer_ids.append(user_id)

        supabase.table("profiles").insert({
            "id": user_id,
            "user_type": "developer",
            "full_name": dev["full_name"],
            "bio": dev["bio"],
            "skills": dev["skills"],
        }).execute()

        print(f"  ✓ {dev['full_name']} oluşturuldu")
    except Exception as e:
        print(f"  ✗ Hata ({dev['full_name']}): {maskele(e)}")

# --- FİKİR SAHİBİ HESAPLARI OLUŞTUR ---
print("\nFikir sahibi hesapları oluşturuluyor...")
founder_ids = []
for i, founder in enumerate(founders):
    email = f"demo.girisimci{i}@cobuildai-demo.com"
    try:
        auth_response = supabase.auth.admin.create_user({
            "email": email,
            "password": DEMO_PASSWORD,
            "email_confirm": True,
        })
        user_id = auth_response.user.id
        founder_ids.append(user_id)

        supabase.table("profiles").insert({
            "id": user_id,
            "user_type": "founder",
            "full_name": founder["full_name"],
        }).execute()

        print(f"  ✓ {founder['full_name']} oluşturuldu")
    except Exception as e:
        print(f"  ✗ Hata ({founder['full_name']}): {maskele(e)}")

# --- PROJELERİ, RASTGELE FİKİR SAHİPLERİNE BAĞLAYARAK EKLE ---
print("\nProjeler ekleniyor...")
for proj in projects:
    if not founder_ids:
        print("  ✗ Hiç fikir sahibi oluşturulamadığı için projeler eklenemiyor.")
        break
    random_founder = random.choice(founder_ids)
    try:
        supabase.table("projects").insert({
            "founder_id": random_founder,
            "title": proj["title"],
            "raw_idea": proj["raw_idea"],
            "required_skills": proj["required_skills"],
            "status": "published",
        }).execute()
        print(f"  ✓ {proj['title']} eklendi")
    except Exception as e:
        print(f"  ✗ Hata ({proj['title']}): {maskele(e)}")

print("\nTamamlandı!")
print(f"Toplam: {len(developer_ids)} yazılımcı, {len(founder_ids)} fikir sahibi, {len(projects)} proje işlendi.")
