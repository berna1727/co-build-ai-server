import os
from dotenv import load_dotenv
import json
import random
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.getenv["SUPABASE_SERVICE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# --- VERİYİ DOSYADAN OKU ---
with open("demo_veri.json", "r", encoding="utf-8") as f:
    data = json.load(f)

developers = data["developers"]
founders = data["founders"]
projects = data["projects"]

print(f"{len(developers)} yazılımcı, {len(founders)} fikir sahibi, {len(projects)} proje yüklenecek.\n")

# --- YAZILIMCI HESAPLARI OLUŞTUR ---
print("Yazılımcı hesapları oluşturuluyor...")
developer_ids = []
for i, dev in enumerate(developers):
    email = f"demo.yazilimci{i}@cobuildai-demo.com"
    try:
        auth_response = supabase.auth.admin.create_user({
            "email": email,
            "password": "DemoSifre123!",
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
        print(f"  ✗ Hata ({dev['full_name']}): {e}")

# --- FİKİR SAHİBİ HESAPLARI OLUŞTUR ---
print("\nFikir sahibi hesapları oluşturuluyor...")
founder_ids = []
for i, founder in enumerate(founders):
    email = f"demo.girisimci{i}@cobuildai-demo.com"
    try:
        auth_response = supabase.auth.admin.create_user({
            "email": email,
            "password": "DemoSifre123!",
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
        print(f"  ✗ Hata ({founder['full_name']}): {e}")

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
        print(f"  ✗ Hata ({proj['title']}): {e}")

print("\nTamamlandı!")
print(f"Toplam: {len(developer_ids)} yazılımcı, {len(founder_ids)} fikir sahibi, {len(projects)} proje işlendi.")