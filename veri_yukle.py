import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

print("Excel dosyası okunuyor...")
df = pd.read_excel("startups.xlsx", sheet_name="main")

# Sadece işimize yarayan sütunları alalım
df = df[["Startups Name", "description", "industry"]]

# Açıklaması boş olan satırları at (RAG için anlamsız olurdu)
df = df.dropna(subset=["description"])

# Çok kısa açıklamaları da at (örn. tek kelimelik, anlamsız girişler)
df = df[df["description"].str.len() > 30]

# 1500 kayıtla sınırlayalım (fazlası gereksiz depolama/süre kaybı)
df = df.head(1500)

print(f"Toplam {len(df)} kayıt işlenecek.")

# Embedding modelini yükle (ilk çalıştırmada indirilecek, ~80-90 MB)
print("Embedding modeli yükleniyor (ilk seferde indirilecek)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB istemcisini başlat (dosya tabanlı, kalıcı depo)
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="startup_ornekleri")

print("Embedding'ler oluşturuluyor ve ChromaDB'ye kaydediliyor...")

batch_size = 50
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i + batch_size]

    descriptions = batch["description"].tolist()
    names = batch["Startups Name"].tolist()
    industries = batch["industry"].fillna("Belirtilmemiş").tolist()
    ids = [f"startup_{i + j}" for j in range(len(batch))]

    embeddings = model.encode(descriptions).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=descriptions,
        metadatas=[
            {"name": n, "industry": ind}
            for n, ind in zip(names, industries)
        ],
    )

    print(f"  {i + len(batch)} / {len(df)} kayıt işlendi...")

print("Tamamlandı! Veri chroma_data/ klasörüne kaydedildi.")
print(f"Toplam kayıt sayısı: {collection.count()}")