from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = OllamaLLM(model="llama3.1:8b", num_predict=2048)

# RAG için: embedding modeli ve ChromaDB bağlantısı
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_data")
startup_collection = chroma_client.get_or_create_collection(name="startup_ornekleri")


def benzer_ornekleri_bul(raw_idea: str, n_results: int = 3) -> str:
    """Kullanıcının fikrine en yakın gerçek proje örneklerini ChromaDB'de arar."""
    query_embedding = embedding_model.encode([raw_idea]).tolist()
    results = startup_collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    if not results["documents"] or not results["documents"][0]:
        return "Bulunamadı."

    ornekler = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        isim = meta.get("name", "Bilinmeyen")
        sektor = meta.get("industry", "Belirtilmemiş")
        kisa_aciklama = doc[:200] + ("..." if len(doc) > 200 else "")
        ornekler.append(f"- {isim} ({sektor}): {kisa_aciklama}")

    return "\n".join(ornekler)


class FikirRequest(BaseModel):
    project_id: str
    title: str
    raw_idea: str


prompt_template = PromptTemplate(
    input_variables=["title", "raw_idea"],
    template="""Sen deneyimli bir teknik ürün yöneticisisin. Aşağıdaki proje fikrini analiz et.

Proje Başlığı: {title}
Ham Fikir: {raw_idea}

Gerçek, var olan benzer projeler (referans amaçlı):
{benzer_ornekler}

Şu formatta, TAM OLARAK bu sıralamayla cevap ver:

## Ürün Özeti
(2-3 cümle ile projenin ne olduğunu özetle)

## Hedef Kullanıcı
(kim kullanacak)

## Temel Özellikler
(madde madde, en az 4 özellik)

## Teknik Gereksinimler
(bu proje için gereken beceri/teknoloji alanlarını listele)

## Benzer Örnekler ve Farklılaşma
(Yukarıda verilen "gerçek, var olan benzer projeler" listesini kullan. Bunlardan en alakalı 1-2 tanesini belirt ve bu projenin onlardan nasıl farklılaşabileceğine dair somut öneriler sun. Eğer liste "Bulunamadı" diyorsa, "Veri tabanımızda doğrudan bir örnek bulunamadı, bu potansiyel bir avantaj olabilir" de.)

## Tahmini Altyapı Maliyeti Kategorisi
(Bu projenin altyapı ihtiyacını Düşük / Orta / Yüksek olarak kategorize et ve kısa bir gerekçe ver. Sonuna şu notu ekle: "Not: Bu, yapay zeka tarafından üretilen kaba bir tahmindir, gerçek maliyetler için bir teknik danışmana başvurulması önerilir.")

## Beceri Etiketleri
(EN FAZLA 5 tane, bu projeye GERÇEKTEN özgü teknik beceri/teknoloji. Alternatifleri aynı anda listeleme, en olası tek birini seç. SADECE virgülle ayrılmış liste halinde yaz, başka hiçbir şey ekleme. Örnek: React Native, Node.js, PostgreSQL)

ÇOK ÖNEMLİ: Yukarıdaki TÜM bölümleri (Ürün Özeti, Hedef Kullanıcı, Temel Özellikler, Teknik Gereksinimler, Benzer Örnekler ve Farklılaşma, Tahmini Altyapı Maliyeti Kategorisi, Beceri Etiketleri) EKSİKSİZ doldur. Hiçbir bölümü atlama veya kısa geçme. Her bölüm en az 2-3 cümle veya 3-4 madde içermeli. PRD'yi Türkçe yaz.""",
)

# İş durumlarını geçici olarak bellekte tutan basit bir depo
# (proje_id -> {"status": "processing" / "done" / "error", ...})
job_store: dict[str, dict] = {}


def generate_prd_task(project_id: str, title: str, raw_idea: str):
    try:
        benzer_ornekler = benzer_ornekleri_bul(raw_idea)
        prompt = prompt_template.format(
            title=title, raw_idea=raw_idea, benzer_ornekler=benzer_ornekler
        )
        result = llm.invoke(prompt)
        
        skills_marker = None
        for marker in ["## Beceri Etiketleri", "**Beceri Etiketleri**", "Beceri Etiketleri"]:
            if marker in result:
                skills_marker = marker
                break

        if skills_marker:
            parts = result.split(skills_marker)
            prd_text = parts[0].strip()
            skills_text = parts[1].strip()
            # Başındaki olası ":" veya "\n" gibi karakterleri temizle
            skills_text = skills_text.lstrip(":\n ").strip()
            skills_list = [s.strip() for s in skills_text.split(",") if s.strip()]
        else:
            prd_text = result
            skills_list = []

        job_store[project_id] = {
            "status": "done",
            "prd": prd_text,
            "skills": skills_list,
        }
    except Exception as e:
        job_store[project_id] = {"status": "error", "message": str(e)}


@app.get("/")
def read_root():
    return {"status": "AI servisi çalışıyor"}


@app.post("/prd-uret-baslat")
def prd_uret_baslat(request: FikirRequest, background_tasks: BackgroundTasks):
    job_store[request.project_id] = {"status": "processing"}
    background_tasks.add_task(
        generate_prd_task, request.project_id, request.title, request.raw_idea
    )
    return {"status": "started"}


@app.get("/prd-durum/{project_id}")
def prd_durum(project_id: str):
    return job_store.get(project_id, {"status": "not_found"})