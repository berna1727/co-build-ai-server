# Co-Build AI — AI Service
 
> This repository contains the AI component of the [Co-Build AI](https://github.com/esmacakar/co-build-ai) platform.
 
## About Co-Build AI
 
Co-Build AI is a marketplace platform that connects non-technical founders with developers. Founders describe their project ideas in their own words; the platform converts these ideas into professional technical specifications, closing the communication and trust gap between founders and developers. The platform consists of a Next.js web application and this standalone AI service.
 
## Role of This Service
 
This service processes raw project ideas submitted through the platform and produces:
 
- A structured Product Requirements Document (PRD)
- A limited set of project-specific technical skill tags
- A differentiation and market-positioning analysis backed by real startup data
All processing is performed using an open-source language model running on local hardware, with no dependency on third-party AI APIs. This architectural choice ensures that user data is never transmitted abroad, supporting compliance with Turkey's data protection law (KVKK) at the technical level.
 
## Architecture
 
```
Next.js Client
      │  POST /prd-uret-baslat
      ▼
FastAPI  ───────────────►  ChromaDB
      │                    (RAG: similar project search)
      ▼
LangChain  ─────────────►  Ollama · llama3.1:8b
      │
      ▼
In-memory job store
      ▲
      │  GET /prd-durum/{id}
Next.js Client  (periodic polling)
```
 
Since inference runs on CPU, a single generation job can take several minutes. To avoid blocking the client, requests are not handled synchronously; the job is started in the background and the client polls a separate endpoint for the result.
 
## Capabilities
 
**PRD Generation**
Converts a raw idea into a structured document covering the product summary, target audience, core features, technical requirements, and an estimated infrastructure cost category.
 
**Skill Tag Extraction**
Parses the model's output into a clean list of at most five technical skills genuinely relevant to the project.
 
**RAG-Backed Differentiation Analysis**
Performs a semantic search against a vector database of real startup descriptions and supplies the closest matches to the model as context, so its differentiation suggestions are grounded in real reference points rather than generated from memory alone.
 
## Tech Stack
 
| Layer | Technology |
|---|---|
| Web framework | FastAPI |
| Model runtime | Ollama (`llama3.1:8b`) |
| Orchestration | LangChain |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
 
## Setup
 
**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) installed with the `llama3.1:8b` model pulled.
 
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
 
Build the vector database (run once, on first setup):
 
```bash
python veri_yukle.py
```
 
Start the server:
 
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
 
The service runs at `http://localhost:8000`. Interactive API documentation is available at `http://localhost:8000/docs`.
 
## API Reference
 
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service health check |
| `POST` | `/prd-uret-baslat` | Starts PRD generation in the background, returns immediately |
| `GET` | `/prd-durum/{project_id}` | Returns the current status of a job and its result once complete |
 
```json
POST /prd-uret-baslat
{
  "project_id": "abc-123",
  "title": "Neighborhood Tool-Sharing App",
  "raw_idea": "An app for neighbors to share rarely-used tools with each other."
}
```
 
## Known Limitations
 
- Inference runs without GPU acceleration; generation time varies with hardware
- Job status is stored in memory only; in-progress jobs are lost if the server restarts
- Concurrent requests can increase processing time
## Data Source & License
 
This project is maintained in a private repository. The dataset used for the RAG component is sourced from [HackerNoon/where-startups-trend](https://huggingface.co/datasets/HackerNoon/where-startups-trend), licensed under MIT.
# Co-Build AI — AI Servisi
 
> Bu depo, [Co-Build AI](https://github.com/esmacakar/co-build-ai) platformunun yapay zeka bileşenidir.
 
## Co-Build AI Hakkında
 
Co-Build AI, teknik bilgisi olmayan fikir sahiplerini yazılımcılarla buluşturan bir pazar yeri platformudur. Fikir sahipleri projelerini kendi cümleleriyle anlatır; platform bu fikri profesyonel bir teknik şartnameye dönüştürerek fikir sahibi ile yazılımcı arasındaki iletişim ve güven açığını kapatır. Platform, Next.js tabanlı bir web uygulaması ile bu depoda yer alan bağımsız bir yapay zeka servisinden oluşur.
 
## Bu Servisin Görevi
 
Bu servis, platforma gelen ham proje fikirlerini işleyerek şunları üretir:
 
- Yapılandırılmış bir Ürün Gereksinimleri Dokümanı (PRD)
- Projeye özgü, sınırlı sayıda teknik beceri etiketi
- Gerçek girişim verileriyle desteklenen bir farklılaşma ve konumlandırma analizi
İşlemlerin tamamı, üçüncü parti bir yapay zeka API'sine bağımlı olmadan, yerel donanım üzerinde çalışan açık kaynaklı bir dil modeliyle gerçekleştirilir. Bu mimari tercih, kullanıcı verisinin hiçbir aşamada yurt dışına aktarılmamasını sağlayarak KVKK uyumluluğunu teknik düzeyde destekler.
 
## Mimari
 
```
Next.js İstemcisi
      │  POST /prd-uret-baslat
      ▼
FastAPI  ───────────────►  ChromaDB
      │                    (RAG: benzer proje araması)
      ▼
LangChain  ─────────────►  Ollama · llama3.1:8b
      │
      ▼
İş durumu deposu (bellek içi)
      ▲
      │  GET /prd-durum/{id}
Next.js İstemcisi  (periyodik sorgulama)
```
 
Modelin CPU üzerinde çalışması nedeniyle bir üretim işlemi birkaç dakika sürebilir. Bu yüzden istekler senkron olarak bekletilmez; işlem arka planda başlatılır ve istemci sonucu ayrı bir uç noktadan periyodik olarak sorgular.
 
## Yetenekler
 
**PRD Üretimi**
Ham fikri; ürün özeti, hedef kullanıcı tanımı, temel özellikler, teknik gereksinimler ve tahmini altyapı maliyeti kategorisinden oluşan yapılandırılmış bir dokümana dönüştürür.
 
**Beceri Etiketi Çıkarımı**
Projeye gerçekten özgü, en fazla beş teknik beceriyi model çıktısından ayrıştırarak yapılandırılmış bir listeye dönüştürür.
 
**RAG Destekli Farklılaşma Analizi**
Gönderilen fikri, gerçek girişim açıklamalarından oluşan bir vektör veritabanında anlamsal olarak arar; bulunan en yakın örnekleri modele bağlam olarak sunarak model çıktısının kendi ezberi yerine gerçek referanslara dayanmasını sağlar.
 
## Teknoloji Yığını
 
| Katman | Teknoloji |
|---|---|
| Web çatısı | FastAPI |
| Model çalıştırma | Ollama (`llama3.1:8b`) |
| Orkestrasyon | LangChain |
| Embedding | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vektör deposu | ChromaDB |
 
## Kurulum
 
**Ön koşullar:** Python 3.11 veya üzeri, [Ollama](https://ollama.com) kurulu ve `llama3.1:8b` modeli indirilmiş olmalıdır.
 
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
 
Vektör veritabanını oluşturmak için, yalnızca ilk kurulumda bir kez çalıştırılır:
 
```bash
python veri_yukle.py
```
 
Sunucuyu başlatmak için:
 
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
 
Servis `http://localhost:8000` adresinde çalışmaya başlar. İnteraktif API dokümantasyonu `http://localhost:8000/docs` altında sunulur.
 
## API Referansı
 
| Metod | Uç Nokta | Açıklama |
|---|---|---|
| `GET` | `/` | Servis durumu kontrolü |
| `POST` | `/prd-uret-baslat` | PRD üretimini arka planda başlatır, anında yanıt döner |
| `GET` | `/prd-durum/{project_id}` | Belirtilen işin güncel durumunu ve, tamamlandıysa, sonucunu döndürür |
 
```json
POST /prd-uret-baslat
{
  "project_id": "abc-123",
  "title": "Komşular Arası Eşya Paylaşım Uygulaması",
  "raw_idea": "İnsanların nadiren kullandığı aletleri komşularıyla paylaşabileceği bir uygulama."
}
```
 
## Bilinen Kısıtlar
 
- Model çıkarımı GPU hızlandırması olmadan yürütülür; üretim süresi donanıma bağlı olarak değişkenlik gösterir
- İş durumları yalnızca bellekte tutulur; sunucu yeniden başlatıldığında devam eden işlerin durumu kaybolur
- Eş zamanlı gelen çok sayıda istek, işlem sürelerini uzatabilir
## Veri Kaynağı ve Lisans
 
Bu proje özel (private) bir depoda tutulmaktadır. RAG bileşeninde kullanılan veri seti [HackerNoon/where-startups-trend](https://huggingface.co/datasets/HackerNoon/where-startups-trend) kaynağından alınmıştır ve MIT Lisansı ile paylaşılmaktadır.
 
