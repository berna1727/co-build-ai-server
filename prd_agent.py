"""
prd_agent.py
============================================================================
Co-Build AI — PRD Üretim Ajanı (LangGraph Self-Reflection Döngüsü)
============================================================================

Ham fikirden PRD üretimini, tek seferlik bir LLM çağrısı yerine bir
LangGraph durum makinesiyle yapar:

    prd_uret --> elestir --(onaylandı ya da 2. deneme)--> eslestir --> END
        ^                        |
        +-- (revize gerekli) ----+

- Katı döngü sınırı: en fazla 2 üretim denemesi (ilk üretim + en fazla 1
  düzeltme). `iteration >= 2` olduğunda eleştiri sonucu ne olursa olsun
  akış devam eder — ajan mükemmeliyetçi davranıp sonsuz döngüye giremez.
- Veri gizliliği odaklı: dış API (OpenAI/Anthropic vb.) YOK. Model, kendi
  kiraladığımız GPU'da (RunPod RTX 4090) vLLM'in sunduğu OpenAI-uyumlu
  yerel API üzerinden çalışır (bkz. VLLM_BASE_URL/VLLM_MODEL_ADI).
- Asenkron: tüm LLM çağrıları `ainvoke` ile yapılır, event loop'u bloklamaz.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from functools import lru_cache
from typing import Optional, TypedDict

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from matchmaking_engine import AramaHatasi, hibrit_eslestirme_yap

logger = logging.getLogger("cobuild.prd_agent")

# .env üzerinden override edilebilir. vLLM sunucusu genelde 8000 portunda,
# OpenAI-uyumlu path'i "/v1" ile açılır (bkz. README/aşağıdaki başlatma
# komutu). RunPod'da SSH tüneliyle localhost'a yönlendiriyorsanız varsayılan
# değer yeterlidir; RunPod'un HTTP proxy'sini kullanıyorsanız tam URL'i girin.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
# vLLM'e --api-key ile bir anahtar VERMEDİYSENİZ bu alan kontrol edilmez,
# herhangi bir değer (varsayılan "EMPTY") kabul edilir.
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
# vLLM sunucusunu başlatırken kullandığınız --model ile BİREBİR aynı olmalı
# (örn. Qwen/Qwen2.5-32B-Instruct-AWQ — 24GB VRAM'e sığması için 4-bit AWQ
# quantized sürüm şart, tam hassasiyetli 32B ~64GB VRAM ister).
VLLM_MODEL_ADI = os.environ.get("VLLM_MODEL_ADI", "Qwen/Qwen2.5-32B-Instruct-AWQ")
MAX_ITERATIONS = 2  # ilk üretim + en fazla 1 düzeltme

llm = ChatOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, model=VLLM_MODEL_ADI, max_tokens=1024)
# Öz-eleştiri yanıtı çok kısa olmalı (ONAYLANDI ya da birkaç maddelik liste);
# ayrı, düşük max_tokens'lı bir istemci kullanıyoruz.
elestiri_llm = ChatOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, model=VLLM_MODEL_ADI, max_tokens=300)


PRD_PROMPT = PromptTemplate(
    input_variables=["title", "raw_idea", "benzer_ornekler", "ek_baglam", "revize_notu"],
    template="""Sen deneyimli bir teknik ürün yöneticisisin. Aşağıdaki proje fikrini analiz et.

Proje Başlığı: {title}
Ham Fikir: {raw_idea}
{ek_baglam}
Gerçek, var olan benzer projeler (referans amaçlı):
{benzer_ornekler}
{revize_notu}
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

ÇOK ÖNEMLİ: Yukarıdaki TÜM bölümleri (Ürün Özeti, Hedef Kullanıcı, Temel Özellikler, Teknik Gereksinimler, Benzer Örnekler ve Farklılaşma, Tahmini Altyapı Maliyeti Kategorisi, Beceri Etiketleri) EKSİKSİZ doldur. Hiçbir bölümü atlama veya kısa geçme. Her bölüm en az 2-3 cümle veya 3-4 madde içermeli.

DİL KURALI (ÇOK ÖNEMLİ): PRD'nin TAMAMINI SADECE Türkçe yaz. Çince, İngilizce veya başka HİÇBİR dilden tek bir kelime, karakter veya cümle bile KULLANMA — cevabının tamamı baştan sona yalnızca Türkçe olmalı.""",
)

ELESTIRI_PROMPT = PromptTemplate(
    input_variables=["prd_metni"],
    template="""Aşağıdaki PRD taslağını, kalite kontrolcüsü gibi incele.

PRD Taslağı:
---
{prd_metni}
---

Şunları kontrol et:
1. Şu 7 bölümün HEPSİ eksiksiz bulunmalı: "Ürün Özeti", "Hedef Kullanıcı",
"Temel Özellikler" (en az 4 madde), "Teknik Gereksinimler", "Benzer Örnekler ve
Farklılaşma", "Tahmini Altyapı Maliyeti Kategorisi", "Beceri Etiketleri"
(virgülle ayrılmış, en fazla 5 teknik beceri; BOŞ OLAMAZ).
2. Metnin TAMAMI SADECE Türkçe olmalı — Çince, İngilizce veya başka bir dilden
TEK BİR kelime/karakter bile geçmemeli. Böyle bir karışım varsa bu ciddi bir hatadır.

Eğer yukarıdakilerin HEPSİ (7 bölüm eksiksiz VE metin tamamen Türkçe) doğruysa
SADECE şu tek kelimeyi yaz: ONAYLANDI
Aksi halde "REVIZE:" ile başlayıp neyin eksik/hatalı olduğunu (dil karışımı dahil)
kısaca (1-3 madde) listele. Başka hiçbir şey ekleme.""",
)


class PRDAgentState(TypedDict):
    title: str
    raw_idea: str
    benzer_ornekler: str
    budget_type: Optional[str]
    sektor: Optional[str]
    prd_metni: str
    skills_list: list[str]
    critique_feedback: Optional[str]
    iteration: int
    onaylandi: bool
    eslesme_sonuclari: list[dict]


def _ek_baglam_metni(state: PRDAgentState) -> str:
    satirlar = []
    if state.get("budget_type"):
        satirlar.append(f"Bütçe Türü: {state['budget_type']}")
    if state.get("sektor"):
        satirlar.append(f"Sektör: {state['sektor']}")
    return ("\n".join(satirlar) + "\n") if satirlar else ""



# Model başlığı tam olarak "Beceri Etiketleri" yazmayabiliyor (örn. sadece
# "Beceriler", "**Beceriler**" gibi paraphrase edebiliyor) — bu yüzden
# "Becer..." ile başlayan, kendi satırını tek başına oluşturan herhangi bir
# başlığı yakalayan esnek bir regex kullanıyoruz.
_SKILLS_BASLIK_RE = re.compile(
    r"(?im)^\s*#{0,3}\s*\*{0,2}\s*Becer[a-zçğıöşü]*(?:\s+Etiketleri)?\s*\*{0,2}\s*:?\s*$"
)
_BULLET_ONEKI_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
MAKS_SKILLS = 8


def _skills_ve_prd_ayikla(result: str) -> tuple[str, list[str]]:
    """
    LLM çıktısını PRD gövdesi ve beceri listesine ayırır. Model bazen
    "React Native, Node.js" gibi tek satırda virgüllü, bazen de "- Java\n-
    Spring Boot" gibi madde işaretli bir liste üretiyor — ikisini de
    destekliyoruz.
    """
    eslesme = _SKILLS_BASLIK_RE.search(result)
    if not eslesme:
        return result.strip(), []

    prd_text = result[: eslesme.start()].strip()
    kalan = result[eslesme.end():].strip()

    # Model başlığı bazen bir de gövdede tekrar eder (örn. "Beceriler: \n...").
    kalan = re.sub(
        r"(?im)^\s*Becer[a-zçğıöşü]*(?:\s+Etiketleri)?\s*:?\s*\n?", "", kalan, count=1
    ).lstrip(":\n ").strip()

    skills_list: list[str] = []
    for satir in kalan.splitlines():
        satir = satir.strip()
        if not satir:
            if skills_list:
                break  # liste bitti, boş satırdan sonrası başka bir bölüm olabilir
            continue
        if satir.startswith("#") or satir.startswith("**"):
            break  # yeni bir başlığa geçildi

        satir = _BULLET_ONEKI_RE.sub("", satir).strip()
        if "," in satir:
            skills_list.extend(s.strip() for s in satir.split(",") if s.strip())
        elif satir:
            skills_list.append(satir)

    # Sırayı koruyarak tekrarları temizle, makul bir üst sınır koy.
    skills_list = list(dict.fromkeys(skills_list))[:MAKS_SKILLS]
    return prd_text, skills_list


async def _prd_uret_node(state: PRDAgentState) -> dict:
    iteration = state["iteration"] + 1

    revize_notu = ""
    if state.get("critique_feedback"):
        revize_notu = (
            f"\nÖNCEKİ TASLAK EKSİKTİ, şu noktaları mutlaka düzelt ve PRD'nin "
            f"TAMAMINI yeniden, eksiksiz olarak yaz:\n{state['critique_feedback']}\n"
        )

    prompt = PRD_PROMPT.format(
        title=state["title"],
        raw_idea=state["raw_idea"],
        benzer_ornekler=state["benzer_ornekler"],
        ek_baglam=_ek_baglam_metni(state),
        revize_notu=revize_notu,
    )

    yanit = await llm.ainvoke(prompt)
    prd_metni, skills_list = _skills_ve_prd_ayikla(yanit.content)

    logger.info("PRD üretildi (deneme %d), beceri sayısı: %d", iteration, len(skills_list))

    return {"prd_metni": prd_metni, "skills_list": skills_list, "iteration": iteration}


# Çince/Japonca/Korece/Kiril gibi Türkçe'de asla geçmeyen alfabelerin Unicode
# aralıkları. Qwen gibi çok dilli modeller nadiren üretimin ortasında bu
# alfabelere kayabiliyor — bunu LLM'in kendi eleştirisine güvenmeden, kesin
# bir regex ile de tespit ediyoruz (LLM bu karışımı fark etmeyebilir).
_YABANCI_ALFABE_RE = re.compile(
    r"[一-鿿぀-ヿ가-힯Ѐ-ӿ]"
)


def _yabanci_alfabe_karismis_mi(metin: str) -> bool:
    return bool(_YABANCI_ALFABE_RE.search(metin))


async def _elestir_node(state: PRDAgentState) -> dict:
    if _yabanci_alfabe_karismis_mi(state["prd_metni"]):
        logger.warning("PRD'ye yabancı alfabe (Çince/Kiril vb.) karışmış (deneme %d), otomatik revize tetikleniyor.", state["iteration"])
        return {
            "onaylandi": False,
            "critique_feedback": "Metne Çince/Japonca/Korece/Kiril gibi Türkçe olmayan bir alfabe karışmış. PRD'nin TAMAMINI, tek bir yabancı karakter bile olmadan, baştan sona SADECE Türkçe olarak yeniden yaz.",
        }

    prompt = ELESTIRI_PROMPT.format(prd_metni=state["prd_metni"])
    yanit = await elestiri_llm.ainvoke(prompt)
    icerik = yanit.content.strip()

    onaylandi = icerik.upper().startswith("ONAYLANDI")
    logger.info("Öz-eleştiri sonucu (deneme %d): %s", state["iteration"], "ONAYLANDI" if onaylandi else "REVIZE")

    return {
        "onaylandi": onaylandi,
        "critique_feedback": None if onaylandi else icerik,
    }


def _karar_ver(state: PRDAgentState) -> str:
    if state["onaylandi"] or state["iteration"] >= MAX_ITERATIONS:
        return "eslestir"
    return "prd_uret"


async def _eslestir_node(state: PRDAgentState) -> dict:
    filtre_metadata = {}
    if state.get("budget_type"):
        filtre_metadata["budget_type"] = state["budget_type"]
    if state.get("sektor"):
        filtre_metadata["sektor"] = state["sektor"]

    try:
        sonuclar = await asyncio.to_thread(
            hibrit_eslestirme_yap,
            prd_metni=state["prd_metni"],
            gerekli_diller=state["skills_list"],
            top_k=5,
            filtre_metadata=filtre_metadata or None,
        )
        eslesme_sonuclari = [
            {
                "developer_id": s.developer_id,
                "ad_soyad": s.ad_soyad,
                "skills": s.skills,
                "bio": s.bio,
                "benzerlik_skoru": s.benzerlik_skoru,
                "hibrit_skor": s.hibrit_skor,
                "uyum_skoru": s.uyum_skoru,
            }
            for s in sonuclar
        ]
    except AramaHatasi as exc:
        logger.warning("PRD kaydedildi ama eşleştirme başarısız oldu: %s", exc)
        eslesme_sonuclari = []

    return {"eslesme_sonuclari": eslesme_sonuclari}


@lru_cache(maxsize=1)
def _graph_olustur():
    grafik = StateGraph(PRDAgentState)
    grafik.add_node("prd_uret", _prd_uret_node)
    grafik.add_node("elestir", _elestir_node)
    grafik.add_node("eslestir", _eslestir_node)

    grafik.set_entry_point("prd_uret")
    grafik.add_edge("prd_uret", "elestir")
    grafik.add_conditional_edges("elestir", _karar_ver, {"prd_uret": "prd_uret", "eslestir": "eslestir"})
    grafik.add_edge("eslestir", END)

    return grafik.compile()


async def prd_uret_ve_eslestir(
    title: str,
    raw_idea: str,
    benzer_ornekler: str,
    budget_type: Optional[str] = None,
    sektor: Optional[str] = None,
) -> dict:
    """
    PRD'yi self-reflection döngüsüyle üretir (en fazla 2 deneme) ve ardından
    hibrit eşleştirme motoruyla önerilen geliştiricileri döner.

    Dönüş: main.py'deki job_store["done"] şekliyle uyumlu bir sözlük:
        {"prd": str, "skills": list[str], "onerilen_gelistiriciler": list[dict],
         "iterasyon_sayisi": int}
    """
    baslangic_durumu: PRDAgentState = {
        "title": title,
        "raw_idea": raw_idea,
        "benzer_ornekler": benzer_ornekler,
        "budget_type": budget_type,
        "sektor": sektor,
        "prd_metni": "",
        "skills_list": [],
        "critique_feedback": None,
        "iteration": 0,
        "onaylandi": False,
        "eslesme_sonuclari": [],
    }

    graph = _graph_olustur()
    # State sayacı döngüyü zaten sınırlıyor; recursion_limit ikinci bir
    # güvenlik ağı (kullanıcının "kesin bir sınır" şartı için).
    sonuc = await graph.ainvoke(baslangic_durumu, config={"recursion_limit": 8})

    return {
        "prd": sonuc["prd_metni"],
        "skills": sonuc["skills_list"],
        "onerilen_gelistiriciler": sonuc["eslesme_sonuclari"],
        "iterasyon_sayisi": sonuc["iteration"],
    }
