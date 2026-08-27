"""
matchmaking_engine.py
============================================================================
Co-Build AI — Semantik Eşleştirme Motoru (Matchmaking Engine)
============================================================================

Ne yapar?
---------
1) `gelistirici_profilini_vektorle(...)`
   Bir yazılımcının profilini (uzmanlık alanları, bildiği diller, bio) alır,
   embedding'e çevirir ve Supabase (pgvector) tablosuna kaydeder/günceller.

2) `en_uygun_gelistiricileri_bul(prd_metni, top_k=5)`
   Üretilen PRD metnini embedding'e çevirir, veritabanında kosinüs
   benzerliğine göre en yakın top_k yazılımcıyı getirir (saf semantik arama).

3) `hibrit_eslestirme_yap(prd_metni, gerekli_diller=None, top_k=5)`
   Bonus: Saf semantik arama + BM25 anahtar kelime araması sonuçlarını
   "Reciprocal Rank Fusion" (RRF) ile birleştirir. Böylece PRD'de "React"
   veya "Python" gibi net bir teknoloji ismi geçtiğinde, o dile sahip
   yazılımcılar sırf embedding benzerliği zayıf kaldı diye elenmez.

Mimari kararlar (neden böyle yaptık?)
--------------------------------------
- Embedding modeli: `sentence-transformers/all-MiniLM-L6-v2` (yerel, ~90MB).
  Projenizin RAG sisteminde (girişim örnekleri için) zaten bu modeli
  kullanıyorsunuz — KVKK gerekçesiyle veri dış API'lere gönderilmiyor.
  Aynı modeli burada da kullanmak hem tutarlılık sağlar hem de ek bir
  dış servis bağımlılığı (OpenAI embedding API gibi) eklemez.

- Vektör veritabanı: Supabase / pgvector (Pinecone değil). Zaten Supabase
  Postgres kullanıyorsunuz; ayrı bir Pinecone hesabı/servisi açmak yerine
  mevcut veritabanınıza pgvector eklentisini eklemek daha az hareketli
  parça, daha az maliyet ve tek bir yerden yönetim demek. Pinecone'a
  geçmek isterseniz dosyanın en altındaki "PINECONE ALTERNATİFİ" notuna
  bakın — fonksiyon imzaları (vektorle / ara) aynı kalacak şekilde
  tasarlandı, sadece bu dosyadaki depolama/arama katmanını değiştirmeniz
  yeterli olur.

- Vektör araması için ham pgvector RPC çağrısı kullanıldı (LangChain'in
  genel `SupabaseVectorStore` sarmalayıcısı yerine). Sebep: bizim
  tablomuzda `skills` (dil filtresi) gibi yapılandırılmış kolonlar var ve
  hibrit aramada bunlara ihtiyacımız var. LangChain'in embedding
  modeli (`HuggingFaceEmbeddings`) yine de kullanılıyor — istenen
  "LangChain kullanarak" şartı bu katmanda karşılanıyor.

Kurulum
-------
    pip install langchain langchain-huggingface sentence-transformers \
                supabase rank_bm25

Ortam değişkenleri (.env)
-------------------------
    SUPABASE_URL=...
    SUPABASE_SERVICE_KEY=...   # yazma işlemleri için service (secret) key şart
                               # (projenizde demo_kullanici_uret.py'nin de kullandığı isim)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from rank_bm25 import BM25Okapi
from supabase import Client, create_client

try:
    # Güncel, önerilen paket (langchain_community üzerinden gelen sürüm artık deprecated)
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - eski kurulumlar için geri düşüş
    from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger("cobuild.matchmaking")

EMBEDDING_MODEL_ADI = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BOYUTU = 384  # all-MiniLM-L6-v2 çıktısı 384 boyutludur
VARSAYILAN_TOP_K = 5
BM25_ADAY_HAVUZU_BOYUTU = 200  # hibrit aramada BM25'in üzerinde çalışacağı aday sayısı


# ============================================================================
# Hata sınıfları — çağıran kod (FastAPI endpoint'leri) bunları try/except ile
# yakalayıp anlamlı HTTP hatalarına çevirebilir.
# ============================================================================

class EslestirmeMotoruHatasi(Exception):
    """Bu modüldeki tüm hataların üst sınıfı."""


class VektorlemeHatasi(EslestirmeMotoruHatasi):
    """Embedding üretimi veya veritabanına yazma sırasında oluşan hata."""


class AramaHatasi(EslestirmeMotoruHatasi):
    """Semantik/hibrit arama sırasında oluşan hata."""


# ============================================================================
# Veri modelleri
# ============================================================================

@dataclass
class GelistiriciProfili:
    """Vektörlenecek yazılımcı profilini temsil eder."""
    developer_id: str          # profiles.id (uuid, string olarak)
    ad_soyad: str
    uzmanlik_alanlari: list[str] = field(default_factory=list)  # örn: ["Backend", "Mobil"]
    bildigi_diller: list[str] = field(default_factory=list)     # örn: ["Python", "React", "Go"]
    bio: str = ""

    def embedding_metnine_donustur(self) -> str:
        """
        Profildeki farklı alanları embedding modeline verilecek TEK bir
        metne birleştirir. Modelin dil/uzmanlık bilgisini iyi yakalaması
        için alanları etiketleyerek yazıyoruz.
        """
        parcalar = [
            f"Uzmanlık alanları: {', '.join(self.uzmanlik_alanlari)}." if self.uzmanlik_alanlari else "",
            f"Bildiği diller ve teknolojiler: {', '.join(self.bildigi_diller)}." if self.bildigi_diller else "",
            f"Hakkında: {self.bio}" if self.bio else "",
        ]
        return " ".join(p for p in parcalar if p).strip()

    @property
    def tum_skiller(self) -> list[str]:
        """skills[] kolonuna yazılacak birleşik liste (filtreleme ve BM25 için)."""
        # Sırayı koruyarak tekrarları temizle
        return list(dict.fromkeys([*self.uzmanlik_alanlari, *self.bildigi_diller]))


@dataclass
class EslesmeSonucu:
    """Bir arama sonucundaki tek bir yazılımcı eşleşmesini temsil eder."""
    developer_id: str
    ad_soyad: str
    skills: list[str]
    bio: str
    benzerlik_skoru: float          # 0-1 arası, saf semantik benzerlik (kosinüs)
    hibrit_skor: Optional[float] = None  # sadece hibrit aramada dolu olur
    uyum_skoru: Optional[int] = None  # 0-100, arayüzde gösterilecek "Uyum Skoru" — bkz. uyum_skorlarini_hesapla_ve_ata


def uyum_skorlarini_hesapla_ve_ata(sonuclar: list["EslesmeSonucu"]) -> list["EslesmeSonucu"]:
    """
    Her sonuca, arayüzde doğrudan gösterilebilecek 0-100 arası bir "Uyum Skoru" atar.

    Neden gerekli?
    --------------
    - `hibrit_skor` (Reciprocal Rank Fusion) ham haliyle küçük ve göreceli bir
      sayıdır (örn. 0.032) — kullanıcıya "%3 uyum" olarak göstermek yanıltıcı
      olur. Bunun yerine, dönen sonuç kümesi İÇİNDE en düşükten en yükseğe
      65-98 aralığına ölçekleniyor (en iyi eşleşme ~98, en zayıf gösterilen
      eşleşme bile ~65 — zaten sadece en iyi top_k sonuç gösteriliyor,
      hiçbiri gerçekte "kötü" değil).
    - `hibrit_skor` yoksa (yani `en_uygun_gelistiricileri_bul`'un saf semantik
      sonucundan geliyorsa), `benzerlik_skoru` zaten 0-1 aralığında bir
      kosinüs benzerliği olduğu için doğrudan yüzdeye çevriliyor.
    """
    if not sonuclar:
        return sonuclar

    hepsi_hibrit = all(s.hibrit_skor is not None for s in sonuclar)

    if hepsi_hibrit:
        skorlar = [s.hibrit_skor for s in sonuclar]  # type: ignore[misc]
        en_dusuk, en_yuksek = min(skorlar), max(skorlar)
        for s in sonuclar:
            if en_yuksek > en_dusuk:
                oran = (s.hibrit_skor - en_dusuk) / (en_yuksek - en_dusuk)  # type: ignore[operator]
                s.uyum_skoru = round(65 + oran * 33)
            else:
                s.uyum_skoru = 90
    else:
        for s in sonuclar:
            s.uyum_skoru = round(max(0.0, min(1.0, s.benzerlik_skoru)) * 100)

    return sonuclar


# ============================================================================
# Tekil (singleton) kaynaklar — embedding modeli ve Supabase client'ı her
# çağrıda yeniden oluşturmak yerine bir kere yükleyip tekrar kullanıyoruz.
# ============================================================================

@lru_cache(maxsize=1)
def _embedding_modelini_getir() -> HuggingFaceEmbeddings:
    """
    sentence-transformers modelini yükler. @lru_cache sayesinde model
    sadece ilk çağrıda diskten/diskten belleğe yüklenir, sonraki tüm
    çağrılar aynı örneği (instance) kullanır.
    """
    logger.info("Embedding modeli yükleniyor: %s", EMBEDDING_MODEL_ADI)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ADI,
        encode_kwargs={"normalize_embeddings": True},  # kosinüs benzerliği için normalize şart
    )


@lru_cache(maxsize=1)
def _supabase_client_getir() -> Client:
    url = os.environ.get("SUPABASE_URL")
    # Projenizde bu anahtar SUPABASE_SERVICE_KEY adıyla kullanılıyor
    # (bkz. .env.example, demo_kullanici_uret.py). Supabase'in resmi
    # dokümanlarında SUPABASE_SERVICE_ROLE_KEY ismi de yaygın olduğu için
    # ikisini de destekliyoruz — hangisi tanımlıysa o kullanılır.
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise VektorlemeHatasi(
            "SUPABASE_URL ve SUPABASE_SERVICE_KEY ortam değişkenleri tanımlı değil "
            "(.env dosyasını kontrol edin, bkz. .env.example)."
        )
    return create_client(url, key)


def _metni_embeddinge_cevir(metin: str) -> list[float]:
    if not metin or not metin.strip():
        raise VektorlemeHatasi("Embedding'e çevrilecek metin boş olamaz.")
    model = _embedding_modelini_getir()
    return model.embed_query(metin)


# ============================================================================
# 1) GELİŞTİRİCİ VEKTÖRLEME
# ============================================================================

def gelistirici_profilini_vektorle(gelistirici: GelistiriciProfili) -> bool:
    """
    Bir yazılımcının profilini embedding'e çevirir ve Supabase'e
    (developer_embeddings tablosu) kaydeder/günceller (upsert).

    Parametreler
    ------------
    gelistirici: GelistiriciProfili
        Vektörlenecek yazılımcı bilgileri.

    Dönüş
    -----
    bool: İşlem başarılıysa True. Hata durumunda VektorlemeHatasi fırlatılır
          (sessizce False dönmüyoruz ki çağıran taraf hatayı fark etmesin
          diye gizlenmiş olmasın).

    Örnek
    -----
        profil = GelistiriciProfili(
            developer_id="a1b2c3...",
            ad_soyad="Berna Y.",
            uzmanlik_alanlari=["Backend", "Yapay Zeka"],
            bildigi_diller=["Python", "FastAPI", "LangChain"],
            bio="RAG sistemleri ve LLM entegrasyonları üzerine çalışıyorum.",
        )
        gelistirici_profilini_vektorle(profil)
    """
    try:
        kaynak_metin = gelistirici.embedding_metnine_donustur()
        if not kaynak_metin:
            raise VektorlemeHatasi(
                f"'{gelistirici.ad_soyad}' için embedding'e çevrilecek yeterli bilgi yok "
                "(uzmanlık alanı, dil veya bio alanlarından en az biri dolu olmalı)."
            )

        vektor = _metni_embeddinge_cevir(kaynak_metin)

        if len(vektor) != EMBEDDING_BOYUTU:
            # Model değiştiyse ya da yanlış bir model yüklendiyse burada erken patlamak,
            # veritabanına yanlış boyutlu vektör yazılmasından çok daha güvenli.
            raise VektorlemeHatasi(
                f"Beklenmeyen embedding boyutu: {len(vektor)} (beklenen {EMBEDDING_BOYUTU}). "
                "supabase_schema.sql içindeki vector(...) boyutunu kontrol edin."
            )

        supabase = _supabase_client_getir()
        supabase.table("developer_embeddings").upsert(
            {
                "developer_id": gelistirici.developer_id,
                "full_name": gelistirici.ad_soyad,
                "skills": gelistirici.tum_skiller,
                "bio": gelistirici.bio,
                "kaynak_metin": kaynak_metin,
                "embedding": vektor,
            },
            on_conflict="developer_id",  # aynı geliştirici tekrar vektörlenirse güncelle
        ).execute()

        logger.info("Geliştirici vektörlendi: %s (%s)", gelistirici.ad_soyad, gelistirici.developer_id)
        return True

    except EslestirmeMotoruHatasi:
        raise  # kendi hata tiplerimizi olduğu gibi yukarı taşı
    except Exception as exc:  # beklenmeyen (network, supabase, model) hataları sarmala
        logger.exception("Geliştirici vektörlenirken beklenmeyen hata: %s", gelistirici.developer_id)
        raise VektorlemeHatasi(f"Vektörleme başarısız oldu: {exc}") from exc


# ============================================================================
# 2) TOP 5 SEMANTİK ARAMA
# ============================================================================

def en_uygun_gelistiricileri_bul(
    prd_metni: str,
    top_k: int = VARSAYILAN_TOP_K,
    filtre_diller: Optional[list[str]] = None,
) -> list[EslesmeSonucu]:
    """
    PRD metnini embedding'e çevirir ve pgvector'de kosinüs benzerliğine göre
    en yakın top_k yazılımcıyı döner (saf semantik arama, BM25 YOK).

    Parametreler
    ------------
    prd_metni: Founder fikrinden üretilen teknik şartname metni.
    top_k: Kaç sonuç döneceği (varsayılan 5).
    filtre_diller: Verilirse, sadece bu dillerden en az birine sahip
                   yazılımcılar aday havuzuna girer (örn. ["React"]).

    Dönüş
    -----
    list[EslesmeSonucu]: Benzerliğe göre azalan sırada, en fazla top_k eleman.
                         Eşleşme yoksa boş liste döner (hata değil).
    """
    try:
        if not prd_metni or not prd_metni.strip():
            raise AramaHatasi("PRD metni boş olamaz.")

        sorgu_vektoru = _metni_embeddinge_cevir(prd_metni)
        supabase = _supabase_client_getir()

        yanit = supabase.rpc(
            "match_developers",
            {
                "query_embedding": sorgu_vektoru,
                "match_count": top_k,
                "filter_skills": filtre_diller,
            },
        ).execute()

        sonuclar = [
            EslesmeSonucu(
                developer_id=satir["developer_id"],
                ad_soyad=satir["full_name"],
                skills=satir["skills"] or [],
                bio=satir.get("bio") or "",
                benzerlik_skoru=round(float(satir["benzerlik"]), 4),
            )
            for satir in (yanit.data or [])
        ]
        uyum_skorlarini_hesapla_ve_ata(sonuclar)

        logger.info("Semantik arama tamamlandı: %d sonuç bulundu.", len(sonuclar))
        return sonuclar

    except EslestirmeMotoruHatasi:
        raise
    except Exception as exc:
        logger.exception("Semantik arama sırasında beklenmeyen hata.")
        raise AramaHatasi(f"Semantik arama başarısız oldu: {exc}") from exc


# ============================================================================
# 3) HİBRİT ARAMA (BONUS): Semantik (vektör) + BM25 (anahtar kelime)
# ============================================================================
#
# Yöntem: Reciprocal Rank Fusion (RRF)
# -------------------------------------
# İki farklı arama yönteminin SKORLARINI doğrudan toplamak yanıltıcıdır
# (kosinüs benzerliği 0-1 arası, BM25 skoru ise sınırsız ve veri setine göre
# değişir). Bunun yerine, her iki listede de bir sonucun SIRASINI (rank)
# kullanan RRF yöntemini uyguluyoruz:
#
#     rrf_skor(d) = sum( 1 / (k + rank_liste_i(d)) )  tüm listeler için
#
# k genelde 60 alınır (literatürdeki standart varsayılan). Bu sayede bir
# yazılımcı ister sadece anlamsal aramada, ister sadece BM25'te üst
# sıralarda çıksın, her iki sinyal de nihai sıralamaya adil şekilde katkı
# verir.
# ============================================================================

_RRF_K = 60


def _bm25_icin_metni_hazirla(skills: list[str], bio: str) -> str:
    """BM25'in tokenize edeceği düz metni üretir (skills'e daha çok ağırlık için tekrar ekliyoruz)."""
    return " ".join([*skills, *skills, bio or ""]).lower()  # skills 2x tekrar = hafif ağırlıklandırma


def _tum_gelistirici_havuzunu_getir(limit: int = BM25_ADAY_HAVUZU_BOYUTU) -> list[dict]:
    """
    BM25 indeksini kurmak için veritabanındaki geliştirici havuzunu çeker.
    Not: Havuz çok büyürse (binlerce yazılımcı) bunu periyodik olarak
    önbelleğe almak (örn. Redis, ya da birkaç dakikalık in-memory cache)
    performans için önerilir — MVP ölçeğinde (onlarca/yüzlerce kayıt)
    doğrudan sorgulamak yeterlidir.
    """
    supabase = _supabase_client_getir()
    yanit = (
        supabase.table("developer_embeddings")
        .select("developer_id, full_name, skills, bio")
        .limit(limit)
        .execute()
    )
    return yanit.data or []


def hibrit_eslestirme_yap(
    prd_metni: str,
    gerekli_diller: Optional[list[str]] = None,
    top_k: int = VARSAYILAN_TOP_K,
    semantik_aday_sayisi: int = 20,
) -> list[EslesmeSonucu]:
    """
    Semantik (vektör) arama + BM25 anahtar kelime aramasını RRF ile
    birleştirip en iyi top_k yazılımcıyı döner.

    Parametreler
    ------------
    prd_metni: PRD metni (hem embedding hem BM25 sorgusu için kullanılır).
    gerekli_diller: PRD'den çıkarılmış "Beceri Etiketleri" varsa buraya
                    verin (örn. ["React", "PostgreSQL"]) — BM25 sorgusuna
                    eklenerek spesifik teknoloji isimlerinin kaçırılmaması
                    sağlanır. Verilmezse sadece prd_metni kullanılır.
    top_k: Nihai olarak döndürülecek sonuç sayısı.
    semantik_aday_sayisi: RRF'ye girecek aday sayısını artırmak için
                           semantik aramadan kaç sonuç çekileceği (top_k'dan
                           büyük tutulması önerilir, ör. 20).

    Dönüş
    -----
    list[EslesmeSonucu]: hibrit_skor alanı dolu, azalan sırada top_k sonuç.
    """
    try:
        if not prd_metni or not prd_metni.strip():
            raise AramaHatasi("PRD metni boş olamaz.")

        # --- 1) Semantik aday listesi -----------------------------------
        semantik_sonuclar = en_uygun_gelistiricileri_bul(
            prd_metni, top_k=semantik_aday_sayisi
        )
        semantik_sira = {
            s.developer_id: rank for rank, s in enumerate(semantik_sonuclar, start=1)
        }
        semantik_detay = {s.developer_id: s for s in semantik_sonuclar}

        # --- 2) BM25 aday listesi ----------------------------------------
        havuz = _tum_gelistirici_havuzunu_getir()
        if not havuz:
            logger.warning("BM25 için geliştirici havuzu boş, sadece semantik sonuçlar döndürülüyor.")
            return semantik_sonuclar[:top_k]

        bm25_dokumanlari = [
            _bm25_icin_metni_hazirla(satir["skills"] or [], satir.get("bio") or "").split()
            for satir in havuz
        ]
        bm25_index = BM25Okapi(bm25_dokumanlari)

        bm25_sorgu_metni = prd_metni
        if gerekli_diller:
            # Spesifik dilleri sorguya birkaç kez ekleyerek BM25'te ağırlıklarını artırıyoruz
            bm25_sorgu_metni += " " + " ".join(gerekli_diller * 3)
        bm25_sorgu_tokenlari = bm25_sorgu_metni.lower().split()

        bm25_skorlari = bm25_index.get_scores(bm25_sorgu_tokenlari)
        bm25_sirali = sorted(
            zip(havuz, bm25_skorlari), key=lambda x: x[1], reverse=True
        )
        bm25_sira = {
            satir["developer_id"]: rank
            for rank, (satir, _skor) in enumerate(bm25_sirali, start=1)
        }
        bm25_detay = {satir["developer_id"]: satir for satir in havuz}

        # --- 3) Reciprocal Rank Fusion ile birleştir ----------------------
        tum_developer_idler = set(semantik_sira) | set(bm25_sira)
        rrf_skorlari: dict[str, float] = {}
        for dev_id in tum_developer_idler:
            skor = 0.0
            if dev_id in semantik_sira:
                skor += 1.0 / (_RRF_K + semantik_sira[dev_id])
            if dev_id in bm25_sira:
                skor += 1.0 / (_RRF_K + bm25_sira[dev_id])
            rrf_skorlari[dev_id] = skor

        siralanmis_idler = sorted(rrf_skorlari, key=rrf_skorlari.get, reverse=True)[:top_k]

        nihai_sonuclar: list[EslesmeSonucu] = []
        for dev_id in siralanmis_idler:
            if dev_id in semantik_detay:
                s = semantik_detay[dev_id]
                nihai_sonuclar.append(
                    EslesmeSonucu(
                        developer_id=s.developer_id,
                        ad_soyad=s.ad_soyad,
                        skills=s.skills,
                        bio=s.bio,
                        benzerlik_skoru=s.benzerlik_skoru,
                        hibrit_skor=round(rrf_skorlari[dev_id], 5),
                    )
                )
            else:
                satir = bm25_detay[dev_id]
                nihai_sonuclar.append(
                    EslesmeSonucu(
                        developer_id=satir["developer_id"],
                        ad_soyad=satir["full_name"],
                        skills=satir["skills"] or [],
                        bio=satir.get("bio") or "",
                        benzerlik_skoru=0.0,  # semantik aday havuzunda hiç görünmedi
                        hibrit_skor=round(rrf_skorlari[dev_id], 5),
                    )
                )

        uyum_skorlarini_hesapla_ve_ata(nihai_sonuclar)

        logger.info("Hibrit eşleştirme tamamlandı: %d sonuç.", len(nihai_sonuclar))
        return nihai_sonuclar

    except EslestirmeMotoruHatasi:
        raise
    except Exception as exc:
        logger.exception("Hibrit eşleştirme sırasında beklenmeyen hata.")
        raise AramaHatasi(f"Hibrit eşleştirme başarısız oldu: {exc}") from exc


# ============================================================================
# PINECONE ALTERNATİFİ (notlar)
# ============================================================================
# Supabase yerine Pinecone kullanmak isterseniz:
#   pip install pinecone-client
#   from pinecone import Pinecone
#   pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
#   index = pc.Index("cobuild-gelistiriciler")
#
#   Vektörleme:  index.upsert(vectors=[(developer_id, vektor, {"full_name": ..., "skills": [...]})])
#   Arama:       index.query(vector=sorgu_vektoru, top_k=5, include_metadata=True)
#
# Dezavantajı: Supabase'de zaten olan `skills` filtresi ve BM25 için gereken
# "tüm havuzu çekme" işlemleri Pinecone'da metadata filtreleriyle ayrıca
# kurgulanmalı, ayrıca ek bir servis/hesap ve maliyet gerekir. MVP ölçeğinde
# (yüzlerce/birkaç bin yazılımcı) pgvector performans olarak yeterlidir.
# ============================================================================
