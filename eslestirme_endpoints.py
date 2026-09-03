"""
eslestirme_endpoints.py
============================================================================
Co-Build AI — Semantik Eşleştirme Motoru — FastAPI Endpoint'leri
============================================================================

Bu dosyayı mevcut `co-build-ai-server` FastAPI uygulamanıza şu şekilde
entegre edebilirsiniz (main.py içinde):

    from eslestirme_endpoints import router as eslestirme_router
    app.include_router(eslestirme_router)

Endpoint'ler
------------
POST /gelistirici/vektorle
    Bir yazılımcı profili kaydedildiğinde/güncellendiğinde çağrılır.
    (Örn. Next.js tarafında profil düzenleme formu kaydedilince tetiklenebilir.)

POST /eslestir/semantik-top5
    Sadece anlamsal (embedding) benzerliğine göre en iyi 5 yazılımcıyı döner.

POST /eslestir/hibrit
    Anlamsal + BM25 (anahtar kelime) hibrit aramasını döner (bonus özellik).
    PRD üretilirken çıkarılan "Beceri Etiketleri" varsa `gerekli_diller`
    alanına verilmesi önerilir — spesifik dillerin kaçırılmamasını sağlar.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from matchmaking_engine import (
    AramaHatasi,
    EslesmeSonucu,
    GelistiriciProfili,
    VektorlemeHatasi,
    en_uygun_gelistiricileri_bul,
    gelistirici_profilini_vektorle,
    hibrit_eslestirme_yap,
)

logger = logging.getLogger("cobuild.matchmaking.api")

router = APIRouter(prefix="", tags=["Eşleştirme Motoru"])


# ============================================================================
# İstek / Yanıt şemaları (Pydantic)
# ============================================================================

class GelistiriciVektorleIstek(BaseModel):
    developer_id: str = Field(..., description="profiles.id (uuid)")
    ad_soyad: str
    uzmanlik_alanlari: list[str] = Field(default_factory=list, description="Örn: ['Backend', 'Mobil']")
    bildigi_diller: list[str] = Field(default_factory=list, description="Örn: ['Python', 'React']")
    bio: str = ""
    butce_turu: str = Field(default="", description="Örn: 'Sabit Ücret' / 'Hisse' (metadata ön-filtreleme için)")
    sektor: str = Field(default="", description="Örn: 'Fintech' (metadata ön-filtreleme için)")


class GelistiriciVektorleYanit(BaseModel):
    basarili: bool
    mesaj: str


class SemantikAramaIstek(BaseModel):
    prd_metni: str = Field(..., min_length=1, description="Üretilen PRD metni")
    top_k: int = Field(default=5, ge=1, le=50)
    filtre_diller: list[str] | None = Field(
        default=None, description="Verilirse sadece bu dillerden birine sahip yazılımcılar aranır"
    )
    filtre_metadata: dict[str, str] | None = Field(
        default=None,
        description="Örn: {'budget_type': 'Sabit Ücret', 'sektor': 'Fintech'} — pgvector aramasını ön-filtreler",
    )


class HibritAramaIstek(BaseModel):
    prd_metni: str = Field(..., min_length=1, description="Üretilen PRD metni")
    gerekli_diller: list[str] | None = Field(
        default=None, description="PRD'den çıkarılan beceri etiketleri (varsa)"
    )
    top_k: int = Field(default=5, ge=1, le=50)
    filtre_metadata: dict[str, str] | None = Field(
        default=None,
        description="Örn: {'budget_type': 'Sabit Ücret', 'sektor': 'Fintech'} — hem semantik hem BM25 havuzunu ön-filtreler",
    )


class EslesmeSonucuYaniti(BaseModel):
    developer_id: str
    ad_soyad: str
    skills: list[str]
    bio: str
    benzerlik_skoru: float
    hibrit_skor: float | None = None
    uyum_skoru: int | None = None  # 0-100, arayüzde doğrudan gösterilecek yüzde

    @classmethod
    def from_dataclass(cls, sonuc: EslesmeSonucu) -> "EslesmeSonucuYaniti":
        return cls(
            developer_id=sonuc.developer_id,
            ad_soyad=sonuc.ad_soyad,
            skills=sonuc.skills,
            bio=sonuc.bio,
            benzerlik_skoru=sonuc.benzerlik_skoru,
            hibrit_skor=sonuc.hibrit_skor,
            uyum_skoru=sonuc.uyum_skoru,
        )


class AramaYaniti(BaseModel):
    sonuc_sayisi: int
    yazilimcilar: list[EslesmeSonucuYaniti]


# ============================================================================
# Endpoint 1: Geliştirici Vektörleme
# ============================================================================

@router.post("/gelistirici/vektorle", response_model=GelistiriciVektorleYanit)
def gelistirici_vektorle_endpoint(istek: GelistiriciVektorleIstek) -> GelistiriciVektorleYanit:
    """
    Bir yazılımcının profilini embedding'e çevirip veritabanına kaydeder.
    Aynı developer_id ile tekrar çağrılırsa mevcut kaydı GÜNCELLER (upsert).
    """
    try:
        profil = GelistiriciProfili(
            developer_id=istek.developer_id,
            ad_soyad=istek.ad_soyad,
            uzmanlik_alanlari=istek.uzmanlik_alanlari,
            bildigi_diller=istek.bildigi_diller,
            bio=istek.bio,
            butce_turu=istek.butce_turu,
            sektor=istek.sektor,
        )
        gelistirici_profilini_vektorle(profil)
        return GelistiriciVektorleYanit(
            basarili=True,
            mesaj=f"'{istek.ad_soyad}' başarıyla vektörlendi.",
        )

    except VektorlemeHatasi as exc:
        # Beklenen/bilinen hata: 400 (isteğin kendisiyle ilgili bir sorun,
        # örn. boş profil) döndürmek daha doğru olur.
        logger.warning("Vektörleme reddedildi: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:  # beklenmeyen her şey için genel güvenlik ağı
        logger.exception("Vektörleme endpoint'inde beklenmeyen hata.")
        raise HTTPException(
            status_code=500, detail="Sunucu tarafında beklenmeyen bir hata oluştu."
        ) from exc


# ============================================================================
# Endpoint 2: Top 5 Semantik Arama
# ============================================================================

@router.post("/eslestir/semantik-top5", response_model=AramaYaniti)
def semantik_arama_endpoint(istek: SemantikAramaIstek) -> AramaYaniti:
    """
    PRD metnini alır, embedding'e çevirir, pgvector'de kosinüs benzerliğine
    göre en yakın top_k yazılımcıyı döner (saf semantik arama).
    """
    try:
        sonuclar = en_uygun_gelistiricileri_bul(
            prd_metni=istek.prd_metni,
            top_k=istek.top_k,
            filtre_diller=istek.filtre_diller,
            filtre_metadata=istek.filtre_metadata,
        )
        return AramaYaniti(
            sonuc_sayisi=len(sonuclar),
            yazilimcilar=[EslesmeSonucuYaniti.from_dataclass(s) for s in sonuclar],
        )

    except AramaHatasi as exc:
        logger.warning("Semantik arama reddedildi: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Semantik arama endpoint'inde beklenmeyen hata.")
        raise HTTPException(
            status_code=500, detail="Sunucu tarafında beklenmeyen bir hata oluştu."
        ) from exc


# ============================================================================
# Endpoint 3: Hibrit Arama (Bonus)
# ============================================================================

@router.post("/eslestir/hibrit", response_model=AramaYaniti)
def hibrit_arama_endpoint(istek: HibritAramaIstek) -> AramaYaniti:
    """
    Semantik (vektör) + BM25 (anahtar kelime) aramasını RRF ile birleştirip
    en iyi top_k yazılımcıyı döner. `gerekli_diller` verilirse (PRD'den
    çıkarılan beceri etiketleri), spesifik teknoloji isimlerinin BM25
    tarafında kaçırılmaması sağlanır.
    """
    try:
        sonuclar = hibrit_eslestirme_yap(
            prd_metni=istek.prd_metni,
            gerekli_diller=istek.gerekli_diller,
            top_k=istek.top_k,
            filtre_metadata=istek.filtre_metadata,
        )
        return AramaYaniti(
            sonuc_sayisi=len(sonuclar),
            yazilimcilar=[EslesmeSonucuYaniti.from_dataclass(s) for s in sonuclar],
        )

    except AramaHatasi as exc:
        logger.warning("Hibrit arama reddedildi: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Hibrit arama endpoint'inde beklenmeyen hata.")
        raise HTTPException(
            status_code=500, detail="Sunucu tarafında beklenmeyen bir hata oluştu."
        ) from exc
