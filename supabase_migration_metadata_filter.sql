-- ============================================================================
-- Co-Build AI — Metadata Ön-Filtreleme Migration'ı (budget_type / sektor)
-- ============================================================================
-- Bu dosyayı Supabase Dashboard > SQL Editor'de ELLE çalıştırın.
-- Amaç: "Bütçe Türü" ve "Sektör" bilgisini pgvector aramasında ve BM25
-- havuzunda bir ön-filtre olarak kullanabilmek (bkz. matchmaking_engine.py
-- içindeki `filtre_metadata` parametresi).
-- ============================================================================

-- 1) developer_embeddings tablosuna yeni (opsiyonel) kolonlar ekle.
--    Mevcut kayıtlar NULL kalır — geriye dönük uyumlu, hiçbir sorguyu bozmaz.
alter table developer_embeddings add column if not exists budget_type text;
alter table developer_embeddings add column if not exists sektor text;

-- 2) match_developers RPC fonksiyonuna iki yeni opsiyonel parametre ve
--    WHERE koşulu ekleyin.
--
--    ÖNEMLİ: Bu dosyada fonksiyonun TAM tanımını (RETURNS TABLE tipleri,
--    LANGUAGE, SECURITY DEFINER/INVOKER gibi ayarları) bilmediğimiz için
--    riskli bir "CREATE OR REPLACE" ile üzerine yazmıyoruz. Bunun yerine:
--
--    a) Supabase Dashboard > Database > Functions > match_developers >
--       Definition kısmına gidip mevcut SQL'i açın.
--    b) Parametre listesine, `filter_skills` parametresinden HEMEN SONRA
--       şu iki satırı ekleyin:
--
--         , filter_budget_type text default null
--         , filter_sektor text default null
--
--    c) `where` koşuluna (mevcut `filter_skills` koşulunun yanına) şu iki
--       satırı ekleyin:
--
--         and (filter_budget_type is null or de.budget_type = filter_budget_type)
--         and (filter_sektor is null or de.sektor = filter_sektor)
--
--    Bilinen mevcut gövde (referans için, tarafınızca paylaşıldı):
--
--        begin
--            return query
--            select
--                de.developer_id,
--                de.full_name,
--                de.skills,
--                de.bio,
--                (1 - (de.embedding <=> query_embedding))::float as benzerlik
--            from developer_embeddings de
--            where (filter_skills is null or de.skills && filter_skills)
--              and (filter_budget_type is null or de.budget_type = filter_budget_type)
--              and (filter_sektor is null or de.sektor = filter_sektor)
--            order by de.embedding <=> query_embedding
--            limit match_count;
--        end;
--
--    d) "Save" / "Run" ile fonksiyonu güncelleyin.
--
-- Not: `filter_budget_type`/`filter_sektor` NULL gönderildiğinde davranış
-- birebir eskisiyle aynı kalır (backward-compatible) — main.py ve
-- eslestirme_endpoints.py'deki yeni alanlar hep opsiyonel.

-- 3) ÖNEMLİ — canlıda tespit edilen sorun: Supabase Dashboard'un "Edit
--    Function" ekranı, parametre listesini değiştirdiğinizde fonksiyonu
--    OLDUĞU GİBİ değiştirmek yerine YENİ bir overload (aynı isimli, farklı
--    parametreli ikinci bir fonksiyon) olarak ekleyebiliyor. Bu durumda
--    PostgREST hangi overload'ı çağıracağını seçemez ve şu hatayı verir:
--
--      "Could not choose the best candidate function between:
--       match_developers(query_embedding, match_count, filter_skills),
--       match_developers(query_embedding, match_count, filter_skills,
--                         filter_budget_type, filter_sektor)"
--
--    Eğer bu hatayı görüyorsanız, ESKİ (3 parametreli) overload'ı silin —
--    SQL Editor'de şunu çalıştırın:

drop function if exists public.match_developers(vector, integer, text[]);

--    Ardından Database > Functions listesinde `match_developers` için
--    SADECE TEK bir tanım kaldığını doğrulayın (5 parametreli olan).
