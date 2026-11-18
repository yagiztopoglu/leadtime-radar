"""
LeadTime Radar - Adim 5: Dogal dilde rapor (LLM) + guvenlik katmani

TASARIM ILKESI:
  - Tum SAYILARI Python hesaplar (tahmin, stok, stoksuz kalma gunu...).
  - Onerilecek AKSIYON adaylarini da Python kurallari secer.
  - LLM sadece bunlari akici Turkce'ye cevirir, yeni sayi uretmesi yasak.
  - Cikti, "izin verilen sayilar" listesiyle kontrol edilir. LLM listede olmayan
    bir sayi yazarsa rapor reddedilir, sablon (kural tabanli) rapora dusulur.
  - API anahtari yoksa / hata olursa arac yine calisir (sablon rapor).

API anahtari: proje klasorundeki .env dosyasina ANTHROPIC_API_KEY=... yaz (ornek: .env.example)
veya ortam degiskeni olarak tanimla. Anahtar koda yazilmaz.
"""
import json
import os
import re

try:  # .env varsa oku (python-dotenv kurulu degilse sessizce gec)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL = "claude-sonnet-5-5"   # istersen degistir
RISK_TR = {"Yuksek": "Yüksek", "Orta": "Orta", "Dusuk": "Düşük"}

SYSTEM_PROMPT = """Sen bir satın alma ekibine yardım eden asistansın. Sana bir siparişin \
HESAPLANMIŞ verileri JSON olarak verilecek.

KURALLAR:
- Sadece JSON'daki sayıları ve bilgileri kullan. Yeni sayı üretme, hesaplama yapma, tahmin ekleme.
- Sayıları JSON'da yazıldığı gibi aktar (örn. 8 gün). Ondalık sayı kullanma.
- Yazı Türkçe, profesyonel ve kısa olsun (en fazla 110 kelime).
- Biçim: önce 1-2 cümle durum özeti, sonra "Önerilen aksiyonlar:" başlığı ve "- " ile başlayan maddeler.
- Aksiyonları sadece "aksiyon_adaylari" listesinden seç ve cümleye dök; risk "Düşük" ise aksiyon gerekmediğini söyle.
- Numaralı liste kullanma. Abartılı, dramatik dil kullanma."""

WEEKLY_SYSTEM_PROMPT = """Sen bir satın alma müdürüne haftalık özet hazırlayan asistansın. Sana en riskli \
siparişlerin HESAPLANMIŞ verileri JSON listesi olarak verilecek.

KURALLAR:
- Sadece JSON'daki sayıları kullan. Yeni sayı üretme, toplama/çıkarma yapma.
- Türkçe, profesyonel, en fazla 150 kelime. Önce tek cümlelik genel durum, sonra her sipariş için tek satır ("- " ile).
- Her satırda tedarikçi, parça, stoksuz kalma günü ve önerilen tek aksiyon olsun.
- Numaralı liste kullanma."""


# ---------------------------------------------------------------------------
# 1) SAYILARI PYTHON HAZIRLAR
# ---------------------------------------------------------------------------
def build_facts(row) -> dict:
    """Bir siparis satirindan LLM'e verilecek 'gercekler' sozlugu."""
    risk = RISK_TR.get(row["risk"], row["risk"])
    shortage = int(round(row["shortage_days_p85"]))
    facts = {
        "po_no": int(row["po_id"]),
        "tedarikci": row.get("supplier_name", f"Tedarikçi {int(row['supplier_id'])}"),
        "parca": row["name"],
        "erp_ye_gore_varis_gun": int(round(row["erp_arrival_days"])),
        "model_p50_varis_gun": int(round(row["p50_arrival_days"])),
        "model_p85_varis_gun": int(round(row["p85_arrival_days"])),
        "stok_yeter_gun": int(round(row["days_to_zero"])),
        "p85_stoksuz_kalma_gun": shortage,
        "ek_ihtiyac_adet": int(round(row["extra_units_needed"])),
        "guvenlik_stogu_yetersiz": bool(row["p85_arrival_days"] > row["days_to_safety"]),
        "risk": risk,
    }
    facts["aksiyon_adaylari"] = suggest_actions(facts)
    return facts


def suggest_actions(f: dict) -> list:
    """Aksiyon adaylarini kural tabanli Python secer (LLM degil)."""
    if f["risk"] == "Yüksek":
        acts = ["Tedarikçiden teslimat tarihini yazılı olarak teyit et ve kısmi/erken sevkiyat talep et",
                "Üretim planlamayı olası stoksuz kalma konusunda bilgilendir"]
        if f["ek_ihtiyac_adet"] > 0:
            acts.append(f"Eksik kalacak yaklaşık {f['ek_ihtiyac_adet']} adet için alternatif tedarikçi "
                        "veya hızlı nakliye seçeneğini değerlendir")
        return acts
    if f["risk"] == "Orta":
        return ["Tedarikçiyle sipariş durumunu haftalık takip et",
                "Bu parça için güvenlik stoğu seviyesini gözden geçir"]
    return ["Aksiyon gerekmiyor, rutin takibe devam et"]


# ---------------------------------------------------------------------------
# 2) SABLON (KURAL TABANLI) RAPOR - LLM yokken veya reddedilince
# ---------------------------------------------------------------------------
def template_report(f: dict) -> str:
    if f["p85_stoksuz_kalma_gun"] > 0:
        head = (f"{f['tedarikci']} tedarikçisinin {f['parca']} parçasında ERP {f['erp_ye_gore_varis_gun']} gün "
                f"gösteriyor; model %85 ihtimalle {f['model_p85_varis_gun']} gün içinde geleceğini öngörüyor. "
                f"Stok {f['stok_yeter_gun']} gün yetiyor, yaklaşık {f['p85_stoksuz_kalma_gun']} gün stoksuz "
                f"kalma riski var. Risk seviyesi: {f['risk']}.")
    else:
        head = (f"{f['tedarikci']} tedarikçisinin {f['parca']} parçası için model %85 ihtimalle "
                f"{f['model_p85_varis_gun']} gün öngörüyor, stok {f['stok_yeter_gun']} gün yetiyor. "
                f"Risk seviyesi: {f['risk']}.")
    if f["guvenlik_stogu_yetersiz"] and f["risk"] != "Düşük":
        head += " Güvenlik stoğu bu senaryoda yetersiz kalıyor."
    return head + "\n\nÖnerilen aksiyonlar:\n" + "\n".join(f"- {a}" for a in f["aksiyon_adaylari"])


# ---------------------------------------------------------------------------
# 3) GUVENLIK KATMANI: LLM yeni sayi uydurduysa yakala
# ---------------------------------------------------------------------------
_NUM = re.compile(r"\d+")
ALWAYS_OK = {50, 85, 95}          # "P85", "%85" gibi etiketler

def allowed_numbers(obj) -> set:
    """Girdideki tum sayilar (metin icindeki rakam dizileri dahil: 'Tedarikci-07' -> 7)."""
    text = json.dumps(obj, ensure_ascii=False)
    return {int(n) for n in _NUM.findall(text)} | ALWAYS_OK

def validate_numbers(text: str, obj) -> list:
    ok = allowed_numbers(obj)
    return sorted({int(n) for n in _NUM.findall(text)} - ok)


# ---------------------------------------------------------------------------
# 4) LLM CAGRISI
# ---------------------------------------------------------------------------
def _call_llm(system: str, payload, api_key=None, client=None) -> str:
    if client is None:
        import anthropic  # pip install anthropic
        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=MODEL, max_tokens=500, temperature=0.2, system=system,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    return msg.content[0].text.strip()


def generate_report(row, api_key=None, client=None):
    """Donus: (rapor_metni, kaynak)  kaynak: 'LLM' | 'Şablon (API anahtarı yok)' | 'Şablon (LLM çıktısı doğrulanamadı)' ..."""
    facts = build_facts(row)
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if client is None and not key:
        return template_report(facts), "Şablon (API anahtarı yok)"
    try:
        text = _call_llm(SYSTEM_PROMPT, facts, api_key=key, client=client)
    except Exception as e:
        return template_report(facts), f"Şablon (LLM hatası: {type(e).__name__})"
    bad = validate_numbers(text, facts)
    if bad:
        return template_report(facts), f"Şablon (LLM çıktısı doğrulanamadı, listede olmayan sayı: {bad})"
    return text, "LLM"


def generate_weekly_summary(df_top, api_key=None, client=None):
    """En riskli siparisler icin tek paragraflik yonetici ozeti."""
    items = [build_facts(r) for _, r in df_top.iterrows()]
    slim = [{k: v for k, v in it.items() if k != "aksiyon_adaylari"} | {"aksiyon": it["aksiyon_adaylari"][0]}
            for it in items]
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    fallback = "Bu hafta öncelikli siparişler:\n" + "\n".join(
        f"- {s['tedarikci']} / {s['parca']}: {s['p85_stoksuz_kalma_gun']} gün stoksuz kalma riski. {s['aksiyon']}."
        for s in slim)
    if client is None and not key:
        return fallback, "Şablon (API anahtarı yok)"
    try:
        text = _call_llm(WEEKLY_SYSTEM_PROMPT, slim, api_key=key, client=client)
    except Exception as e:
        return fallback, f"Şablon (LLM hatası: {type(e).__name__})"
    bad = validate_numbers(text, slim)
    if bad:
        return fallback, f"Şablon (LLM çıktısı doğrulanamadı, listede olmayan sayı: {bad})"
    return text, "LLM"
