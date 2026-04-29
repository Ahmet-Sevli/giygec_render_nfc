from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os, json, firebase_admin, uuid, shutil, base64, tempfile, requests, re
from firebase_admin import credentials, firestore
from gradio_client import Client, file

app = FastAPI()

# ==========================================
# 1. FIREBASE KURULUMU
# ==========================================
firebase_key_raw = os.environ.get('FIREBASE_KEY')
db = None

if firebase_key_raw:
    try:
        cred = credentials.Certificate(json.loads(firebase_key_raw))
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase bağlantısı başarılı!")
    except Exception as e:
        print(f"❌ Firebase başlatılamadı: {e}")

# ==========================================
# 2. TEMEL ENDPOINT'LER
# ==========================================
@app.get("/")
async def root():
    return {"status": "ok", "message": "GİYGEÇ Sistemi Ayakta ve AI Kombin Motoru Devrede!"}

@app.get("/check-payment")
async def check_payment(uid: str):
    if not uid:
        return {"status": "error", "message": "UID eksik"}
    if not db:
        return {"status": "error", "message": "Veritabanı bağlantısı yok."}
    try:
        doc_ref = db.collection('products').document(uid)
        doc = doc_ref.get()
        if doc.exists:
            is_paid = doc.to_dict().get('isPaid', False)
            return {"paid": is_paid}
        return {"paid": False, "message": "Ürün bulunamadı"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==========================================
# 3. VIRTUAL TRY-ON (HuggingFace IDM-VTON)
# ==========================================
@app.post("/virtual-try-on")
def virtual_try_on(person_image: UploadFile = File(...), garment_image: UploadFile = File(...)):
    temp_dir = tempfile.gettempdir()
    temp_id = str(uuid.uuid4())
    person_path = os.path.join(temp_dir, f"person_{temp_id}.jpg")
    garment_path = os.path.join(temp_dir, f"garment_{temp_id}.jpg")

    try:
        with open(person_path, "wb") as buffer:
            shutil.copyfileobj(person_image.file, buffer)
        with open(garment_path, "wb") as buffer:
            shutil.copyfileobj(garment_image.file, buffer)

        hf_token = os.environ.get('HF_TOKEN')
        client = Client("yisol/IDM-VTON", token=hf_token)

        result = client.predict(
            dict={"background": file(person_path), "layers": [], "composite": None},
            garm_img=file(garment_path),
            garment_des="A stylish garment",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )

        with open(result[0], "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        return {"status": "success", "message": "Kıyafet başarıyla giydirildi!", "image_base64": encoded_string}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if os.path.exists(person_path): os.remove(person_path)
        if os.path.exists(garment_path): os.remove(garment_path)

# ==========================================
# 4. AI KOMBİN ÖNERİ SİSTEMİ (GELİŞTİRİLMİŞ)
# ==========================================
class CartItem(BaseModel):
    name: str = ""
    category: str = ""
    styleTags: list = []
    price: float = 0

class KombinRequest(BaseModel):
    user_text: str
    cart_items: list[CartItem] = []

@app.post("/kombin-oner")
async def kombin_oner(request: KombinRequest):
    if not db:
        raise HTTPException(status_code=500, detail="Veritabanı bağlantısı yok.")

    try:
        # 1. Sepet bilgisini Grok için hazırla
        cart_context = ""
        if request.cart_items:
            cart_lines = []
            for item in request.cart_items:
                tags_str = ", ".join(item.styleTags) if item.styleTags else "etiket yok"
                cart_lines.append(f"- {item.name} | Kategori: {item.category} | Etiketler: [{tags_str}]")
            cart_context = "\n".join(cart_lines)

        # 2. Grok API'ye Analiz Yaptır
        grok_analysis = ask_grok(request.user_text, cart_context)

        categories = grok_analysis.get("determined_categories", [])

        if not categories:
            return {"status": "success", "message": "Kategori anlaşılamadı.", "recommendations": [], "ai_explanation": ""}

        # 3. Firestore'dan Gerekli Kategorileri Çek
        db_products = fetch_products(categories)

        # 4. Sepetteki ürün ID'lerini filtrele (aynı ürünü önermemek için)
        cart_ids = set()
        if request.cart_items:
            for item in request.cart_items:
                if hasattr(item, 'id') and item.id:
                    cart_ids.add(item.id)

        # 5. Akıllı Puanlama
        best_matches = calculate_scores(db_products, grok_analysis, cart_ids)

        # 6. Kategori başına en iyi 1 ürün seç (kombin oluştur)
        kombin = pick_best_per_category(best_matches, categories)

        # 7. Grok'a son kombin onayı/stil notu yaptır
        style_note = ""
        if kombin:
            style_note = grok_analysis.get("explanation", "")

        return {
            "status": "success",
            "ai_explanation": style_note,
            "applied_filters": {
                "max_price": grok_analysis.get("max_price"),
                "excluded_tags": grok_analysis.get("exclude_tags", [])
            },
            "recommendations": kombin
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# --- YARDIMCI FONKSİYONLAR ---
# ==========================================

def ask_grok(user_text: str, cart_context: str = ""):
    """Grok API'ye istek atıp metni JSON parametrelere çevirir. Moda kuralları dahil."""
    grok_api_key = os.environ.get('GROK_API_KEY')
    if not grok_api_key:
        raise Exception("GROK_API_KEY bulunamadı!")

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json"
    }

    # ─── Moda Kuralları Entegre Edilmiş System Prompt ───
    system_prompt = """
Sen "GiyGeç" uygulamasının UZMAN AI moda danışmanısın. Kullanıcının doğal dil isteğini analiz et ve veritabanı sorgusu için JSON formatına çevir.

═══════════════════════════════════════
MODA KURALLARI (Bu kuralları her zaman uygula):
═══════════════════════════════════════

1. 60-30-10 RENK KURALI: Bir kombinde %60 ana renk, %30 ikincil renk, %10 vurgu rengi olmalı. Renk önerirken bu dengeyi gözet.

2. ORANT KURALI (1/3 - 2/3): Yüksek bel pantolon varsa üste kısa/crop/fitted parça öner. Uzun üst varsa yüksek bel değil normal bel öner. Vücudu 1/2 - 1/2 bölme!

3. HACİM DENGESİ: Üst oversize/dökümlü ise alt dar (slim/skinny) olmalı. Alt geniş (wide-leg/palazzo) ise üst fitted/oturan olmalı. İki geniş veya iki dar önerme!

4. DESEN KURALI: Bir kombinde en fazla 1 desenli parça olabilir. Üst desenli ise alt düz renk olmalı. Asla hem çizgili hem leopar önerme.

5. DRESS CODE UYUMU:
   - Smart Casual: Temiz beyaz sneaker tamam, koşu ayakkabısı değil. Jean ise koyu renk ve yırtıksız.
   - Business Formal: Siyah, lacivert, füme. Minimal desen.
   - Bohem: Toprak tonları, floral desenler, doğal kumaşlar.
   - Sportif: Jogger, sneaker, sweatshirt uyumu.

═══════════════════════════════════════
KURALLAR:
═══════════════════════════════════════

1. 'determined_categories': Kullanıcının isteğine göre [pantolon, tisort, elbise, ayakkabi] listesinden seç.
   - Elbise seçersen, pantolon veya tisort EKLEMEMELİSİN (elbise tek başına üst+alt).
   - Kullanıcı komple kombin istiyorsa uyumlu kategorileri beraber seç.
   - Sepette zaten olan kategorileri TEKRAR SEÇMEMELİSİN (kullanıcı özellikle istemediği sürece).

2. 'style_analysis': Kullanıcının istediği tarz, renk, kumaş, ortam etiketlerini belirle VE her bir etiketin EŞ ANLAMLILARINI da ekle.
   Örnek: Kullanıcı "nişan" dediyse → "nişan": 0.9, "düğün": 0.7, "abiye": 0.8, "özel gün": 0.6, "şık": 0.7, "elegant": 0.5
   Örnek: Kullanıcı "yazlık" dediyse → "yazlık": 0.9, "yaz": 0.8, "hafif": 0.6, "ince": 0.5, "serin": 0.4
   Her birine 0.1 ile 1.0 arası ağırlık ver. Ana kelime en yüksek, eşanlamlılar kademeli olarak düşük.

3. 'exclude_tags': İstenmeyenler. Moda kurallarına göre sen de ekle (örn: üst desenli ise exclude_tags'e diğer desenleri ekle). Yoksa boş liste [].

4. 'max_price': Bütçe belirtildiyse sayı olarak. Belirtilmediyse null.

5. 'volume_preference': Hacim dengesi kuralına göre üst veya altın dar/geniş olması gerekiyorsa belirt.
   Format: {"üst": "fitted", "alt": "geniş"} veya {"üst": "oversize", "alt": "dar"} veya null.

6. 'explanation': Kararını ve moda kuralı gerekçeni anlatan 1-2 cümle.

7. SADECE GEÇERLİ JSON DÖN. Markdown, backtick veya başka metin KULLANMA.
"""

    # Kullanıcı mesajını oluştur (sepet bağlamı ile)
    user_message = user_text
    if cart_context:
        user_message = f"""Kullanıcının isteği: {user_text}

Kullanıcının sepetindeki mevcut ürünler:
{cart_context}

Sepetteki ürünleri dikkate alarak, eksik parçaları tamamla. Sepette olan kategorileri tekrar önerme."""

    data = {
        "model": "grok-3-mini-fast",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.15
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    result_text = response.json()['choices'][0]['message']['content']

    # Güçlü JSON temizleme
    result_text = clean_json_response(result_text)

    return json.loads(result_text)


def clean_json_response(text: str) -> str:
    """Grok'un döndürdüğü metinden JSON'u temiz bir şekilde çıkarır."""
    # 1. ```json ... ``` bloğu varsa içini al
    json_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if json_block:
        return json_block.group(1).strip()

    # 2. { ... } bloğunu bul
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        return brace_match.group(0).strip()

    # 3. Hiçbiri yoksa olduğu gibi dön
    return text.strip()


def fetch_products(categories: list):
    """Firestore'da 'catalog → [kategori] → products' altındaki ürünleri çeker."""
    products = []
    for cat in categories:
        try:
            docs = db.collection("catalog").document(cat).collection("products").stream()
            for doc in docs:
                product_data = doc.to_dict()
                product_data["id"] = doc.id  # Doküman ID'sini ekle
                product_data["_source_category"] = cat  # Hangi kategoriden geldiğini ekle
                products.append(product_data)
        except Exception as e:
            print(f"HATA: {cat} kategorisi çekilemedi - {e}")
    return products


def calculate_scores(products: list, grok_analysis: dict, cart_ids: set = None):
    """Grok'tan gelen analizdeki filtrelere ve ağırlıklara göre ürünleri puanlar."""
    weights = grok_analysis.get("style_analysis", {})
    excludes = grok_analysis.get("exclude_tags", [])
    max_price = grok_analysis.get("max_price")

    scored_products = []
    excludes_lower = [ex.lower() for ex in excludes] if excludes else []

    for product in products:
        product_id = product.get("id", "")
        product_price = product.get("price", 0)
        product_tags = [tag.lower() for tag in product.get("styleTags", [])]

        # Sepette zaten olan ürünü önerme
        if cart_ids and product_id in cart_ids:
            continue

        # KESİN FİLTRE: Bütçe
        if max_price is not None and product_price > max_price:
            continue

        # KESİN FİLTRE: İstenmeyen etiketler
        if any(bad_tag in product_tags for bad_tag in excludes_lower):
            continue

        # PUANLAMA
        score = 0.0
        matched_tags = []
        for ai_tag, weight in weights.items():
            ai_tag_lower = ai_tag.lower()
            # Tam eşleşme
            if ai_tag_lower in product_tags:
                score += float(weight)
                matched_tags.append(ai_tag)
            else:
                # Kısmi eşleşme (contains) — "yazlık" → "yaz" gibi
                for pt in product_tags:
                    if ai_tag_lower in pt or pt in ai_tag_lower:
                        score += float(weight) * 0.6  # Kısmi eşleşme düşük puan
                        matched_tags.append(f"{ai_tag}~{pt}")
                        break

        if score > 0:
            product["ai_match_score"] = round(score, 2)
            product["matched_tags"] = matched_tags
            scored_products.append(product)

    scored_products.sort(key=lambda x: x.get("ai_match_score", 0), reverse=True)
    return scored_products


def pick_best_per_category(scored_products: list, categories: list):
    """Her kategori için en yüksek puanlı 1 ürünü seçer → Kombin oluşturur."""
    kombin = []
    used_categories = set()

    for product in scored_products:
        cat = product.get("_source_category", product.get("category", ""))
        if cat not in used_categories:
            used_categories.add(cat)
            # Dahili alanları temizle
            clean_product = {k: v for k, v in product.items() if not k.startswith("_")}
            kombin.append(clean_product)

        if len(used_categories) >= len(categories):
            break

    return kombin
