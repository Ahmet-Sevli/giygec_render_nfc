from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os, json, firebase_admin, uuid, shutil, base64, tempfile, requests, re
from firebase_admin import credentials, firestore
from gradio_client import Client, file as gradio_file

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
    return {"status": "ok", "message": "GİYGEÇ Sistemi — AI Kombin Motoru & Sanal Kabin Devrede!"}

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
# 3. VIRTUAL TRY-ON — TEK PARÇA
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
            dict={"background": gradio_file(person_path), "layers": [], "composite": None},
            garm_img=gradio_file(garment_path),
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
# 4. VIRTUAL TRY-ON — ZİNCİRLEME KOMBİN
# ==========================================
class ChainedTryOnRequest(BaseModel):
    person_image_base64: str
    garments: list  # [{"image_url": "...", "description": "Siyah slim fit pantolon, dar kesim, uzun paça"}, ...]

@app.post("/virtual-try-on-chain")
async def virtual_try_on_chain(request: ChainedTryOnRequest):
    """
    Zincirleme sanal deneme: Birden fazla kıyafeti sırayla giydirir.
    1. Kullanıcı fotoğrafı + ilk kıyafet → sonuç1
    2. sonuç1 + ikinci kıyafet → sonuç2
    3. sonuç2 + üçüncü kıyafet → sonuç3 (final)
    """
    temp_dir = tempfile.gettempdir()
    session_id = str(uuid.uuid4())
    hf_token = os.environ.get('HF_TOKEN')
    temp_files = []

    try:
        # Kullanıcı fotoğrafını decode et
        person_bytes = base64.b64decode(request.person_image_base64)
        current_person_path = os.path.join(temp_dir, f"chain_person_{session_id}.jpg")
        with open(current_person_path, "wb") as f:
            f.write(person_bytes)
        temp_files.append(current_person_path)

        client = Client("yisol/IDM-VTON", token=hf_token)

        progress_results = []  # Her adımın sonucunu tut

        for i, garment in enumerate(request.garments):
            garment_url = garment.get("image_url", "")
            garment_desc = garment.get("description", "A stylish garment")

            if not garment_url:
                continue

            # Kıyafet resmini URL'den indir
            garment_path = os.path.join(temp_dir, f"chain_garment_{session_id}_{i}.jpg")
            temp_files.append(garment_path)

            img_response = requests.get(garment_url, timeout=30)
            img_response.raise_for_status()
            with open(garment_path, "wb") as f:
                f.write(img_response.content)

            # HuggingFace'e gönder
            result = client.predict(
                dict={"background": gradio_file(current_person_path), "layers": [], "composite": None},
                garm_img=gradio_file(garment_path),
                garment_des=garment_desc,
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )

            # Sonucu bir sonraki adımın "person" resmi olarak kaydet
            next_person_path = os.path.join(temp_dir, f"chain_result_{session_id}_{i}.jpg")
            temp_files.append(next_person_path)
            shutil.copy2(result[0], next_person_path)
            current_person_path = next_person_path

            # Bu adımın sonuç resmini base64 olarak kaydet
            with open(result[0], "rb") as img_file:
                step_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                progress_results.append({
                    "step": i + 1,
                    "garment_name": garment.get("name", f"Parça {i+1}"),
                    "image_base64": step_base64
                })

        if not progress_results:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Hiçbir kıyafet giydirilemedi."})

        return {
            "status": "success",
            "message": f"{len(progress_results)} parça başarıyla giydirildi!",
            "final_image_base64": progress_results[-1]["image_base64"],
            "steps": progress_results
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        for tf in temp_files:
            if os.path.exists(tf):
                try:
                    os.remove(tf)
                except:
                    pass

# ==========================================
# 4b. VIRTUAL TRY-ON — TEK ADIM (Adım Adım Canlı Giydirme)
# ==========================================
class SingleStepTryOnRequest(BaseModel):
    person_image_base64: str
    garment_image_url: str
    garment_description: str = "A stylish garment"

@app.post("/virtual-try-on-step")
async def virtual_try_on_step(request: SingleStepTryOnRequest):
    """
    Tek bir kıyafeti giydirir ve sonucu döner.
    Flutter bu endpoint'i her kıyafet için sırayla çağırır.
    Bir önceki sonucu person_image olarak gönderir → zincirleme etki.
    """
    temp_dir = tempfile.gettempdir()
    session_id = str(uuid.uuid4())
    hf_token = os.environ.get('HF_TOKEN')
    temp_files = []

    try:
        # Kullanıcı/önceki sonuç fotoğrafını decode et
        person_bytes = base64.b64decode(request.person_image_base64)
        person_path = os.path.join(temp_dir, f"step_person_{session_id}.jpg")
        with open(person_path, "wb") as f:
            f.write(person_bytes)
        temp_files.append(person_path)

        # Kıyafet resmini URL'den indir
        garment_path = os.path.join(temp_dir, f"step_garment_{session_id}.jpg")
        img_response = requests.get(request.garment_image_url, timeout=30)
        img_response.raise_for_status()
        with open(garment_path, "wb") as f:
            f.write(img_response.content)
        temp_files.append(garment_path)

        # HuggingFace'e gönder
        client = Client("yisol/IDM-VTON", token=hf_token)
        result = client.predict(
            dict={"background": gradio_file(person_path), "layers": [], "composite": None},
            garm_img=gradio_file(garment_path),
            garment_des=request.garment_description,
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=20,
            seed=42,
            api_name="/tryon"
        )

        with open(result[0], "rb") as img_file:
            result_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        return {
            "status": "success",
            "message": "Kıyafet başarıyla giydirildi!",
            "result_image_base64": result_base64
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        for tf in temp_files:
            if os.path.exists(tf):
                try:
                    os.remove(tf)
                except:
                    pass

# ==========================================
# 5. AI KOMBİN ÖNERİ SİSTEMİ
# ==========================================
class CartItemModel(BaseModel):
    id: str = ""
    name: str = ""
    category: str = ""
    styleTags: list = []
    price: float = 0

class KombinRequest(BaseModel):
    user_text: str
    cart_items: List[CartItemModel] = []

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
        grok_analysis = ask_grok_kombin(request.user_text, cart_context)

        categories = grok_analysis.get("determined_categories", [])

        if not categories:
            return {"status": "success", "message": "Kategori anlaşılamadı.", "recommendations": [], "ai_explanation": ""}

        # 3. Firestore'dan Gerekli Kategorileri Çek
        db_products = fetch_products(categories)

        # 4. Sepetteki ürün ID'lerini filtrele
        cart_ids = set()
        if request.cart_items:
            for item in request.cart_items:
                if item.id:
                    cart_ids.add(item.id)

        # 5. Akıllı Puanlama
        best_matches = calculate_scores(db_products, grok_analysis, cart_ids)

        # 6. Kategori başına en iyi 1 ürün seç (kombin oluştur)
        kombin = pick_best_per_category(best_matches, categories)

        return {
            "status": "success",
            "ai_explanation": grok_analysis.get("explanation", ""),
            "applied_filters": {
                "max_price": grok_analysis.get("max_price"),
                "excluded_tags": grok_analysis.get("exclude_tags", [])
            },
            "recommendations": kombin
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 6. GROK VİSİON ANALİZİ (Render Üzerinden)
# ==========================================
class VisionRequest(BaseModel):
    image_base64: str
    mime_type: str = "image/jpeg"
    user_text: str = ""
    cart_items: List[CartItemModel] = []

@app.post("/vision-analiz")
async def vision_analiz(request: VisionRequest):
    if not db:
        raise HTTPException(status_code=500, detail="Veritabanı bağlantısı yok.")

    try:
        # 1. Sepet bilgisini hazırla
        cart_context = ""
        if request.cart_items:
            cart_lines = []
            for item in request.cart_items:
                tags_str = ", ".join(item.styleTags) if item.styleTags else "etiket yok"
                cart_lines.append(f"- {item.name} | Kategori: {item.category} | Etiketler: [{tags_str}]")
            cart_context = "\n".join(cart_lines)

        # 2. Grok Vision API'ye gönder
        grok_analysis = ask_grok_vision(request.image_base64, request.mime_type, request.user_text, cart_context)

        categories = grok_analysis.get("determined_categories", [])

        if not categories:
            return {"status": "success", "message": "Fotoğraftan kategori belirlenemedi.", "recommendations": [], "ai_explanation": ""}

        # 3. Firestore'dan Çek
        db_products = fetch_products(categories)

        # 4. Sepet filtrele
        cart_ids = set()
        if request.cart_items:
            for item in request.cart_items:
                if item.id:
                    cart_ids.add(item.id)

        # 5. Puanla
        best_matches = calculate_scores(db_products, grok_analysis, cart_ids)

        # 6. Kombin oluştur
        kombin = pick_best_per_category(best_matches, categories)

        return {
            "status": "success",
            "ai_explanation": grok_analysis.get("explanation", ""),
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

# ─── MODA KURALLARI (Ortak System Prompt) ───
FASHION_RULES_PROMPT = """
Sen "GiyGeç" uygulamasının UZMAN AI moda danışmanısın.

═══════════════════════════════════════
EVRENSEL MODA KURALLARI (Her zaman uygula):
═══════════════════════════════════════

1. 60-30-10 RENK KURALI:
   - %60 Ana renk (pantolon/etek gibi geniş alan kaplayan parça)
   - %30 İkincil renk (tişört/gömlek gibi üst parça)
   - %10 Vurgu rengi (ayakkabı/aksesuar)
   Renk önerirken bu dengeyi koru.

2. ORANT KURALI (1/3 - 2/3):
   - Yüksek bel pantolon + kısa/crop üst = DOĞRU
   - Uzun üst + düşük bel = YANLIŞ (boyu kısa gösterir)
   - Yüksek bel geniş pantolon gördüğünde fitted/crop üst öner

3. HACİM DENGESİ:
   - Üst oversize → Alt dar (slim/skinny) olmalı
   - Alt geniş (wide-leg/palazzo) → Üst fitted/oturan olmalı
   - İki geniş veya iki dar parça ASLA ÖNERME

4. DESEN KURALI:
   - Bir kombinde MAX 1 desenli parça
   - Üst desenli ise alt MUTLAKA düz renk
   - Çizgili + leopar = ASLA

5. DRESS CODE:
   - Smart Casual: Temiz sneaker OK, koşu ayakkabısı HAYIR. Jean = koyu + yırtıksız
   - Business Formal: Siyah/lacivert/füme. Minimal desen
   - Bohem: Toprak tonları, floral, doğal kumaş
   - Sportif: Jogger, sneaker, sweatshirt

6. ELBİSE KURALI: Elbise seçersen pantolon/tişört EKLEME (elbise üst+alt'ı kapsar). Sadece ayakkabı/aksesuar ekle.

═══════════════════════════════════════
JSON ÇIKTI KURALLARI:
═══════════════════════════════════════

1. 'determined_categories': [pantolon, tisort, elbise, ayakkabi] listesinden seç.
   - Sepette olan kategorileri TEKRAR SEÇME (kullanıcı özellikle istemedikçe).

2. 'style_analysis': Kullanıcının istediği etiketleri VE EŞ ANLAMLILARINI üret.
   Örnek: "nişan" → {"nişan": 0.9, "düğün": 0.7, "abiye": 0.8, "özel gün": 0.6, "şık": 0.7, "elegant": 0.5}
   Örnek: "yazlık" → {"yazlık": 0.9, "yaz": 0.8, "hafif": 0.6, "ince": 0.5, "serin": 0.4}
   Her zaman en az 6-8 etiket üret.

3. 'exclude_tags': İstenmeyen + moda kuralına göre uyumsuz etiketler. Boşsa [].

4. 'max_price': Bütçe varsa sayı, yoksa null.

5. 'explanation': Kararını ve moda kuralı gerekçeni anlatan 2-3 cümle. Türkçe. Kullanıcıya hitap et.

6. SADECE GEÇERLİ JSON DÖN. Markdown/backtick KULLANMA.
"""


def ask_grok_kombin(user_text: str, cart_context: str = ""):
    """Metin tabanlı kombin analizi için Grok'a istek atar."""
    grok_api_key = os.environ.get('GROK_API_KEY')
    if not grok_api_key:
        raise Exception("GROK_API_KEY bulunamadı!")

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json"
    }

    user_message = user_text
    if cart_context:
        user_message = f"""Kullanıcının isteği: {user_text}

Kullanıcının sepetindeki mevcut ürünler:
{cart_context}

Sepetteki ürünleri dikkate alarak eksik parçaları tamamla. Sepette olan kategorileri tekrar önerme. Hacim ve renk dengesini sepetteki ürünlere göre ayarla."""

    data = {
        "model": "grok-4-1-fast-non-reasoning",
        "messages": [
            {"role": "system", "content": FASHION_RULES_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.15
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    result_text = response.json()['choices'][0]['message']['content']
    result_text = clean_json_response(result_text)

    return json.loads(result_text)


def ask_grok_vision(image_base64: str, mime_type: str, user_text: str = "", cart_context: str = ""):
    """Görsel tabanlı kombin analizi için Grok Vision'a istek atar."""
    grok_api_key = os.environ.get('GROK_API_KEY')
    if not grok_api_key:
        raise Exception("GROK_API_KEY bulunamadı!")

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json"
    }

    text_content = "Fotoğraftaki kıyafeti analiz et ve uyumlu tamamlayıcı parçalar öner."
    if user_text:
        text_content += f"\n\nKullanıcının ek isteği: {user_text}"
    if cart_context:
        text_content += f"\n\nKullanıcının sepetindeki ürünler:\n{cart_context}\nSepetteki ürünleri dikkate al."

    data = {
        "model": "grok-4-1-fast-non-reasoning",
        "messages": [
            {"role": "system", "content": FASHION_RULES_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_content},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
                    }
                ]
            }
        ],
        "temperature": 0.15
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    result_text = response.json()['choices'][0]['message']['content']
    result_text = clean_json_response(result_text)

    return json.loads(result_text)


def clean_json_response(text: str) -> str:
    """Grok'un döndürdüğü metinden JSON'u güvenli şekilde çıkarır."""
    # 1. ```json ... ``` bloğu varsa içini al
    json_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if json_block:
        return json_block.group(1).strip()

    # 2. { ... } bloğunu bul
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        return brace_match.group(0).strip()

    # 3. Olduğu gibi dön
    return text.strip()


def fetch_products(categories: list):
    """Firestore'da 'catalog → [kategori] → products' altındaki ürünleri çeker."""
    products = []
    for cat in categories:
        try:
            docs = db.collection("catalog").document(cat).collection("products").stream()
            for doc in docs:
                product_data = doc.to_dict()
                product_data["id"] = doc.id
                product_data["_source_category"] = cat
                products.append(product_data)
        except Exception as e:
            print(f"HATA: {cat} kategorisi çekilemedi - {e}")
    return products


def calculate_scores(products: list, grok_analysis: dict, cart_ids: set = None):
    """Ürünleri Grok ağırlıklarına göre puanlar."""
    weights = grok_analysis.get("style_analysis", {})
    excludes = grok_analysis.get("exclude_tags", [])
    max_price = grok_analysis.get("max_price")

    scored_products = []
    excludes_lower = [ex.lower() for ex in excludes] if excludes else []

    for product in products:
        product_id = product.get("id", "")
        product_price = product.get("price", 0)
        product_tags = [tag.lower() for tag in product.get("styleTags", [])]

        # Sepette olan ürünü önerme
        if cart_ids and product_id in cart_ids:
            continue

        # Bütçe filtresi
        if max_price is not None and product_price > max_price:
            continue

        # İstenmeyen etiket filtresi
        if any(bad_tag in product_tags for bad_tag in excludes_lower):
            continue

        # Puanlama
        score = 0.0
        matched_tags = []
        for ai_tag, weight in weights.items():
            ai_tag_lower = ai_tag.lower()
            # Tam eşleşme
            if ai_tag_lower in product_tags:
                score += float(weight)
                matched_tags.append(ai_tag)
            else:
                # Kısmi eşleşme (contains)
                for pt in product_tags:
                    if ai_tag_lower in pt or pt in ai_tag_lower:
                        score += float(weight) * 0.6
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
