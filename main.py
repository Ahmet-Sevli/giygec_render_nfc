from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os, json, firebase_admin, uuid, shutil, base64, tempfile, requests, re
from firebase_admin import credentials, firestore
from gradio_client import Client, file as gradio_file
from datetime import datetime

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
    category: str = "üst giyim"

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
    gender: str = ""  # "erkek" veya "kadın"

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
        grok_analysis = ask_grok_kombin(request.user_text, cart_context, request.gender)

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

        # 5. Akıllı Puanlama (cinsiyet filtresiyle)
        best_matches = calculate_scores(db_products, grok_analysis, cart_ids, request.gender)

        # 6. Benzersiz alternatif kombinleri üret (maksimum 3 adet)
        all_combinations = pick_multiple_kombins(best_matches, categories, max_kombins=3)
        if not all_combinations:
            return {"status": "success", "message": "Yeterli ürün bulunamadı.", "results": []}

        # 7. Her kombin için ayrı bir TTS özeti üret (tek API çağrısıyla)
        summaries = generate_multiple_summaries(all_combinations, grok_analysis.get("explanation", ""))

        # 8. Sonuçları hazırla (her kombine kendi açıklaması ve popülerlik puanı atanır)
        results = []
        for i, komb in enumerate(all_combinations):
            pop = calculate_popularity(komb)
            results.append({
                "products": komb,
                "ai_explanation": summaries[i] if i < len(summaries) else grok_analysis.get("explanation", ""),
                "popularity": pop
            })

        return {
            "status": "success",
            "applied_filters": {
                "max_price": grok_analysis.get("max_price"),
                "excluded_tags": grok_analysis.get("exclude_tags", [])
            },
            "results": results
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
    gender: str = ""

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
        grok_analysis = ask_grok_vision(request.image_base64, request.mime_type, request.user_text, cart_context, request.gender)

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

        # 5. Akıllı Puanlama (cinsiyet filtresiyle)
        best_matches = calculate_scores(db_products, grok_analysis, cart_ids, request.gender)

        # 6. Kombinleri üret
        all_combinations = pick_multiple_kombins(best_matches, categories, max_kombins=3)
        if not all_combinations:
            return {"status": "success", "message": "Yeterli ürün bulunamadı.", "results": []}

        # 7. Her kombin için ayrı bir TTS özeti üret
        summaries = generate_multiple_summaries(all_combinations, grok_analysis.get("explanation", ""))

        # 8. Sonuçları hazırla
        results = []
        for i, komb in enumerate(all_combinations):
            pop = calculate_popularity(komb)
            results.append({
                "products": komb,
                "ai_explanation": summaries[i] if i < len(summaries) else grok_analysis.get("explanation", ""),
                "popularity": pop
            })

        return {
            "status": "success",
            "applied_filters": {
                "max_price": grok_analysis.get("max_price"),
                "excluded_tags": grok_analysis.get("exclude_tags", [])
            },
            "results": results
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
CİNSİYET KURALI (KRİTİK):
═══════════════════════════════════════
Kullanıcının cinsiyeti belirtilmişse SADECE o cinsiyete uygun ürünler öner.
- Erkek → erkek pantolonu, erkek tişörtü, erkek ayakkabısı. ASLA kadın ürünü önerme.
- Kadın → kadın pantolonu, kadın elbisesi, kadın ayakkabısı. ASLA erkek ürünü önerme.
style_analysis'e cinsiyet etiketini de ekle (örn: "erkek": 0.9 veya "kadın": 0.9).
Ayrıca HER ZAMAN "unisex" etiketini de style_analysis içine ekle (örn: "unisex": 0.7), böylece unisex ürünler de uygun şekilde eşleşebilsin.

═══════════════════════════════════════
EVRENSEL MODA KURALLARI (Her zaman uygula):
═══════════════════════════════════════

1. 60-30-10 RENK KURALI:
   - %60 Ana renk (pantolon/etek gibi geniş alan kaplayan parça)
   - %30 İkincil renk (tişört/gömlek gibi üst parça)
   - %10 Vurgu rengi (ayakkabı/aksesuar)

2. ORANT KURALI (1/3 - 2/3):
   - Yüksek bel pantolon + kısa/crop üst = DOĞRU
   - Uzun üst + düşük bel = YANLIŞ

3. HACİM DENGESİ:
   - Üst oversize → Alt dar olmalı
   - Alt geniş → Üst fitted olmalı

4. DESEN KURALI:
   - Bir kombinde MAX 1 desenli parça

5. DRESS CODE:
   - Smart Casual: Temiz sneaker OK, koşu ayakkabısı HAYIR
   - Business Formal: Siyah/lacivert/füme
   - Bohem: Toprak tonları, floral, doğal kumaş
   - Sportif: Jogger, sneaker, sweatshirt

6. ELBİSE KURALI: Elbise seçersen pantolon/tişört EKLEME.

═══════════════════════════════════════
JSON ÇIKTI KURALLARI:
═══════════════════════════════════════

1. 'determined_categories': [pantolon, tisort, elbise, ayakkabi] listesinden seç. Sadece bu 4 kategoriden bahset, gömlek/kazak gibi diğerlerini kullanma!

2. 'style_analysis': Kullanıcının istediği HER KELİME için kapsamlı eş anlamlıları üret.
   HER kelime için en az 5-8 eş anlamlı/ilişkili etiket üret. Örnekler:
   "nişan" → {"nişan": 0.9, "düğün": 0.7, "abiye": 0.8, "özel gün": 0.6, "şık": 0.7, "elegant": 0.5, "davet": 0.6, "gece": 0.4}
   "yazlık" → {"yazlık": 0.9, "yaz": 0.8, "hafif": 0.6, "ince": 0.5, "serin": 0.4, "plaj": 0.3, "tatil": 0.4}
   "spor" → {"spor": 0.9, "sportif": 0.8, "rahat": 0.7, "casual": 0.6, "günlük": 0.5, "aktif": 0.5, "fitness": 0.4}
   "klasik" → {"klasik": 0.9, "formal": 0.8, "iş": 0.7, "ofis": 0.6, "business": 0.5, "resmi": 0.7, "düz": 0.4}
   "romantik" → {"romantik": 0.9, "sevgililer günü": 0.7, "floral": 0.6, "pastel": 0.5, "zarif": 0.7, "feminen": 0.6}
   "sokak" → {"sokak": 0.9, "street": 0.8, "urban": 0.7, "hip-hop": 0.5, "günlük": 0.6, "cool": 0.5}

3. 'exclude_tags': İstenmeyen + moda kuralına göre uyumsuz etiketler. Boşsa [].

4. 'max_price': Bütçe varsa sayı, yoksa null.

5. 'explanation': Kararını ve gerekçeni anlatan çok KISA ve ÖZ 1-2 cümle. Çok uzatma, sesli asistan okuyacağı için özet geç! Türkçe.

6. SADECE GEÇERLİ JSON DÖN. Markdown/backtick KULLANMA.
"""


def ask_grok_kombin(user_text: str, cart_context: str = "", gender: str = ""):
    """Metin tabanlı kombin analizi için Grok'a istek atar."""
    grok_api_key = os.environ.get('GROK_API_KEY')
    if not grok_api_key:
        raise Exception("GROK_API_KEY bulunamadı!")

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json"
    }

    gender_prefix = ""
    if gender:
        gender_prefix = f"[CİNSİYET: {gender.upper()}] — SADECE {gender} kıyafetleri öner!\n\n"

    user_message = gender_prefix + user_text
    if cart_context:
        user_message = f"""{gender_prefix}Kullanıcının isteği: {user_text}

Kullanıcının sepetindeki mevcut ürünler:
{cart_context}

Sepetteki ürünleri dikkate alarak eksik parçaları tamamla."""

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


def ask_grok_vision(image_base64: str, mime_type: str, user_text: str = "", cart_context: str = "", gender: str = ""):
    """Görsel tabanlı kombin analizi için Grok Vision'a istek atar."""
    grok_api_key = os.environ.get('GROK_API_KEY')
    if not grok_api_key:
        raise Exception("GROK_API_KEY bulunamadı!")

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json"
    }

    gender_prefix = ""
    if gender:
        gender_prefix = f"[CİNSİYET: {gender.upper()}] — SADECE {gender} kıyafetleri öner!\n"

    text_content = gender_prefix + "Fotoğraftaki kıyafeti analiz et ve uyumlu tamamlayıcı parçalar öner."
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


def generate_multiple_summaries(kombins: list, grok_explanation: str) -> list:
    """Birden fazla kombin için tek bir Grok isteği ile ayrı ayrı özetler üretir."""
    if not kombins:
        return []

    grok_api_key = os.environ.get('GROK_API_KEY')
    if not grok_api_key:
        return [grok_explanation] * len(kombins)

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json"
    }

    prompt_lines = [f"Genel Tarz: {grok_explanation}", ""]
    for i, komb in enumerate(kombins):
        prompt_lines.append(f"Kombin {i+1}:")
        for p in komb:
            name = p.get('name', '')
            desc = p.get('description', '')
            prompt_lines.append(f"- {name}: {desc}")
        prompt_lines.append("")

    user_message = "\n".join(prompt_lines)

    system_prompt = """Sen akıcı konuşan bir yapay zeka stil danışmanısın. 
Sana seçtiğimiz BİRDEN FAZLA kombinin (Kombin 1, Kombin 2 vb.) detayları verilecek.
Görevin: Her bir kombin için, o kombinin özelliklerini (kumaş, kesim vb.) katarak ayrı ayrı ÇOK KISA (1-2 cümlelik) ve AKICI özetler yazmak. 
Çıktıyı KESİNLİKLE aşağıdaki formatta geçerli bir JSON array olarak dön:
[
  "Senin için birinci kombin olarak şunları seçtim...",
  "İkinci bir alternatif olarak şu ürünleri bir araya getirdim...",
  "Son olarak bu parçalarla harika bir tarz yakalayabilirsin..."
]
Markdown (```json vs) veya liste kullanma, SADECE saf bir JSON array döndür."""

    data = {
        "model": "grok-4-1-fast-non-reasoning",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        text = response.json()['choices'][0]['message']['content'].strip()
        text = clean_json_response(text)
        summaries = json.loads(text)
        if isinstance(summaries, list) and len(summaries) >= len(kombins):
            return summaries[:len(kombins)]
        else:
            return [grok_explanation] * len(kombins)
    except Exception as e:
        print(f"Multiple summary error: {e}")
        return [grok_explanation] * len(kombins)


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


def calculate_scores(products: list, grok_analysis: dict, cart_ids: set = None, gender: str = ""):
    """Ürünleri Grok ağırlıklarına göre puanlar + description ek puanı + cinsiyet filtresi."""
    weights = grok_analysis.get("style_analysis", {})
    excludes = grok_analysis.get("exclude_tags", [])
    max_price = grok_analysis.get("max_price")

    scored_products = []
    excludes_lower = [ex.lower() for ex in excludes] if excludes else []
    
    # Cinsiyet filtresi hazırla
    gender_lower = gender.lower() if gender else ""
    # Erkek istendiğinde: erkek veya unisex etiketli
    # Kadın istendiğinde: kadın veya unisex etiketli
    # Boşsa: filtre yok
    allowed_gender_tags = set()
    if gender_lower in ("erkek", "erkek"):
        allowed_gender_tags = {"erkek", "unisex"}
    elif gender_lower in ("kadın", "kadin"):
        allowed_gender_tags = {"kadın", "unisex"}

    for product in products:
        product_id = product.get("id", "")
        product_price = product.get("price", 0)
        product_tags = [tag.lower() for tag in product.get("styleTags", [])]
        product_desc = product.get("description", "").lower()

        # Sepette olan ürünü önerme
        if cart_ids and product_id in cart_ids:
            continue

        # Bütçe filtresi
        if max_price is not None and product_price > max_price:
            continue

        # İstenmeyen etiket filtresi
        if any(bad_tag in product_tags for bad_tag in excludes_lower):
            continue

        # Cinsiyet filtresi: eğer allowed_gender_tags doluysa ürün bu etiketlerden birini taşımalı
        if allowed_gender_tags:
            product_gender_tags = set(product_tags) & {"erkek", "kadın", "unisex"}
            # Ürün hiç cinsiyet etiketi taşımıyorsa veya izin verilen etiket yoksa atla
            if not product_gender_tags or not (product_gender_tags & allowed_gender_tags):
                continue

        # --- StyleTags Puanlama ---
        tag_score = 0.0
        matched_tags = []
        for ai_tag, weight in weights.items():
            ai_tag_lower = ai_tag.lower()
            if ai_tag_lower in product_tags:
                tag_score += float(weight)
                matched_tags.append(ai_tag)
            else:
                for pt in product_tags:
                    if ai_tag_lower in pt or pt in ai_tag_lower:
                        tag_score += float(weight) * 0.6
                        matched_tags.append(f"{ai_tag}~{pt}")
                        break

        # --- Description Ek Puanlama (1-10) ---
        desc_score = 0.0
        desc_matches = 0
        if product_desc:
            for ai_tag, weight in weights.items():
                ai_tag_lower = ai_tag.lower()
                if ai_tag_lower in product_desc:
                    desc_matches += 1

        if len(weights) > 0 and desc_matches > 0:
            desc_ratio = desc_matches / len(weights)
            desc_score = round(desc_ratio * 10, 1)  # 1-10 arası
            desc_score = min(desc_score, 10.0)

        # Toplam skor = tag puanı + description ek puanı (ağırlıklı)
        total_score = tag_score + (desc_score * 0.3)

        if total_score > 0:
            product["ai_match_score"] = round(total_score, 2)
            product["tag_score"] = round(tag_score, 2)
            product["desc_score"] = round(desc_score, 1)
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
            clean_product = {k: v for k, v in product.items() if not k.startswith("_")}
            kombin.append(clean_product)

        if len(used_categories) >= len(categories):
            break

    return kombin


def pick_multiple_kombins(scored_products: list, categories: list, max_kombins: int = 3):
    """
    Her kategori için ayrı ayrı en iyi ürünleri alır ve bunları aynı sıradakilerle eşleştirerek
    (1.ler birbiriyle, 2.ler birbiriyle) birbirinden bağımsız kombinler oluşturur.
    Çaprazlama (itertools.product) YAPMAZ, böylece 3 farklı kombinde aynı tişört tekrar etmez,
    tamamen farklı görünümler sunulur.
    """
    # Kategori başına top N ürün listesi oluştur
    by_cat = {cat: [] for cat in categories}
    for product in scored_products:  # zaten puana göre sıralı
        cat = product.get("_source_category", product.get("category", ""))
        if cat in by_cat and len(by_cat[cat]) < max_kombins:
            clean = {k: v for k, v in product.items() if not k.startswith("_")}
            by_cat[cat].append(clean)

    kombins = []
    for i in range(max_kombins):
        kombin = []
        for cat in categories:
            cat_products = by_cat.get(cat, [])
            if cat_products:
                # İlgili sıradaki ürün varsa onu al, yoksa eldeki en sonuncuyu tekrar et (fallback)
                idx = min(i, len(cat_products) - 1)
                kombin.append(cat_products[idx])
        if kombin:
            kombins.append(kombin)

    # Tamamen aynı kombinleri filtrele
    seen_keys = set()
    unique_kombins = []
    for komb in kombins:
        key = "|".join(sorted(p.get("id", p.get("name", "")) for p in komb))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_kombins.append(komb)
            if len(unique_kombins) >= max_kombins:
                break

    return unique_kombins


# ==========================================
# POPÜLERLIK HESAPLAMA
# ==========================================
UPPER_CATEGORIES = {"tisort", "elbise"}
LOWER_CATEGORIES = {"pantolon", "ayakkabi"}

def calculate_popularity(kombin: list):
    """Tüm kaydedilmiş kombinlere bakarak benzerlik yüzdesi hesaplar."""
    if not db or not kombin:
        return {"percentage": 0, "message": ""}

    try:
        # Tüm kullanıcıların kayıtlı kombinlerini çek
        saved_docs = db.collection("saved_outfits").stream()
        all_saved = []
        for doc in saved_docs:
            data = doc.to_dict()
            products = data.get("products", [])
            if products:
                all_saved.append(products)

        if not all_saved:
            return {"percentage": 0, "message": ""}

        # Yeni kombindeki ürün ID'leri ve kategorileri
        new_ids = set()
        new_names = set()
        for p in kombin:
            if p.get("id"):
                new_ids.add(p["id"])
            if p.get("name"):
                new_names.add(p["name"].lower())

        similar_count = 0
        total_similarity_score = 0.0

        for saved_products in all_saved:
            outfit_similarity = 0.0
            has_match = False

            for sp in saved_products:
                sp_id = sp.get("id", sp.get("uid", ""))
                sp_name = sp.get("name", "").lower()
                sp_cat = sp.get("category", "").lower()

                matched = sp_id in new_ids or sp_name in new_names
                if not matched:
                    continue

                has_match = True
                # Üst giyim → yüksek katsayı, alt giyim → düşük katsayı
                if sp_cat in UPPER_CATEGORIES:
                    outfit_similarity += 0.4
                elif sp_cat in LOWER_CATEGORIES:
                    outfit_similarity += 0.2
                else:
                    outfit_similarity += 0.3

            if has_match:
                similar_count += 1
                total_similarity_score += min(outfit_similarity, 1.0)

        if similar_count == 0:
            return {"percentage": 0, "message": ""}

        avg_similarity = total_similarity_score / similar_count
        percentage = round((similar_count / len(all_saved)) * 100 * avg_similarity)
        percentage = min(percentage, 100)

        # Yüzdeye göre zenginleştirilmiş mesaj
        if percentage <= 15:
            message = f"🌟 Nadir bir kombin keşfettiniz! Bu kombinasyon diğer kullanıcıların yalnızca %{percentage}'inde görülüyor. Tarzınız gerçekten özgün!"
        elif percentage <= 50:
            message = f"✨ Kullanıcıların %{percentage}'i bu kombinleri beğendi. Modadan anlıyorsunuz!"
        else:
            message = f"🔥 Bu kombinasyon çok popüler! Kullanıcıların %{percentage}'i bu tarz kombinleri seviyor. Herkes bu trendi takip ediyor!"

        return {"percentage": percentage, "message": message}

    except Exception as e:
        print(f"Popülerlik hesaplama hatası: {e}")
        return {"percentage": 0, "message": ""}


# ==========================================
# DEPO ENDPOINTS
# ==========================================

# ── Helper: Write a log to Firestore ──
def _write_log(log_type: str, message: str, details: dict = None):
    """Firestore logs koleksiyonuna kayıt ekler."""
    if not db:
        return
    try:
        log_data = {
            "type": log_type,
            "message": message,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "details": details or {}
        }
        db.collection("logs").add(log_data)
    except Exception as e:
        print(f"Log yazma hatası: {e}")


# ── 1. Barkod–NFC Eşleştirme: NFC'ye barkod yaz ──
class NfcBarcodeMatchRequest(BaseModel):
    uid: str          # NFC UID
    barkod: str       # Barkod numarası

@app.post("/depo/nfc-barkod-esle")
async def nfc_barkod_esle(request: NfcBarcodeMatchRequest):
    """
    Verilen UID'li NFC etiketine barkod numarasını yazar.
    kullanim_sayisi +1, son_islem = now
    """
    if not db:
        raise HTTPException(status_code=500, detail="Veritabanı bağlantısı yok.")
    try:
        doc_ref = db.collection("products").document(request.uid)
        doc = doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"UID bulunamadı: {request.uid}")

        data = doc.to_dict()
        current_count = data.get("kullanim_sayisi", 0) or 0

        doc_ref.update({
            "barkod": request.barkod,
            "kullanim_sayisi": current_count + 1,
            "son_islem": firestore.SERVER_TIMESTAMP,
        })

        _write_log(
            "barcode_match",
            f"NFC eşleştirildi: UID={request.uid} → Barkod={request.barkod}",
            {"uid": request.uid, "barkod": request.barkod}
        )

        return {"status": "ok", "message": "NFC eşleştirildi", "uid": request.uid, "barkod": request.barkod}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. NFC Bilgi Görüntüleme ──
@app.get("/depo/nfc-bilgi")
async def nfc_bilgi(uid: str):
    """
    Bir UID'ye ait NFC içeriğini + barkoddan amz_id + catalogdan ürün detaylarını döner.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Veritabanı bağlantısı yok.")
    try:
        # 1. products/{uid} belgesini çek
        prod_doc = db.collection("products").document(uid).get()
        if not prod_doc.exists:
            raise HTTPException(status_code=404, detail=f"UID bulunamadı: {uid}")

        prod_data = prod_doc.to_dict()
        barkod = prod_data.get("barkod", "0000000000000")

        # 2. Eğer barkod default (sıfır) ise sadece NFC bilgilerini döndür
        if barkod == "0000000000000" or not barkod:
            return {
                "status": "ok",
                "nfc": _serialize_firestore(prod_data),
                "barkod_detay": None,
                "urun": None,
            }

        # 3. barcodes/{barkod} belgesini çek
        barkod_doc = db.collection("barcodes").document(barkod).get()
        barkod_data = barkod_doc.to_dict() if barkod_doc.exists else {}

        amz_id = barkod_data.get("amz_id", "")
        renk = barkod_data.get("renk", "")
        size = barkod_data.get("size", "")
        stok = barkod_data.get("stok", 0)
        category = barkod_data.get("category", "")

        # 4. catalog'dan ürünü bul
        urun_data = None
        if amz_id:
            cat_col = category if category else _guess_category(amz_id)
            if cat_col:
                urun_docs = (
                    db.collection("catalog")
                    .document(cat_col)
                    .collection("products")
                    .where("amz_id", "==", amz_id)
                    .limit(1)
                    .stream()
                )
                for u in urun_docs:
                    urun_data = u.to_dict()
                    urun_data["catalog_id"] = u.id
                    break

            # Fallback: tüm kategorilerde ara
            if not urun_data:
                for cat_name in ["ayakkabi", "tisort", "pantolon", "elbise"]:
                    urun_docs = (
                        db.collection("catalog")
                        .document(cat_name)
                        .collection("products")
                        .where("amz_id", "==", amz_id)
                        .limit(1)
                        .stream()
                    )
                    for u in urun_docs:
                        urun_data = u.to_dict()
                        urun_data["catalog_id"] = u.id
                        break
                    if urun_data:
                        break

        return {
            "status": "ok",
            "nfc": _serialize_firestore(prod_data),
            "barkod_detay": {
                "amz_id": amz_id,
                "renk": renk,
                "size": size,
                "stok": stok,
                "category": category,
            },
            "urun": _serialize_firestore(urun_data) if urun_data else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. NFC Yeniden Eşleştir (tek NFC için) ──
class NfcReeMatchRequest(BaseModel):
    uid: str
    barkod: str

@app.post("/depo/nfc-yeniden-esle")
async def nfc_yeniden_esle(request: NfcReeMatchRequest):
    """Tek bir NFC için barkod günceller, kullanim_sayisi ve son_islem yeniler."""
    return await nfc_barkod_esle(request)


# ── 4. Stok Kontrolü (NFC ile) ──
@app.get("/depo/stok-nfc")
async def stok_nfc(uid: str):
    """
    NFC UID'sine göre stok bilgisi getirir.
    Ana ürün + tüm kardeş barkodların size bazlı stoğu döner.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Veritabanı bağlantısı yok.")
    try:
        # 1. UID → barkod
        prod_doc = db.collection("products").document(uid).get()
        if not prod_doc.exists:
            raise HTTPException(status_code=404, detail=f"UID bulunamadı: {uid}")
        barkod = prod_doc.to_dict().get("barkod", "")

        if not barkod or barkod == "0000000000000":
            raise HTTPException(status_code=400, detail="Bu NFC henüz bir barkoda eşleştirilmemiş.")

        return await _stok_by_barkod(barkod)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 5. Stok Kontrolü (Barkod ile) ──
@app.get("/depo/stok-barkod")
async def stok_barkod(barkod: str):
    """Barkod numarasına göre stok bilgisi getirir."""
    if not db:
        raise HTTPException(status_code=500, detail="Veritabanı bağlantısı yok.")
    try:
        return await _stok_by_barkod(barkod)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _stok_by_barkod(barkod: str):
    """Ortak stok sorgulama mantığı."""
    # 1. barcodes/{barkod}
    barkod_doc = db.collection("barcodes").document(barkod).get()
    if not barkod_doc.exists:
        raise HTTPException(status_code=404, detail=f"Barkod bulunamadı: {barkod}")

    barkod_data = barkod_doc.to_dict()
    amz_id = barkod_data.get("amz_id", "")
    category = barkod_data.get("category", "")

    # 2. catalog'dan ana ürün bilgisi
    urun_data = None
    if amz_id:
        for cat_name in [category] + ["ayakkabi", "tisort", "pantolon", "elbise"]:
            if not cat_name:
                continue
            urun_docs = (
                db.collection("catalog")
                .document(cat_name)
                .collection("products")
                .where("amz_id", "==", amz_id)
                .limit(1)
                .stream()
            )
            for u in urun_docs:
                urun_data = u.to_dict()
                urun_data["catalog_id"] = u.id
                break
            if urun_data:
                break

    # 3. Aynı amz_id'ye sahip tüm kardeş barkodları çek
    kardas_barkodlar = []
    if amz_id:
        kardas_docs = (
            db.collection("barcodes")
            .where("amz_id", "==", amz_id)
            .stream()
        )
        for k in kardas_docs:
            k_data = k.to_dict()
            kardas_barkodlar.append({
                "barkod_id": k.id,
                "size": k_data.get("size", ""),
                "renk": k_data.get("renk", ""),
                "stok": k_data.get("stok", 0),
                "category": k_data.get("category", ""),
            })

        # Boyuta göre sırala
        kardas_barkodlar.sort(key=lambda x: str(x.get("size", "")))

    return {
        "status": "ok",
        "amz_id": amz_id,
        "urun": _serialize_firestore(urun_data) if urun_data else None,
        "secili_barkod": {
            "barkod_id": barkod,
            "amz_id": amz_id,
            "renk": barkod_data.get("renk", ""),
            "size": barkod_data.get("size", ""),
            "stok": barkod_data.get("stok", 0),
            "category": barkod_data.get("category", ""),
        },
        "tum_beden_stoklari": kardas_barkodlar,
    }


# ── 6. Operasyon Logları ──
@app.get("/depo/logs")
async def get_logs(log_type: str = None, limit: int = 50):
    """Depo operasyon loglarını döner."""
    if not db:
        raise HTTPException(status_code=500, detail="Veritabanı bağlantısı yok.")
    try:
        query = db.collection("logs")
        if log_type:
            query = query.where("type", "==", log_type)
        docs = query.limit(limit).stream()

        logs = []
        for doc in docs:
            d = doc.to_dict()
            d["log_id"] = doc.id
            logs.append(_serialize_firestore(d))

        # Sort by timestamp descending (client side since we may not have index)
        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return {"status": "ok", "logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 7. NFC Kayıt (Yeni NFC ekle) ──
class NfcRegisterRequest(BaseModel):
    uid: str

@app.post("/depo/nfc-kayit")
async def nfc_kayit(request: NfcRegisterRequest):
    """
    Yeni bir NFC etiketini sisteme default değerlerle kaydeder.
    Zaten kayıtlıysa hata döner.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Veritabanı bağlantısı yok.")
    try:
        doc_ref = db.collection("products").document(request.uid)
        doc = doc_ref.get()

        if doc.exists:
            return {"status": "already_exists", "message": "Bu UID zaten kayıtlı.", "uid": request.uid}

        doc_ref.set({
            "uid": request.uid,
            "barkod": "0000000000000",
            "isPaid": False,
            "kullanim_sayisi": 0,
            "son_islem": firestore.SERVER_TIMESTAMP,
        })

        _write_log(
            "nfc_register",
            f"Yeni NFC kaydedildi: UID={request.uid}",
            {"uid": request.uid}
        )

        return {"status": "ok", "message": "NFC başarıyla kaydedildi.", "uid": request.uid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 8. Satın Alma — Stok Azalt ──
class PurchaseRequest(BaseModel):
    uid: str   # NFC UID

@app.post("/depo/satin-al")
async def satin_al(request: PurchaseRequest):
    """
    Satın alma işleminde:
    1. products/{uid} → barkod al
    2. barcodes/{barkod} → stok -= 1
    3. products/{uid} → isPaid=True
    """
    if not db:
        raise HTTPException(status_code=500, detail="Veritabanı bağlantısı yok.")
    try:
        prod_doc = db.collection("products").document(request.uid).get()
        if not prod_doc.exists:
            raise HTTPException(status_code=404, detail=f"UID bulunamadı: {request.uid}")

        barkod = prod_doc.to_dict().get("barkod", "")
        if not barkod or barkod == "0000000000000":
            raise HTTPException(status_code=400, detail="Bu NFC henüz barkoda eşleştirilmemiş.")

        barkod_ref = db.collection("barcodes").document(barkod)
        barkod_doc = barkod_ref.get()
        if not barkod_doc.exists:
            raise HTTPException(status_code=404, detail=f"Barkod bulunamadı: {barkod}")

        current_stok = barkod_doc.to_dict().get("stok", 0) or 0
        new_stok = max(0, current_stok - 1)

        batch = db.batch()
        batch.update(barkod_ref, {"stok": new_stok})
        batch.update(db.collection("products").document(request.uid), {"isPaid": True, "son_islem": firestore.SERVER_TIMESTAMP})
        batch.commit()

        _write_log(
            "purchase",
            f"Satın alma: UID={request.uid}, Barkod={barkod}, Yeni Stok={new_stok}",
            {"uid": request.uid, "barkod": barkod, "stok_oncesi": current_stok, "stok_sonrasi": new_stok}
        )

        return {"status": "ok", "barkod": barkod, "stok_kalan": new_stok}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Yardımcı: Firestore Timestamp'leri serialize et ──
def _serialize_firestore(data):
    """Firestore Timestamp gibi serialize edilemeyen tipleri string'e çevirir."""
    if data is None:
        return None
    result = {}
    for k, v in data.items():
        if hasattr(v, 'isoformat'):
            result[k] = v.isoformat()
        elif hasattr(v, 'timestamp_pb'):
            # Firestore Timestamp
            try:
                result[k] = v.strftime("%Y-%m-%d %H:%M:%S") if hasattr(v, 'strftime') else str(v)
            except:
                result[k] = str(v)
        else:
            result[k] = v
    return result


def _guess_category(amz_id: str) -> str:
    """AMZ ID'ye göre kategori tahmin et (fallback)."""
    return ""
