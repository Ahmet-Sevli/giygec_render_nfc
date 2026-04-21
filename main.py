from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os, json, firebase_admin, uuid, shutil, base64, tempfile
from firebase_admin import credentials, firestore
from gradio_client import Client, file

app = FastAPI()

# 1. Firebase Kurulumu
firebase_key_raw = os.environ.get('FIREBASE_KEY')
db = None

if firebase_key_raw:
    try:
        cred = credentials.Certificate(json.loads(firebase_key_raw))
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase bağlantısı başarılı!")
    except Exception as e:
        print(f"Firebase başlatılamadı: {e}")

# 2. Leyleğin (Cron-job) Vuracağı Kapı
@app.get("/")
async def root():
    return {"status": "ok", "message": "GİYGEÇ Sistemi Ayakta!"}

# 3. NodeMCU Ödeme Sorgu Kapısı
@app.get("/check-payment")
async def check_payment(uid: str):
    if not uid:
        return {"status": "error", "message": "UID eksik"}
    
    if not db:
        return {"status": "error", "message": "Veritabanı bağlantısı yok"}

    try:
        doc_ref = db.collection('products').document(uid)
        doc = doc_ref.get()

        if doc.exists:
            is_paid = doc.to_dict().get('isPaid', False)
            return {"paid": is_paid}
        return {"paid": False, "message": "Urun bulunamadi"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 4. Hugging Face IDM-VTON Kıyafet Giydirme Kapısı
@app.post("/virtual-try-on")
def virtual_try_on(
    person_image: UploadFile = File(...), 
    garment_image: UploadFile = File(...)
):
    temp_dir = tempfile.gettempdir()
    temp_id = str(uuid.uuid4())
    
    person_path = os.path.join(temp_dir, f"person_{temp_id}.jpg")
    garment_path = os.path.join(temp_dir, f"garment_{temp_id}.jpg")

    try:
        with open(person_path, "wb") as buffer:
            shutil.copyfileobj(person_image.file, buffer)
        with open(garment_path, "wb") as buffer:
            shutil.copyfileobj(garment_image.file, buffer)

        # --- GÜNCELLENEN KISIM: VIP (Token) Girişi ---
        hf_token = os.environ.get('HF_TOKEN')
        client = Client("yisol/IDM-VTON", hf_token=hf_token)
        # ---------------------------------------------

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

        result_image_path = result[0]

        with open(result_image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        return {
            "status": "success", 
            "message": "Kıyafet başarıyla giydirildi!",
            "image_base64": encoded_string
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    finally:
        if os.path.exists(person_path): os.remove(person_path)
        if os.path.exists(garment_path): os.remove(garment_path)
