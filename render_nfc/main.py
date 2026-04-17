from fastapi import FastAPI
import os, json, firebase_admin
from firebase_admin import credentials, firestore

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