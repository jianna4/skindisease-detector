from fastapi import FastAPI, File, UploadFile
from PIL import Image
import requests
import io
import numpy as np

app = FastAPI(title="Skin Disease Detection API")

HF_API_URL = "https://api-inference.huggingface.co/models/jianna4/skin-disease-cnn"
HF_TOKEN = ""

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Preprocess image: resize, convert to RGB, etc.
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # match your model input
    # Convert to bytes (Hugging Face API can accept raw bytes)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read uploaded file
    image_bytes = await file.read()
    
    # Preprocess the image
    processed_bytes = preprocess_image(image_bytes)
    
    # Send to Hugging Face API
    response = requests.post(HF_API_URL, headers=headers, data=processed_bytes)
    
    if response.status_code != 200:
        return {"error": f"Inference API returned status {response.status_code}", "details": response.text}
    
    prediction = response.json()
    return {"filename": file.filename, "prediction": prediction}
