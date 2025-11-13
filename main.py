from fastapi import FastAPI, File, UploadFile
from PIL import Image
import requests
import io
import numpy as np
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Skin Disease Detection API")

 #Add CORS middleware - UPDATE THIS SECTION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://127.0.0.1:3000",  # Alternative localhost
        "http://localhost:5500",  # Live Server extension
        "http://127.0.0.1:5500",  # Alternative Live Server
        "file://",  # For local HTML files
        "*"  # Allow all origins (use only in development!)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Hugging Face token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("Hugging Face token not found in environment variable HF_TOKEN")

HF_API_URL = "https://router.huggingface.co/hf-inference/models/jianna4/skin-disease-cnn"


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
