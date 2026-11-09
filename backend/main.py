from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import onnxruntime as ort
from transformers import AutoTokenizer
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_DIR = os.path.join(BASE_DIR, "ml", "models", "xlmr-final")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models", "onnx")

tokenizer = None
ort_session = None
labels = ["Negative", "Neutral", "Positive"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, ort_session
    try:
        print(f"Loading Tokenizer from: {TOKENIZER_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        
        onnx_model_path = os.path.join(MODEL_DIR, "model.onnx")
        print(f"Loading ONNX Model from: {onnx_model_path}")
        
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"Model file not found at {onnx_model_path}")
            
        ort_session = ort.InferenceSession(onnx_model_path)
        print("Model successfully loaded into memory!")
    except Exception as e:
        print(f"Error loading model: {e}")
        
    yield
    print("Shutting down API...")

app = FastAPI(
    title="UrduSense Sentiment API",
    description="Roman Urdu Sentiment Analysis using fine-tuned XLM-RoBERTa & ONNX",
    version="1.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Development ke liye sab domains allow hain (Next.js bhi)
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, OPTIONS sab allow
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": ort_session is not None}

@app.post("/api/v1/predict")
def predict_sentiment(request: PredictionRequest):
    if not ort_session or not tokenizer:
        raise HTTPException(status_code=500, detail="Model is not loaded yet.")
    
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    try:
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=128)
        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        
        outputs = ort_session.run(None, onnx_inputs)
        logits = outputs[0]
        
        import numpy as np
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        pred_idx = int(np.argmax(probs[0]))
        confidence = float(probs[0][pred_idx])
        predicted_label = labels[pred_idx]
        
        return {
            "text": text,
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                labels[i]: round(float(probs[0][i]), 4) for i in range(len(labels))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))