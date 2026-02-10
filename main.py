from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import os

app = FastAPI()

# ===== LOAD MODEL AND TOKENIZER (Global scope to load once) =====
MODEL_PATH = "./deberta_emotion_model"

try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Load model configuration (important for id2label mapping)
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # Load model weights
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config=config)
    model.eval() # Set model to evaluation mode
    
    print("Model and tokenizer loaded successfully for FastAPI!")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None
    config = None

# ===== EMOTION MAPPING (for simplified output) =====
GOEMOTION_TO_APP_MAP = {
    "anger": "Anger", "annoyance": "Anger",
    "fear": "Fear", "nervousness": "Fear",
    "joy": "Joy", "love": "Joy", "optimism": "Joy",
    "sadness": "Sadness", "grief": "Sadness", "disappointment": "Sadness", "remorse": "Sadness",
    "calm": "Calm", "relief": "Calm",
    "surprise": "Surprise",
    "admiration": "Positive", "amusement": "Positive", "approval": "Positive", "caring": "Positive",
    "excitement": "Positive", "gratitude": "Positive", "pride": "Positive",
    "desire": "Mixed", "realization": "Mixed", "confusion": "Mixed", "curiosity": "Mixed", "embarrassment": "Mixed",
    "disapproval": "Negative", "disgust": "Negative",
    "neutral": "Neutral"
}

# Define a threshold for considering a prediction strong enough
CONFIDENCE_THRESHOLD = 0.45

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "FastAPI is running with emotion prediction model!"}

@app.post("/predict")
def predict_emotion(data: TextInput):
    if tokenizer is None or model is None:
        return {"error": "Model not loaded. Check server logs for details."}, 500

    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Apply softmax for single-label interpretation, or sigmoid for multi-label
    # Given the previous context and desired output, softmax for highest prob seems more suitable for a single mood
    probs = F.softmax(outputs.logits, dim=-1)[0] # [0] to get probabilities for the single input

    max_score = 0.0
    predicted_emotion_label = "Neutral"
    
    # Find the highest confidence emotion from the model's output
    top_prob, top_idx = torch.max(probs, dim=0)
    max_score = top_prob.item()
    raw_label = config.id2label[top_idx.item()]
    predicted_emotion_label = GOEMOTION_TO_APP_MAP.get(raw_label, "Neutral")

    # Apply smart neutral logic: if confidence is below threshold, default to Neutral
    if max_score < CONFIDENCE_THRESHOLD:
        predicted_emotion_label = "Neutral"

    return {
        "emotion": predicted_emotion_label,
        "confidence": round(max_score, 4),
        "raw_label": raw_label
    }
