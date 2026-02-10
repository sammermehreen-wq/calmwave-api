# CalmWave Emotion Detection API

CalmWave is a FastAPI-based emotion detection API powered by a fine-tuned **DeBERTa** model.
It predicts emotions from text input and is designed to be used by mobile and web applications.

---

## 🚀 Features

- FastAPI backend
- DeBERTa-based emotion classification
- Local model loading (no external API calls)
- Ready for deployment (Render / Hugging Face)
- Android-friendly REST API

---

## 📁 Project Structure
calmwave-api/
├── main.py
├── requirements.txt
├── deberta_emotion_model/
│ ├── config.json
│ ├── model.safetensors
│ ├── tokenizer.json
│ ├── tokenizer_config.json
│ └── special_tokens_map.json


---

## 🧪 Run Locally

### 1️⃣ Install Python
Make sure Python **3.9+** is installed and added to PATH.

Check:
```bash
python --version
