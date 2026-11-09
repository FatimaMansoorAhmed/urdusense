# 🇵🇰 UrduSense

**Sentiment + toxicity analysis for Urdu and Roman Urdu, with token-level explanations.**

UrduSense is an end-to-end NLP app that takes Roman Urdu text, classifies its **sentiment** and **toxicity**, and highlights *which words* drove the prediction, so results are explainable rather than a black box.
---

## ✨ Features

- **Sentiment classification**: Positive / Neutral / Negative
- **Toxicity detection**: flags abusive or offensive content
- **Token-level explanations**: word highlights showing each token's contribution
- **Dual-script support**: Urdu script and Roman Urdu (code-mixed text with English too)
- **REST API**: FastAPI with auto-generated Swagger docs
- **Clean web UI**: Next.js frontend with live highlighting

## 🧠 How it works

```
Input text → Normalization → XLM-R tokenizer → Fine-tuned XLM-R
          → [sentiment head] + [toxicity head] → Attribution (token scores) → JSON → UI
```

| Component | Details |
|---|---|
| Base model | `xlm-roberta-base` (fine-tuned) |
| Tasks | Sentiment (3-class), Toxicity (binary) |
| Explainability | `<method: e.g. Integrated Gradients / SHAP / attention>` |
| Preprocessing | Roman Urdu normalization (spelling variants, e.g. *acha / achha / accha*) |


## 🛠️ Tech Stack

- **ML:** PyTorch, Hugging Face Transformers, Scikit-learn
- **Backend:** FastAPI, Uvicorn, Pydantic
- **Frontend:** Next.js, React, Tailwind CSS
- **Deployment:** `<Docker / Vercel / Render / HF Spaces>`

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Clone
```bash
git clone https://github.com/<username>/urdusense.git
cd urdusense
```

### 2. Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API runs at `http://localhost:8000` · Docs at `http://localhost:8000/docs`

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```
App runs at `http://localhost:3000`

> Model weights: `<download link / auto-downloaded from Hugging Face Hub: username/urdusense>`

## 📡 API Usage

**`POST /analyze`**

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "yeh movie bohat kamaal ki thi"}'
```

```json
{
  "text": "yeh movie bohat kamaal ki thi",
  "sentiment": { "label": "positive", "confidence": 0.94 },
  "toxicity": { "label": "non-toxic", "confidence": 0.98 },
  "tokens": [
    { "token": "yeh", "score": 0.02 },
    { "token": "movie", "score": 0.05 },
    { "token": "bohat", "score": 0.31 },
    { "token": "kamaal", "score": 0.72 }
  ]
}
```

## 📁 Project Structure

```
urdusense/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI entrypoint
│   │   ├── model.py         # Model loading + inference
│   │   ├── explain.py       # Token attribution
│   │   └── schemas.py       # Pydantic models
│   └── requirements.txt
├── frontend/                # Next.js app
├── notebooks/               # Training + evaluation
├── docs/
└── README.md
```

## 🏋️ Training

```bash
cd notebooks
# 1. Data prep:   01_data_preparation.ipynb
# 2. Fine-tuning: 02_finetune_xlmr.ipynb
# 3. Evaluation:  03_evaluation.ipynb
```

Hyperparameters: `lr=<>`, `batch_size=<>`, `epochs=<>`, `max_len=<>`

## ⚠️ Limitations

- Roman Urdu has no standard spelling, so uncommon variants may be misclassified
- Sarcasm and heavy code-mixing remain difficult
- Toxicity labels reflect the training dataset's annotation biases
- Not intended for automated moderation without human review

## 🗺️ Roadmap

- [ ] Add emotion classification
- [ ] Batch / CSV upload
- [ ] Browser extension
- [ ] Larger Roman Urdu training corpus

