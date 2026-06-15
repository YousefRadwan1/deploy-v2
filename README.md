# 🍼 Baby Cry Detection API

Two-stage infant cry detection and classification — deployed on **Azure App Service (Code Deploy)**.

> Models are downloaded automatically from Google Drive on first startup into `/home/models` (Azure persistent storage).

---

## 🗂️ Project Structure

```
baby_cry_api/
├── app/
│   ├── __init__.py
│   ├── main.py               ← FastAPI app
│   ├── model_downloader.py   ← Auto-downloads models from Google Drive
│   ├── pipeline.py           ← VAD → Stage 1 → Stage 2
│   ├── audio_utils.py        ← Audio loading, VAD, MFCC
│   ├── stage1_model.py       ← PyTorch CNN architecture
│   ├── stage2_model.py       ← Wav2Vec2+ECAPA architecture
│   └── schemas.py            ← Pydantic response models
├── startup.sh                ← Azure startup command
├── requirements.txt
└── .env.example
```

---

## 🚀 Azure App Service Setup

### 1. Create App Service
- Runtime: **Python 3.11**
- OS: **Linux**

### 2. Set Startup Command
Go to: **Azure Portal → App Service → Configuration → General Settings**

Set **Startup Command** to:
```
startup.sh
```

### 3. Set Application Settings
Go to: **Azure Portal → App Service → Configuration → Application Settings**

Add these:
```
MODELS_DIR            = /home/models
STAGE1_MODEL_PATH     = /home/models/stage1_cnn.pt
STAGE2_MODEL_PATH     = /home/models/best_w2v_ecapa.pt
SCM_DO_BUILD_DURING_DEPLOYMENT = true
```

### 4. Deploy the code
Via GitHub Actions, Azure CLI, or VS Code Azure extension.

### 5. Check it's running
```
GET https://<your-app>.azurewebsites.net/health
```

Expected:
```json
{"status": "ok", "stage1_loaded": true, "stage2_loaded": true}
```

---

## 📋 Startup Flow

```
1. startup.sh runs
2. pip install requirements
3. gunicorn starts → triggers lifespan
4. model_downloader checks /home/models/
   ├── First run  → downloads from Google Drive (~few minutes)
   └── Next runs  → files already exist → skip download instantly
5. Models loaded into memory
6. API ready ✅
```

> `/home/` on Azure App Service is **persistent storage** — survives restarts and redeployments.

---

## 🧪 Endpoints

### `GET /health`
```json
{"status": "ok", "stage1_loaded": true, "stage2_loaded": true}
```

### `POST /predict`
- Body: `form-data`
- Key: `file` (type: File)
- Supported: `.wav` `.mp3` `.flac` `.ogg` `.opus` `.m4a` `.aac`

**Response:**
```json
{
  "filename": "baby.opus",
  "duration_sec": 18.4,
  "is_cry": true,
  "stage1": {"verdict": "cry", "confidence": 0.87},
  "stage2": {"cry_type": "needs", "confidence": 0.74},
  "processing_time_sec": 2.35
}
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `stage1_loaded: false` | Check logs in Azure → Log Stream |
| Models re-downloading on every restart | Normal only on first deploy; `/home/` is persistent after that |
| 503 on `/health` | App still starting up (model download in progress) — wait a few minutes |
