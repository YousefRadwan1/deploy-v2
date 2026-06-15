"""
Baby Cry Detection API
======================
Pipeline:
  1. VAD  — energy-based voice activity detection
  2. Stage 1 — PyTorch CNN (MFCC) → cry / not_cry
  3. Stage 2 — PyTorch Wav2Vec2+ECAPA → cry type

Models are downloaded automatically from Google Drive on first startup.
"""

import os
import tempfile
import time
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.model_downloader import download_models
from app.pipeline import CryDetectionPipeline
from app.schemas import PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

pipeline: CryDetectionPipeline = None

SUPPORTED_EXTS = {".wav", ".mp3", ".ogg", ".opus", ".m4a", ".flac", ".aac"}

# Azure Web App persistent storage is at /home
MODELS_DIR        = os.environ.get("MODELS_DIR", "/home/models")
STAGE1_MODEL_PATH = os.environ.get("STAGE1_MODEL_PATH", os.path.join(MODELS_DIR, "stage1_cnn.pt"))
STAGE2_MODEL_PATH = os.environ.get("STAGE2_MODEL_PATH", os.path.join(MODELS_DIR, "best_w2v_ecapa.pt"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    # Step 1: Download models from Google Drive if not present
    logger.info("Checking model files...")
    download_models(models_dir=MODELS_DIR)

    # Step 2: Load models into memory
    logger.info("Loading models into memory...")
    pipeline = CryDetectionPipeline(
        stage1_model_path=STAGE1_MODEL_PATH,
        stage2_model_path=STAGE2_MODEL_PATH,
    )
    logger.info("Models loaded. API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Baby Cry Detection API",
    description="Two-stage infant cry detection and classification pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "stage1_loaded": pipeline.stage1_loaded if pipeline else False,
        "stage2_loaded": pipeline.stage2_loaded if pipeline else False,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTS)}",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        t0 = time.time()
        result = pipeline.run(tmp_path, filename=file.filename)
        result["processing_time_sec"] = round(time.time() - t0, 3)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
