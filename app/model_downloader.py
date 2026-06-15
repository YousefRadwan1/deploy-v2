"""
model_downloader.py
===================
Downloads model files from Google Drive at startup if they don't exist locally.
Uses requests with streaming to handle large files (bypass virus scan warning).
"""

import logging
import os
import requests

logger = logging.getLogger(__name__)

# ── Google Drive file IDs ─────────────────────────────────────────────────────
MODELS = {
    "stage1_cnn.pt":     "1OZ7zoX6IREg7StghrO0Hoq23RTNnDPJZ",
    "best_w2v_ecapa.pt": "19Zk2RupUZz2jtBUSggsVKwx1tTm0WzWT",
}

GDRIVE_BASE = "https://drive.google.com/uc"
CHUNK_SIZE  = 32 * 1024 * 1024   # 32 MB chunks


def _get_confirm_token(response: requests.Response) -> str | None:
    """Extract virus-scan confirmation token for large files."""
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _download_file(file_id: str, dest_path: str) -> None:
    """Stream-download a file from Google Drive with virus-scan bypass."""
    session  = requests.Session()
    params   = {"id": file_id, "export": "download"}
    response = session.get(GDRIVE_BASE, params=params, stream=True, timeout=60)
    response.raise_for_status()

    token = _get_confirm_token(response)
    if token:
        params["confirm"] = token
        response = session.get(GDRIVE_BASE, params=params, stream=True, timeout=60)
        response.raise_for_status()

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                logger.info(f"  ↓ {os.path.basename(dest_path)}: {downloaded / 1e6:.1f} MB...")

    logger.info(f"  ✓ {os.path.basename(dest_path)} saved ({downloaded / 1e6:.1f} MB)")


def download_models(models_dir: str = "/home/models") -> None:
    """
    Check each model — download from Google Drive if missing.
    Called once at API startup.
    """
    os.makedirs(models_dir, exist_ok=True)

    for filename, file_id in MODELS.items():
        dest = os.path.join(models_dir, filename)

        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            logger.info(f"Model already exists, skipping: {dest}")
            continue

        logger.info(f"Downloading {filename} from Google Drive...")
        try:
            _download_file(file_id, dest)
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise RuntimeError(
                f"Could not download '{filename}' from Google Drive. "
                f"Error: {e}"
            ) from e
