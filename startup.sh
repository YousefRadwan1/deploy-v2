#!/bin/bash
# Azure Web App startup script for Baby Cry Detection API

# Install system dependency for soundfile (if not already present)
apt-get install -y libsndfile1 2>/dev/null || true

# Install CPU-only PyTorch (lighter, faster install)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet

# Install all requirements
pip install -r requirements.txt --quiet

# Start the app with gunicorn
gunicorn app.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 1 \
  --bind 0.0.0.0:8000 \
  --timeout 300 \
  --log-level info
