"""
audio_utils.py
==============
Pure-Python audio loading — NO ffmpeg binary required.

Format support:
  WAV, FLAC, OGG, MP3       → soundfile  (libsndfile bundled in pip package)
  OPUS, AAC, M4A, WEBM      → PyAV       (bundles its own codecs via pip)
  Everything else            → PyAV fallback

Resampling: scipy.signal.resample (high quality, pure Python)
"""

import io
import logging
from typing import Union

import av
import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample as scipy_resample

logger = logging.getLogger(__name__)

TARGET_SR       = 16_000
SEGMENT_SEC     = 4
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_SEC

# MFCC config — must match Stage 1 training exactly
N_MFCC     = 16
N_FFT      = 800
HOP_LENGTH = 400

# Formats soundfile handles natively (no codec needed)
_SOUNDFILE_EXTS = {
    ".wav", ".wave",
    ".flac",
    ".ogg",
    ".mp3",
    ".aiff", ".aif",
    ".au",
    ".caf",
    ".rf64",
}

# Formats that need PyAV (bundled codecs, no system ffmpeg)
_PYAV_EXTS = {
    ".opus",
    ".aac",
    ".m4a",
    ".mp4",
    ".webm",
    ".wma",
    ".3gp",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality resampling via scipy (no ffmpeg, no binary deps)."""
    if orig_sr == target_sr:
        return audio
    new_length = int(len(audio) * target_sr / orig_sr)
    return scipy_resample(audio, new_length).astype(np.float32)


def _load_via_soundfile(source: Union[str, io.BytesIO]) -> tuple[np.ndarray, int]:
    """Load with soundfile — handles WAV/FLAC/OGG/MP3/AIFF natively."""
    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)   # stereo → mono
    else:
        audio = audio[:, 0]
    return audio.astype(np.float32), int(sr)


def _load_via_pyav(path: str) -> tuple[np.ndarray, int]:
    """
    Load with PyAV — handles OPUS, AAC, M4A, WEBM and anything else.
    PyAV ships its own codecs inside the wheel; no system ffmpeg needed.
    """
    container = av.open(path)
    stream    = container.streams.audio[0]

    # Force mono float32 reformat inside PyAV
    stream.codec_context.request_in_format = "fltp"

    frames = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray()          # (channels, samples)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)        # mix to mono
        frames.append(arr.astype(np.float32))

    container.close()

    if not frames:
        raise ValueError("PyAV decoded zero audio frames.")

    audio = np.concatenate(frames)
    sr    = int(stream.codec_context.sample_rate)
    return audio, sr


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_resample(path: str) -> np.ndarray:
    """
    Load any audio file → 16 kHz mono float32 numpy array, peak-normalised.

    Strategy (no ffmpeg binary needed at any step):
      1. Detect format by extension
      2. Try soundfile for common formats (WAV/FLAC/OGG/MP3)
      3. Fall back to PyAV for OPUS/AAC/M4A/WEBM
      4. Final fallback: librosa (uses soundfile internally)

    Returns float32 array normalised to [-1, 1].
    """
    import os
    ext = os.path.splitext(path)[-1].lower()

    audio, sr = None, None

    # ── Try soundfile first for its native formats ────────────────────────────
    if ext in _SOUNDFILE_EXTS:
        try:
            audio, sr = _load_via_soundfile(path)
            logger.debug(f"soundfile loaded {os.path.basename(path)}: sr={sr}, samples={len(audio)}")
        except Exception as e:
            logger.warning(f"soundfile failed ({e}), trying PyAV...")

    # ── PyAV for opus / aac / m4a / webm — or as fallback ────────────────────
    if audio is None:
        try:
            audio, sr = _load_via_pyav(path)
            logger.debug(f"PyAV loaded {os.path.basename(path)}: sr={sr}, samples={len(audio)}")
        except Exception as e:
            logger.warning(f"PyAV failed ({e}), trying librosa...")

    # ── librosa as last resort ────────────────────────────────────────────────
    if audio is None:
        audio, sr = librosa.load(path, sr=None, mono=True)
        audio = audio.astype(np.float32)
        logger.debug(f"librosa loaded {os.path.basename(path)}: sr={sr}")

    # ── Resample to 16 kHz ────────────────────────────────────────────────────
    if sr != TARGET_SR:
        audio = _resample(audio, sr, TARGET_SR)

    # ── Peak normalise ────────────────────────────────────────────────────────
    peak = np.abs(audio).max()
    if peak > 1e-8:
        audio = audio / peak

    return audio.astype(np.float32)


def load_audio_bytes(data: bytes, ext: str = ".wav") -> np.ndarray:
    """
    Load audio from raw bytes (e.g. from an in-memory upload buffer).
    `ext` hints the format; defaults to WAV.
    Useful if you want to avoid writing a temp file entirely.
    """
    buf = io.BytesIO(data)

    # soundfile can read from BytesIO for most formats
    try:
        audio, sr = _load_via_soundfile(buf)
    except Exception:
        # PyAV needs a real file path — write to a temp file
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            audio, sr = _load_via_pyav(tmp_path)
        finally:
            os.unlink(tmp_path)

    if sr != TARGET_SR:
        audio = _resample(audio, sr, TARGET_SR)

    peak = np.abs(audio).max()
    if peak > 1e-8:
        audio = audio / peak

    return audio.astype(np.float32)


# ── VAD ───────────────────────────────────────────────────────────────────────

def energy_vad(
    wav:                  np.ndarray,
    sr:                   int   = TARGET_SR,
    frame_ms:             int   = 30,
    hop_ms:               int   = 10,
    energy_threshold_db:  float = -38.0,
    min_speech_ms:        int   = 300,
    context_ms:           int   = 100,
) -> list[tuple[int, int]]:
    """
    Energy-based Voice Activity Detection.
    Returns a list of (start_sample, end_sample) tuples for speech regions.
    Returns an empty list if the chunk is silent.
    """
    frame_len  = int(sr * frame_ms  / 1000)
    hop_len    = int(sr * hop_ms    / 1000)
    min_frames = max(1, int(min_speech_ms / hop_ms))
    ctx_frames = max(1, int(context_ms   / hop_ms))
    n_frames   = (len(wav) - frame_len) // hop_len + 1

    if n_frames <= 0:
        return []

    energy    = np.array([
        np.sqrt(np.mean(wav[i * hop_len: i * hop_len + frame_len] ** 2) + 1e-12)
        for i in range(n_frames)
    ])
    is_speech = 20 * np.log10(energy) > energy_threshold_db

    # Context smoothing — fill short gaps between speech frames
    for i in range(ctx_frames, len(is_speech) - ctx_frames):
        if is_speech[max(0, i - ctx_frames): i + ctx_frames].any():
            is_speech[i] = True

    segments, in_seg, seg_start = [], False, 0
    for i, v in enumerate(is_speech):
        if v and not in_seg:
            in_seg, seg_start = True, i
        elif not v and in_seg:
            in_seg = False
            if i - seg_start >= min_frames:
                segments.append((
                    seg_start * hop_len,
                    min(i * hop_len + frame_len, len(wav)),
                ))
    if in_seg and len(is_speech) - seg_start >= min_frames:
        segments.append((seg_start * hop_len, len(wav)))

    return segments


# ── Segmentation ──────────────────────────────────────────────────────────────

def segment_audio(
    wav:             np.ndarray,
    segment_samples: int = SEGMENT_SAMPLES,
) -> list[np.ndarray]:
    """
    Split a waveform into fixed-length chunks.
    The last chunk is zero-padded if shorter than segment_samples.
    """
    chunks, start = [], 0
    while start < len(wav):
        chunk = wav[start: start + segment_samples]
        if len(chunk) < segment_samples:
            chunk = np.pad(chunk, (0, segment_samples - len(chunk)))
        chunks.append(chunk.astype(np.float32))
        start += segment_samples
    return chunks


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_mfcc(wav_chunk: np.ndarray) -> np.ndarray:
    """
    MFCC extraction matching Stage 1 training exactly.
    Input  : float32 array, length = SEGMENT_SAMPLES (4 sec @ 16 kHz)
    Output : ndarray shape (N_MFCC, time_frames, 1) — channel-last for TF Conv2D
    """
    audio = wav_chunk / (np.abs(wav_chunk).max() + 1e-6)
    if len(audio) > SEGMENT_SAMPLES:
        audio = audio[:SEGMENT_SAMPLES]
    elif len(audio) < SEGMENT_SAMPLES:
        audio = np.pad(audio, (0, SEGMENT_SAMPLES - len(audio)))

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    return np.expand_dims(mfcc, axis=0)  # (1, 16, T)


def preprocess_for_w2v2(wav_chunk: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Prepare a 4-second waveform chunk for the Wav2Vec2+ECAPA model.
    Applies mean-std normalisation (same as training inference cell).
    Returns tensor shape (1, T) on `device`.
    """
    peak = np.abs(wav_chunk).max()
    if peak > 1e-8:
        wav_chunk = wav_chunk / peak

    x = torch.from_numpy(wav_chunk.astype(np.float32)).unsqueeze(0).to(device)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x
