"""
pipeline.py
===========
Orchestrates the full cry-detection pipeline (100% PyTorch):
"""

import logging
import os
from collections import Counter

import numpy as np
import torch

from app.audio_utils import (
    load_and_resample,
    energy_vad,
    segment_audio,
    extract_mfcc,
    preprocess_for_w2v2,
    TARGET_SR,
)

logger = logging.getLogger(__name__)

CLASSES_STAGE2 = ["scared", "needs", "physical_pain", "burping"]


class CryDetectionPipeline:
    def __init__(self, stage1_model_path: str, stage2_model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.stage1_loaded = False
        self.stage2_loaded = False
        self.stage1_model  = None
        self.stage2_model  = None
        self.aam_head      = None

        self._load_stage1(stage1_model_path)
        self._load_stage2(stage2_model_path)

    # ── Model loaders ─────────────────────────────────────────────────────────
    def _load_stage1(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"Stage 1 model not found at {path}. Stage 1 will be skipped.")
            return
        try:
            from app.stage1_model import Stage1CNN
            self.stage1_model = Stage1CNN().to(self.device)
            self.stage1_model.load_state_dict(torch.load(path, map_location=self.device))
            self.stage1_model.eval()
            self.stage1_loaded = True
            logger.info(f"Stage 1 (PyTorch CNN) loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load Stage 1 model: {e}")

    def _load_stage2(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"Stage 2 model not found at {path}. Stage 2 will be skipped.")
            return
        try:
            from app.stage2_model import build_stage2_model
            model, aam_head = build_stage2_model(self.device)

            ckpt = torch.load(path, map_location=self.device)
            model_state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
            aam_state   = ckpt.get("aam_state_dict",   ckpt.get("aam_head", None))

            def strip_dp(state_dict):
                return {
                    (k[7:] if k.startswith("module.") else k): v
                    for k, v in state_dict.items()
                }

            model.load_state_dict(strip_dp(model_state), strict=False)
            if aam_state is not None:
                aam_head.load_state_dict(strip_dp(aam_state), strict=False)

            model.eval()
            aam_head.eval()
            self.stage2_model = model
            self.aam_head     = aam_head
            self.stage2_loaded = True
            logger.info(f"Stage 2 (Wav2Vec2+ECAPA) loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load Stage 2 model: {e}")

    # ── Inference helpers ─────────────────────────────────────────────────────
    @torch.no_grad()
    def _run_stage1(self, wav_chunk: np.ndarray) -> tuple[str, float]:
        """Returns ('cry'|'not_cry', confidence)."""
        mfcc = extract_mfcc(wav_chunk)                               # (1, 16, T)
        x = torch.from_numpy(mfcc).unsqueeze(0).to(self.device)      # (1, 1, 16, T)
        
        prob = self.stage1_model(x).item()
        label = "cry" if prob >= 0.5 else "not_cry"
        conf  = prob if prob >= 0.5 else 1.0 - prob
        return label, round(conf, 4)

    @torch.no_grad()
    def _run_stage2(self, wav_chunk: np.ndarray) -> tuple[str, float]:
        """Returns (cry_type_label, confidence)."""
        x      = preprocess_for_w2v2(wav_chunk, self.device)  # (1, T)
        emb    = self.stage2_model(x)                         # (1, 512)
        logits = self.aam_head.get_logits(emb).squeeze(0)      # (4,)
        probs  = torch.softmax(logits, dim=0)
        conf, idx = probs.max(0)
        return CLASSES_STAGE2[idx.item()], round(conf.item(), 4)

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self, audio_path: str, filename: str = "") -> dict:
        wav = load_and_resample(audio_path)
        duration_sec = round(len(wav) / TARGET_SR, 2)
        segments = segment_audio(wav)
        total_segments = len(segments)

        segment_results = []
        for i, chunk in enumerate(segments):
            seg_start = round(i * 4.0, 2)
            seg_end   = round(seg_start + 4.0, 2)

            speech_regions = energy_vad(chunk, TARGET_SR)
            if not speech_regions:
                segment_results.append({
                    "segment_index": i,
                    "start_sec":     seg_start,
                    "end_sec":       seg_end,
                    "vad_result":    "silence",
                    "stage1_result": None,
                    "stage2_result": None,
                })
                continue

            seg_info = {
                "segment_index": i,
                "start_sec":     seg_start,
                "end_sec":       seg_end,
                "vad_result":    "sound_detected",
                "stage1_result": None,
                "stage2_result": None,
            }

            if self.stage1_loaded:
                s1_label, s1_conf = self._run_stage1(chunk)
                seg_info["stage1_result"] = {
                    "label":      s1_label,
                    "confidence": s1_conf,
                }

                if s1_label == "cry" and self.stage2_loaded:
                    s2_label, s2_conf = self._run_stage2(chunk)
                    seg_info["stage2_result"] = {
                        "cry_type":   s2_label,
                        "confidence": s2_conf,
                    }
            else:
                if self.stage2_loaded:
                    s2_label, s2_conf = self._run_stage2(chunk)
                    seg_info["stage2_result"] = {
                        "cry_type":   s2_label,
                        "confidence": s2_conf,
                    }

            segment_results.append(seg_info)

        return self._summarize(segment_results, duration_sec, total_segments, filename)

    # ── Majority-vote summary ─────────────────────────────────────────────────
    def _summarize(self, segment_results: list[dict], duration_sec: float, total_segments: int, filename: str) -> dict:
        silence_segs   = [s for s in segment_results if s["vad_result"] == "silence"]
        sound_segs     = [s for s in segment_results if s["vad_result"] == "sound_detected"]
        cry_segs       = [s for s in sound_segs if s.get("stage1_result", {}) and s["stage1_result"]["label"] == "cry"]
        not_cry_segs   = [s for s in sound_segs if s.get("stage1_result", {}) and s["stage1_result"]["label"] == "not_cry"]
        stage2_segs    = [s for s in cry_segs if s.get("stage2_result") is not None]

        is_cry = len(cry_segs) > 0

        stage1_label, stage1_confidence = None, None
        if sound_segs and any(s["stage1_result"] for s in sound_segs):
            labels = [s["stage1_result"]["label"] for s in sound_segs if s["stage1_result"]]
            confs  = [s["stage1_result"]["confidence"] for s in sound_segs if s["stage1_result"]]
            stage1_label = Counter(labels).most_common(1)[0][0]
            stage1_confidence = round(sum(c for l, c in zip(labels, confs) if l == stage1_label) / max(labels.count(stage1_label), 1), 4)

        cry_type, cry_type_confidence = None, None
        if stage2_segs:
            types = [s["stage2_result"]["cry_type"] for s in stage2_segs]
            confs = [s["stage2_result"]["confidence"] for s in stage2_segs]
            cry_type = Counter(types).most_common(1)[0][0]
            cry_type_confidence = round(sum(c for t, c in zip(types, confs) if t == cry_type) / max(types.count(cry_type), 1), 4)

        return {
            "filename":            filename,
            "duration_sec":        duration_sec,
            "total_segments":      total_segments,
            "silence_segments":    len(silence_segs),
            "sound_segments":      len(sound_segs),
            "cry_segments":        len(cry_segs),
            "not_cry_segments":    len(not_cry_segs),
            "is_cry":              is_cry,
            "stage1": {"verdict": stage1_label, "confidence": stage1_confidence},
            "stage2": {"cry_type": cry_type, "confidence": cry_type_confidence},
            "segment_details": segment_results,
        }