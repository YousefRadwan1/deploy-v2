"""
schemas.py
==========
Pydantic models for API request/response validation and OpenAPI docs.
"""

from typing import Optional
from pydantic import BaseModel


class Stage1Result(BaseModel):
    label:      Optional[str]   = None   # "cry" | "not_cry"
    confidence: Optional[float] = None


class Stage2Result(BaseModel):
    cry_type:   Optional[str]   = None   # "scared" | "needs" | "physical_pain" | "burping"
    confidence: Optional[float] = None


class Stage1Summary(BaseModel):
    verdict:    Optional[str]   = None
    confidence: Optional[float] = None


class Stage2Summary(BaseModel):
    cry_type:   Optional[str]   = None
    confidence: Optional[float] = None


class SegmentDetail(BaseModel):
    segment_index: int
    start_sec:     float
    end_sec:       float
    vad_result:    str                     # "silence" | "sound_detected"
    stage1_result: Optional[Stage1Result] = None
    stage2_result: Optional[Stage2Result] = None


class PredictionResponse(BaseModel):
    filename:           str
    duration_sec:       float
    total_segments:     int
    silence_segments:   int
    sound_segments:     int
    cry_segments:       int
    not_cry_segments:   int
    is_cry:             bool
    stage1:             Stage1Summary
    stage2:             Stage2Summary
    segment_details:    list[SegmentDetail]
    processing_time_sec: Optional[float] = None
