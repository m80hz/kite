"""Failure locating prompt and parsing helpers."""
from typing import List, Dict, Any
import json

FAILURE_LOCATING_INSTRUCTION = (
    "You are given key frames from a task execution plus structured context. "
    "Identify up to 3 candidate failure frames (by its frame number 0, 1, 2, ...) where failure likely occurred. "
    "Confidence in [0,1]. "
    "Return STRICT JSON only: "
    "{\"candidates\":[{\"frame_num\": int, \"confidence\": float}]}. "
)

def build_failure_locating_prompt() -> str:
    return FAILURE_LOCATING_INSTRUCTION

def parse_failure_candidates(raw: str) -> List[Dict[str,Any]]:
    """Extract candidate list from model raw output that should be JSON.
    Tolerates fenced code blocks.
    """
    txt = raw.strip()
    if '```json' in txt:
        try:
            txt = txt.split('```json',1)[1].split('```',1)[0]
        except Exception:
            pass
    try:
        data = json.loads(txt)
    except Exception:
        return []
    cands = data.get('candidates', []) if isinstance(data, dict) else []
    out = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        if 'frame_num' in c and 'confidence' in c:
            out.append({
                'frame_num': int(c.get('frame_num', 0)),
                'confidence': float(c.get('confidence', 0.0)),
            })
    out.sort(key=lambda x: x['confidence'], reverse=True)
    return out[:3]  # top 3