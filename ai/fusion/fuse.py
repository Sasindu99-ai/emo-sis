"""Multimodal late-fusion engine.

Combines probability distributions and distress indicators across:
1. Facial branch (expression probability)
2. Audio branch (vocal stress emotion probability)
3. Text branch (transcribed distress keywords & sentiment)

Computes the final fused distress score and determines if an alert threshold is breached.
"""

from typing import Dict, Any, Optional

DEFAULT_DISTRESS_THRESHOLD = 0.45  # Derived from EDA cost-of-error analysis

# Modality weights (customizable based on signal reliability)
DEFAULT_WEIGHTS = {
    "facial": 0.35,
    "audio": 0.35,
    "text": 0.30
}


def compute_facial_distress(facial_probs: Dict[str, float]) -> float:
    """Computes aggregate distress score from facial emotion probabilities."""
    # Pain, panic, and agitation represent distress
    pain = facial_probs.get("pain", 0.0)
    panic = facial_probs.get("panic", 0.0)
    agitation = facial_probs.get("agitation", 0.0)
    
    # Non-linear combination: intense pain or panic dominates
    score = (pain * 1.0) + (panic * 1.0) + (agitation * 0.7)
    return float(min(1.0, max(0.0, score)))


def compute_audio_distress(audio_probs: Dict[str, float]) -> float:
    """Computes aggregate distress score from acoustic emotion probabilities."""
    # Fear, anger, and sadness represent acute vocal distress
    fear = audio_probs.get("fear", 0.0)
    sad = audio_probs.get("sad", 0.0)
    angry = audio_probs.get("angry", 0.0)
    distress = audio_probs.get("distress", 0.0)
    
    score = (fear * 1.0) + (sad * 0.8) + (angry * 0.7) + (distress * 1.0)
    return float(min(1.0, max(0.0, score)))


def fuse_signals(
    facial_probs: Optional[Dict[str, float]] = None,
    audio_probs: Optional[Dict[str, float]] = None,
    text_result: Optional[Dict[str, Any]] = None,
    threshold: float = DEFAULT_DISTRESS_THRESHOLD,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Late-fusion logic combining multi-branch signals.

    Handles missing modalities dynamically (e.g. if patient face is turned away).
    """
    w = weights or DEFAULT_WEIGHTS.copy()
    active_weights = {}
    modality_scores = {}

    if facial_probs:
        score_f = compute_facial_distress(facial_probs)
        modality_scores["facial"] = score_f
        active_weights["facial"] = w.get("facial", 0.35)

    if audio_probs:
        score_a = compute_audio_distress(audio_probs)
        modality_scores["audio"] = score_a
        active_weights["audio"] = w.get("audio", 0.35)

    if text_result:
        score_t = float(text_result.get("distress_score", 0.0))
        modality_scores["text"] = score_t
        active_weights["text"] = w.get("text", 0.30)

    if not active_weights:
        return {"distress_score": 0.0, "alert": False, "confidence": 0.0}

    # Normalize weights over available modalities
    total_w = sum(active_weights.values())
    normalized_w = {k: v / total_w for k, v in active_weights.items()}

    # Weighted sum
    fused_score = sum(modality_scores[k] * normalized_w[k] for k in active_weights)
    
    # Multi-branch synergy boost: if 2+ modalities detect distress simultaneously
    high_count = sum(1 for s in modality_scores.values() if s >= 0.5)
    if high_count >= 2:
        fused_score = min(1.0, fused_score * 1.15)

    fused_score = float(round(fused_score, 4))
    alert = fused_score >= threshold

    return {
        "distress_score": fused_score,
        "alert": alert,
        "threshold": threshold,
        "modality_scores": modality_scores
    }
