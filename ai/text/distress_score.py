"""Text distress and urgency scoring module.

Analyzes transcribed patient speech using emergency keyword matching,
lexical distress intensity, repetition weighting, and multilingual support
(English, Sinhala, Tamil hospital contexts).
"""

import re
from typing import Dict, List, Tuple

# Weighted clinical distress lexicon
DISTRESS_KEYWORDS: Dict[str, float] = {
    # High urgency / acute pain & panic (0.8 - 1.0)
    "help": 0.85,
    "help me": 0.95,
    "hurts": 0.85,
    "pain": 0.85,
    "agony": 0.95,
    "suffocating": 0.95,
    "cannot breathe": 0.95,
    "can't breathe": 0.95,
    "choking": 0.95,
    "dying": 0.90,
    "emergency": 0.90,
    "nurse": 0.70,
    "doctor": 0.70,
    "please": 0.60,
    "scared": 0.75,
    "terrified": 0.85,
    "fall": 0.75,
    "fallen": 0.80,
    "bleeding": 0.85,

    # Sinhala (phonetic / romanized & unicode)
    "udaw": 0.90,
    "udawwa": 0.90,
    "ridenawa": 0.85,
    "amaru": 0.80,
    "amaruwen": 0.85,
    "baya": 0.75,
    "bayai": 0.80,
    "husma": 0.90,
    "pana": 0.85,
    "උදව්": 0.90,
    "රිදෙනවා": 0.85,
    "අමාරුයි": 0.85,
    "හුස්ම": 0.90,

    # Tamil (phonetic / romanized & unicode)
    "uthavi": 0.90,
    "valikkuthu": 0.85,
    "vali": 0.80,
    "bayam": 0.80,
    "bayama": 0.80,
    "moochu": 0.90,
    "உதவி": 0.90,
    "வலி": 0.85,
    "பயம்": 0.80
}


def score_text_distress(text: str) -> Tuple[float, List[str]]:
    """Calculates distress score (0.0 to 1.0) and lists matched keywords.
    
    Args:
        text: Transcribed speech.
        
    Returns:
        (score: float, matched_keywords: List[str])
    """
    if not text or not text.strip():
        return 0.0, []

    text_lower = text.lower()
    matches = []
    max_weight = 0.0
    accumulated_weight = 0.0

    for phrase, weight in sorted(DISTRESS_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r"\b" + re.escape(phrase) + r"\b" if phrase.isascii() and phrase.isalnum() else re.escape(phrase)
        found = re.findall(pattern, text_lower)
        if found:
            count = len(found)
            matches.append(phrase)
            max_weight = max(max_weight, weight)
            # Repeated cries increase urgency
            accumulated_weight += weight * min(count, 3) * 0.3

    if not matches:
        return 0.0, []

    # Non-linear fusion: strong base on peak distress word + repetition boost
    combined = max_weight + (1.0 - max_weight) * (1.0 - (1.0 / (1.0 + accumulated_weight)))
    final_score = float(min(1.0, round(combined, 4)))

    return final_score, matches
