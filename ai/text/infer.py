"""Text branch inference module.

Transcribes audio and computes linguistic distress scores.
Exposes:
    predict(audio_path: str) -> dict
"""

import sys
import json
from typing import Dict, Any

from ai.text.transcribe import transcribe
from ai.text.distress_score import score_text_distress


def predict(audio_path: str) -> Dict[str, Any]:
    """Transcribes audio and returns distress score.
    
    Returns:
        {
            "transcript": str,
            "distress_score": float,
            "keywords": list[str]
        }
    """
    text = transcribe(audio_path)
    score, keywords = score_text_distress(text)

    return {
        "transcript": text,
        "distress_score": score,
        "keywords": keywords
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ai.text.infer <sample.wav>")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(json.dumps(result, indent=2))
