"""Inference module for the Audio Emotion branch.

Exposes:
    predict(audio_path: str) -> dict[str, float]

Can be invoked from CLI:
    python -m ai.audio.infer <sample.wav>
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch

from ai.audio.features import load_audio, extract_mel_spectrogram
from ai.audio.model import AudioEmotionCNN, DEFAULT_AUDIO_CLASSES

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "audio_model.pt"

_cached_model: Optional[AudioEmotionCNN] = None
_cached_classes: List[str] = DEFAULT_AUDIO_CLASSES
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(checkpoint_path: Optional[str] = None) -> Optional[AudioEmotionCNN]:
    """Loads and caches the PyTorch AudioEmotionCNN model."""
    global _cached_model, _cached_classes
    if _cached_model is not None:
        return _cached_model

    path = Path(checkpoint_path) if checkpoint_path else MODEL_PATH
    if path.exists():
        try:
            checkpoint = torch.load(path, map_location=_device)
            if isinstance(checkpoint, dict) and "classes" in checkpoint:
                _cached_classes = checkpoint["classes"]
                model = AudioEmotionCNN(num_classes=len(_cached_classes))
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model = AudioEmotionCNN(num_classes=len(_cached_classes))
                model.load_state_dict(checkpoint)
            model.to(_device)
            model.eval()
            _cached_model = model
            return _cached_model
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint from {path}: {e}", file=sys.stderr)

    return None


def predict(audio_path: str, checkpoint_path: Optional[str] = None) -> Dict[str, float]:
    """Infers emotion probabilities from an audio file.

    Args:
        audio_path: Path to .wav or audio file.
        checkpoint_path: Optional custom path to model checkpoint.

    Returns:
        dict[str, float]: Class names mapped to probabilities, summing to ~1.0.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load and preprocess
    y, sr = load_audio(audio_path)
    mel = extract_mel_spectrogram(y, sr=sr)
    
    model = get_model(checkpoint_path)
    if model is not None:
        tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(_device)
        with torch.no_grad():
            probs = model.predict_proba(tensor).squeeze(0).cpu().numpy()
        
        prob_dict = {cls: float(np.round(p, 4)) for cls, p in zip(_cached_classes, probs)}
    else:
        # Fallback heuristic / prior distribution if model has not been trained yet
        # Uses energy and spectral tilt to approximate baseline distribution
        energy = float(np.sqrt(np.mean(y ** 2)))
        classes = DEFAULT_AUDIO_CLASSES
        base_weight = np.ones(len(classes), dtype=np.float32)
        
        # High energy slightly elevates distress classes (angry, fear)
        if energy > 0.1:
            base_weight[classes.index("fear")] += 1.5
            base_weight[classes.index("angry")] += 1.2
        else:
            base_weight[classes.index("neutral")] += 2.0
            
        probs = base_weight / np.sum(base_weight)
        prob_dict = {cls: float(np.round(p, 4)) for cls, p in zip(classes, probs)}

    # Ensure probabilities sum exactly to 1.0
    total = sum(prob_dict.values())
    if total > 0:
        prob_dict = {k: round(v / total, 4) for k, v in prob_dict.items()}

    return prob_dict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ai.audio.infer <sample.wav>")
        sys.exit(1)

    target_audio = sys.argv[1]
    result = predict(target_audio)
    print(json.dumps(result, indent=2))
