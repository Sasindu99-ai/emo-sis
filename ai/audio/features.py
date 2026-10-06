"""Audio feature extraction and preprocessing routines for speech emotion recognition.

Includes:
- Spectral gating noise reduction (noisereduce)
- Mel spectrogram extraction (for CNN branch input)
- Statistical summary features (MFCC, pitch/F0, RMS energy, ZCR) for baseline models
"""

from typing import Optional, Tuple
import numpy as np
import librosa

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False


DEFAULT_SR = 22050
DEFAULT_DURATION = 3.0  # seconds (matches windowed pipeline spec)
DEFAULT_N_MELS = 128
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_FFT = 2048


def clean_audio(y: np.ndarray, sr: int = DEFAULT_SR, prop_decrease: float = 0.8) -> np.ndarray:
    """Applies spectral gating noise reduction to an audio signal."""
    if not HAS_NOISEREDUCE:
        return y
    try:
        # Stationary noise reduction
        reduced = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease, stationary=True)
        return reduced
    except Exception:
        # Fallback if noise reduce fails on edge-case audio
        return y


def load_audio(
    audio_path: str,
    target_sr: int = DEFAULT_SR,
    duration: Optional[float] = DEFAULT_DURATION,
    clean: bool = True
) -> Tuple[np.ndarray, int]:
    """Loads an audio file, resamples, pads/trims to target duration, and cleans noise."""
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    if clean:
        y = clean_audio(y, sr=sr)
        
    if duration is not None:
        target_len = int(target_sr * duration)
        if len(y) < target_len:
            # Pad with zeros (or reflect)
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        else:
            # Truncate to fixed duration
            y = y[:target_len]
            
    return y, sr


def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH
) -> np.ndarray:
    """Computes a log-mel spectrogram (dB scale) normalized for CNN model input.
    
    Returns:
        np.ndarray with shape (n_mels, time_steps)
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize between 0 and 1 or z-score
    norm_mel = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return norm_mel


def extract_pitch_f0(y: np.ndarray, sr: int = DEFAULT_SR) -> Tuple[float, float]:
    """Estimates fundamental frequency (F0) mean and std using librosa pyin."""
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        valid_f0 = f0[voiced_flag & ~np.isnan(f0)] if voiced_flag is not None else np.array([])
        if len(valid_f0) > 0:
            return float(np.mean(valid_f0)), float(np.std(valid_f0))
    except Exception:
        pass
    return 0.0, 0.0


def extract_feature_vector(y: np.ndarray, sr: int = DEFAULT_SR) -> np.ndarray:
    """Extracts a 1D statistical feature vector (MFCCs, Chroma, Mel, Pitch, RMS, ZCR)
    for baseline tabular classifiers (KNN, SVM) and Mahalanobis distance analysis.
    """
    # 1. MFCC (mean & std across 20 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # 2. RMS Energy
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    # 3. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # 4. Spectral Centroid & Rolloff
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    cent_mean, cent_std = np.mean(cent), np.std(cent)
    rolloff_mean, rolloff_std = np.mean(rolloff), np.std(rolloff)
    
    # 5. Pitch
    pitch_mean, pitch_std = extract_pitch_f0(y, sr)
    
    features = np.hstack([
        mfcc_mean,
        mfcc_std,
        [rms_mean, rms_std, zcr_mean, zcr_std, cent_mean, cent_std, rolloff_mean, rolloff_std, pitch_mean, pitch_std]
    ])
    return features.astype(np.float32)
