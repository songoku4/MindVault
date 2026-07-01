import os
import json
import subprocess
import numpy as np
import librosa
from faster_whisper import WhisperModel
from transformers import pipeline


FFMPEG_PATH = r"C:\Users\Aaditya Sharma\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")
_model = None

def get_whisper():
    global _model
    if _model is None:
        print(f"[WHISPER] Loading model: {WHISPER_MODEL_SIZE}")
        _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        print("[WHISPER] Model ready")
    return _model

def transcribe(audio_path: str) -> dict:
    model = get_whisper()
    segments, info = model.transcribe(audio_path, beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    return {
        "transcript": text,
        "language": info.language,
        "duration": info.duration
    }

def convert_to_wav(audio_path: str) -> str:
    wav_path = audio_path + "_converted.wav"
    result = subprocess.run([
        FFMPEG_PATH,
        "-y", "-i", audio_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        wav_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ACOUSTIC] ffmpeg stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"ffmpeg conversion failed with code {result.returncode}")

    print(f"[ACOUSTIC] Converted to WAV: {wav_path}")
    return wav_path

def extract_acoustic_features(audio_path: str) -> dict:
    print(f"[ACOUSTIC] Extracting features from {audio_path}")

    wav_path = convert_to_wav(audio_path)

    try:
        y, sr = librosa.load(wav_path, sr=16000)
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    f0_voiced = f0[voiced_flag] if voiced_flag is not None and voiced_flag.any() else np.array([0.0])
    pitch_mean = float(np.nanmean(f0_voiced))
    pitch_std  = float(np.nanstd(f0_voiced))

    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    speech_rate = float(np.mean(zcr))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1).tolist()

    energy_norm   = min(energy_mean * 100, 1.0)
    pitch_norm    = min(pitch_mean / 300, 1.0)
    acoustic_mood = round((energy_norm * 0.6 + pitch_norm * 0.4), 4)

    print(f"[ACOUSTIC] pitch={pitch_mean:.1f}Hz energy={energy_mean:.4f} mood={acoustic_mood:.3f}")

    return {
        "pitch_mean":    pitch_mean,
        "pitch_std":     pitch_std,
        "energy_mean":   energy_mean,
        "speech_rate":   speech_rate,
        "mfcc_mean":     json.dumps([round(x, 4) for x in mfcc_means]),
        "acoustic_mood": acoustic_mood
    }

from transformers import pipeline

_sentiment_model = None

def get_sentiment():
    global _sentiment_model
    if _sentiment_model is None:
        print("[SENTIMENT] Loading model...")
        _sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # CPU
        )
        print("[SENTIMENT] Model ready")
    return _sentiment_model

def analyse_sentiment(text: str) -> dict:
    if not text or len(text.strip()) < 3:
        return {"label": "neutral", "score": 0.5}
    
    model = get_sentiment()
    result = model(text[:512])[0]  # truncate to model max
    
    label = result["label"].lower()  # POSITIVE or NEGATIVE
    raw_score = result["score"]
    
    # Convert to 0-1 scale where 0.5 = neutral
    if label == "positive":
        score = 0.5 + (raw_score - 0.5) * 0.5
    else:
        score = 0.5 - (raw_score - 0.5) * 0.5

    print(f"[SENTIMENT] {label} ({raw_score:.3f}) → score={score:.3f}")
    return {"label": label, "score": round(score, 4)}