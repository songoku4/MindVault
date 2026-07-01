from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from datetime import datetime
import os, tempfile, json

from app.database import init_db, get_db, CheckIn
from app.audio import transcribe, extract_acoustic_features

app = FastAPI(title="MindVault — Mental Wellness Monitor")

os.makedirs("recordings", exist_ok=True)

@app.on_event("startup")
def startup():
    init_db()
    print("[MINDVAULT] Database ready")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("app/templates/index.html") as f:
        return f.read()

@app.post("/checkin")
async def checkin(
    file: UploadFile = File(...),
    user_id: str = "default",
    notes: str = "",
    db: Session = Depends(get_db)
):
    if not file.filename.endswith((".wav", ".mp3", ".m4a", ".webm", ".ogg")):
        raise HTTPException(400, "Audio file required (.wav, .mp3, .m4a, .webm, .ogg)")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(contents)
        tmp_path = f.name

    try:
        print(f"[CHECKIN] Processing audio for user={user_id}")

        # Transcribe
        transcription = transcribe(tmp_path)
        print(f"[CHECKIN] Transcript: {transcription['transcript'][:80]}...")

        # Acoustic features
        acoustic = extract_acoustic_features(tmp_path)

        # Simple text sentiment (we'll upgrade this on Day 5)
        transcript_lower = transcription["transcript"].lower()
        positive_words = ["good","great","happy","well","better","fine","excited","calm","peaceful","motivated"]
        negative_words = ["bad","tired","sad","anxious","stressed","worried","awful","terrible","exhausted","low"]
        pos = sum(1 for w in positive_words if w in transcript_lower)
        neg = sum(1 for w in negative_words if w in transcript_lower)
        sentiment_score = round((pos - neg) / max(pos + neg, 1) * 0.5 + 0.5, 4)
        sentiment_label = "positive" if sentiment_score > 0.55 else "negative" if sentiment_score < 0.45 else "neutral"

        # Composite mood (fuse acoustic + text)
        composite_mood = round(acoustic["acoustic_mood"] * 0.5 + sentiment_score * 0.5, 4)

        # Save to database
        entry = CheckIn(
            user_id=user_id,
            timestamp=datetime.utcnow(),
            transcript=transcription["transcript"],
            duration_sec=transcription["duration"],
            notes=notes,
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            composite_mood=composite_mood,
            **{k: acoustic[k] for k in ["pitch_mean","pitch_std","energy_mean","speech_rate","mfcc_mean","acoustic_mood"]}
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)

        print(f"[CHECKIN] Saved entry id={entry.id} mood={composite_mood}")

        return {
            "id": entry.id,
            "transcript": transcription["transcript"],
            "duration_sec": transcription["duration"],
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "acoustic_mood": acoustic["acoustic_mood"],
            "composite_mood": composite_mood,
            "pitch_hz": round(acoustic["pitch_mean"], 1),
            "energy": round(acoustic["energy_mean"], 5),
        }
    finally:
        os.unlink(tmp_path)

@app.get("/history")
def history(user_id: str = "default", limit: int = 30, db: Session = Depends(get_db)):
    entries = (
        db.query(CheckIn)
        .filter(CheckIn.user_id == user_id)
        .order_by(CheckIn.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": e.id,
            "timestamp": e.timestamp.isoformat(),
            "transcript": e.transcript,
            "sentiment_label": e.sentiment_label,
            "composite_mood": e.composite_mood,
            "acoustic_mood": e.acoustic_mood,
            "pitch_hz": round(e.pitch_mean or 0, 1),
        }
        for e in entries
    ]

@app.get("/health")
def health():
    return {"status": "ok"}