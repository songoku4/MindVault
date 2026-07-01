import os
import time
import tempfile
import mlflow
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

from app.database import init_db, get_db, CheckIn
from app.audio import transcribe, extract_acoustic_features, analyse_sentiment

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

CHECKINS_TOTAL = Counter("mindvault_checkins_total", "Total check-ins recorded")
MOOD_SCORE = Histogram("mindvault_mood_score", "Composite mood score distribution", buckets=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
ACOUSTIC_SCORE = Histogram("mindvault_acoustic_score", "Acoustic mood score distribution", buckets=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
PITCH_GAUGE = Gauge("mindvault_pitch_hz", "Most recent pitch measurement in Hz")
PROCESSING_TIME = Histogram("mindvault_processing_seconds", "Time to process a check-in", buckets=[5,10,15,20,30,45,60])

app = FastAPI(title="MindVault")
Instrumentator().instrument(app).expose(app)
os.makedirs("recordings", exist_ok=True)

def setup_mlflow():
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("mindvault")
        print(f"[MLFLOW] Connected to {MLFLOW_URI}")
    except Exception as e:
        print(f"[MLFLOW] Setup failed (offline mode): {e}")

@app.on_event("startup")
def startup():
    init_db()
    setup_mlflow()
    print("[MINDVAULT] Ready")

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
        raise HTTPException(400, "Audio file required")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(contents)
        tmp_path = f.name

    start_time = time.time()

    try:
        print(f"[CHECKIN] Processing for user={user_id}")
        transcription = transcribe(tmp_path)
        print(f"[CHECKIN] Transcript: {transcription['transcript'][:80]}...")
        acoustic = extract_acoustic_features(tmp_path)
        sentiment = analyse_sentiment(transcription["transcript"])
        composite_mood = round(acoustic["acoustic_mood"] * 0.5 + sentiment["score"] * 0.5, 4)

        entry = CheckIn(
            user_id=user_id,
            timestamp=datetime.utcnow(),
            transcript=transcription["transcript"],
            duration_sec=transcription["duration"],
            notes=notes,
            sentiment_label=sentiment["label"],
            sentiment_score=sentiment["score"],
            composite_mood=composite_mood,
            **{k: acoustic[k] for k in ["pitch_mean","pitch_std","energy_mean","speech_rate","mfcc_mean","acoustic_mood"]}
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)

        CHECKINS_TOTAL.inc()
        MOOD_SCORE.observe(composite_mood)
        ACOUSTIC_SCORE.observe(acoustic["acoustic_mood"])
        PITCH_GAUGE.set(acoustic["pitch_mean"])
        PROCESSING_TIME.observe(time.time() - start_time)

        try:
            with mlflow.start_run(run_name=f"checkin_{user_id}"):
                mlflow.log_param("user_id", user_id)
                mlflow.log_param("sentiment_label", sentiment["label"])
                mlflow.log_metric("composite_mood", composite_mood)
                mlflow.log_metric("acoustic_mood", acoustic["acoustic_mood"])
                mlflow.log_metric("sentiment_score", sentiment["score"])
                mlflow.log_metric("pitch_hz", acoustic["pitch_mean"])
                mlflow.log_metric("energy", acoustic["energy_mean"])
                mlflow.log_metric("duration_sec", transcription["duration"])
                mlflow.log_metric("processing_sec", round(time.time() - start_time, 2))
        except Exception as e:
            print(f"[MLFLOW] Logging skipped: {e}")

        print(f"[CHECKIN] Saved id={entry.id} mood={composite_mood} in {round(time.time()-start_time)}s")

        return {
            "id": entry.id,
            "transcript": transcription["transcript"],
            "duration_sec": transcription["duration"],
            "sentiment_label": sentiment["label"],
            "sentiment_score": sentiment["score"],
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