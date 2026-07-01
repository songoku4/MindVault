# MindVault

**A passive longitudinal mental wellness monitoring system using multimodal acoustic and semantic analysis.**

*Built by Aaditya Sharma — Senior DevOps and MLOps Engineer, Master of IT (AI) student at the University of Melbourne.*

---

I built MindVault because I wanted to explore what happens when you stop asking people how they feel and start *listening* to how they sound.

Most mood tracking apps rely on self-reported scores: tap a number, pick an emoji, done. The problem is that humans are notoriously bad at accurately reporting their own emotional state, especially under stress. MindVault takes a different approach: you speak freely for 60 seconds, and the system analyses *both* what you said and how you said it. Pitch, energy, speech rate, MFCCs from the acoustic layer. Sentiment, keyword salience, and emotional tone from the text layer. Both signals are fused into a composite mood score that gets tracked over time.

The thing I find genuinely interesting and what I think makes this worth demoing is the divergence signal. When your acoustic mood is high (energetic, elevated pitch) but your text sentiment is low (negative words, anxious content), the system flags that as an emotionally complex session. That's clinically meaningful. It's the difference between someone who *sounds* fine and someone who *is* fine.

---

## What it does

- **Voice check-in** — record 30-60 seconds of free speech daily
- **Acoustic analysis** — extract pitch (Hz), energy (RMS), speech rate, MFCCs via librosa
- **Transcription** — Whisper (local, no data leaves your machine)
- **Sentiment analysis** — DistilBERT fine-tuned on SST-2 via HuggingFace
- **Composite mood scoring** — acoustic and semantic signals fused into a single 0-1 score
- **Trend dashboard** — 14-day mood chart, stat cards, insight engine, check-in history
- **Divergence detection** — flags when voice energy and word sentiment disagree
- **MLflow experiment tracking** — every check-in logged as a run with full metrics
- **Prometheus + Grafana monitoring** — real-time observability on processing time, mood distributions, pitch
- **Research export** — structured JSON/CSV data suitable for longitudinal studies
- **Local-first** — all audio processing on-device, optional cloud sync

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser UI                           │
│         (DM Serif + warm cream palette, Chart.js)           │
└────────────────────────┬────────────────────────────────────┘
                         │ audio/webm
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI backend                         │
│                                                             │
│  ┌─────────────────┐    ┌──────────────────────────────┐   │
│  │  Whisper (tiny) │    │  librosa acoustic extraction  │   │
│  │  transcription  │    │  pitch · energy · MFCC · ZCR  │   │
│  └────────┬────────┘    └──────────────┬───────────────┘   │
│           │                            │                    │
│           ▼                            ▼                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         DistilBERT sentiment analysis               │   │
│  │         (HuggingFace, SST-2 fine-tuned)             │   │
│  └──────────────────────────┬──────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Composite mood fusion (0-1 score)           │   │
│  │         acoustic 50% + sentiment 50%                │   │
│  └──────┬───────────────────┬───────────────────┬──────┘   │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│      SQLite            MLflow run          Prometheus       │
│      (local)           (metrics)           (/metrics)       │
└─────────────────────────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          Grafana               MLflow UI
          :3000                  :5001
```

---

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| Transcription | faster-whisper (tiny) | Local speech-to-text, no API needed |
| Acoustic analysis | librosa | Pitch, energy, MFCCs, ZCR extraction |
| Sentiment | HuggingFace DistilBERT | Text emotion classification |
| Backend | FastAPI | REST API, /checkin, /history, /health, /metrics |
| Database | SQLite (local) / PostgreSQL (cloud) | Check-in persistence |
| Experiment tracking | MLflow | Every check-in logged as a run with full metrics |
| Monitoring | Prometheus + Grafana | Real-time processing time, mood distributions, pitch gauge |
| Containerisation | Docker + docker-compose | Single command deployment |
| Orchestration | Kubernetes (minikube) | 2-replica deployment with readiness probes |
| CI/CD | GitHub Actions | Test, lint, Docker build on every push |
| Audio conversion | ffmpeg | WebM to WAV conversion |

---

## The acoustic + semantic fusion: why it matters

Most sentiment tools only read text. But speech carries emotional information that words alone don't capture.

```
Your words: "I'm fine, everything is okay."
Your voice: low energy, flat pitch, slow speech rate

→ Composite mood: 42% (flagged as emotionally complex)
```

MindVault extracts these features from every recording:

- **Pitch (Hz)** — fundamental frequency via pyin algorithm. Elevated pitch often correlates with arousal and positive affect.
- **Energy (RMS)** — root mean square amplitude. Low energy is a consistent acoustic marker of low mood and fatigue.
- **MFCCs** — 13 mel-frequency cepstral coefficients capturing vocal tract shape. Used in clinical depression detection research.
- **Zero crossing rate** — proxy for speech rate and articulatory precision.

These are then fused with DistilBERT's sentiment score (0-1 scale, positive/negative classification with confidence) into a single composite mood score, logged to both SQLite and MLflow.

---

## Getting started

### Prerequisites

- Python 3.11+
- Docker Desktop
- ffmpeg (`winget install ffmpeg` on Windows, `brew install ffmpeg` on Mac)

### Run locally

```bash
git clone https://github.com/songoku4/MindVault.git
cd MindVault

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1

pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000`

### Run with Docker (full stack)

```bash
docker-compose up -d
```

This starts four services:

| Service | URL | Description |
|---|---|---|
| MindVault app | http://localhost:8000 | Main application |
| MLflow | http://localhost:5001 | Experiment tracking |
| Prometheus | http://localhost:9090 | Metrics scraping |
| Grafana | http://localhost:3000 | Dashboards (admin / mindvault123) |

### Deploy to Kubernetes

```bash
minikube start --driver=docker
minikube image build -t mindvault-app:latest .
kubectl apply -f infra/k8s/
minikube service mindvault-service --url
```

---

## Project structure

```
mindvault/
├── app/
│   ├── main.py           # FastAPI app, Prometheus metrics
│   ├── audio.py          # Whisper transcription, librosa features, DistilBERT sentiment
│   ├── database.py       # SQLAlchemy models, SQLite
│   └── templates/
│       └── index.html    # Dashboard UI
├── infra/
│   └── k8s/              # Kubernetes manifests
├── monitoring/
│   └── prometheus.yml    # Scrape config
├── tests/
│   └── test_api.py       # Pytest unit tests
├── Dockerfile
├── docker-compose.yml
└── .github/
    └── workflows/
        └── ci.yml        # GitHub Actions CI pipeline
```

---

## Metrics tracked per check-in

Every check-in is logged as an MLflow experiment run and exposed via Prometheus:

| Metric | Description |
|---|---|
| `composite_mood` | Fused acoustic + semantic score (0-1) |
| `acoustic_mood` | Voice-only mood signal (0-1) |
| `sentiment_score` | DistilBERT confidence score (0-1) |
| `pitch_hz` | Fundamental frequency in Hz |
| `energy` | RMS energy of the audio signal |
| `duration_sec` | Recording length |
| `processing_sec` | End-to-end processing time |

---

## Research and clinical potential

MindVault was designed from the start to be research-ready:

**For researchers:** configurable check-in prompts, structured JSON/CSV export, participant-level data isolation, IRB-friendly local-first data handling. The acoustic + semantic fusion approach is directly applicable to longitudinal mood studies.

**For clinicians:** weekly trend reports, between-session passive monitoring, divergence alerts when acoustic and semantic signals disagree significantly.

**The hypothesis this enables testing:** acoustic features alone can predict self-reported mood deterioration 24-48 hours before the individual consciously recognises it.

If you're a researcher at a mental health or psychology department interested in piloting MindVault with study participants, I'd genuinely love to talk. Reach out at aaditya4aug@gmail.com.

---

## Related work from my portfolio

- **[RAG-Ops](https://github.com/songoku4/RagOps)** — Production-grade RAG pipeline with LangChain, ChromaDB, FastAPI, Kubernetes, MLflow, and Prometheus. Built with the same MLOps principles applied here.

---

## Author

**Aaditya Sharma**
Senior DevOps and MLOps Engineer
Master of IT (AI), University of Melbourne

[Portfolio](https://songoku4.github.io) · [LinkedIn](https://www.linkedin.com/in/aaditya-sharma-88a54b160/) · [GitHub](https://github.com/songoku4)

---

*MindVault runs entirely locally. No audio, transcript, or personal data is ever sent to an external server unless you explicitly configure cloud sync. Your voice stays yours.*
