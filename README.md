# emo-sis: Multimodal Patient Distress Monitoring

A research system exploring whether a multimodal AI pipeline can monitor hospital rooms (camera + microphone) and flag emotional/psychological distress — panic, acute pain, agitation, crying — in non-verbal, psychiatric, or otherwise unattended patients in the gaps between nurse rounds.

---

## 1. Architecture Overview

```
                ┌─────────────────────┐
  camera ──────▶│   Facial branch      │──┐
                │  (OpenCV + CNN)      │  │
                └─────────────────────┘  │
                                          │   probability vectors
                ┌─────────────────────┐  │   (per-branch, per-window)
  mic ─────────▶│   Audio branch       │──┼──────────────┐
                │  (noisereduce        │  │              │
                │   + CNN)             │  │              ▼
                └─────────────────────┘  │     ┌───────────────────┐
                                          │     │  Fusion engine     │
                ┌─────────────────────┐  │     │  (FastAPI service) │
  mic ─────────▶│   Text branch        │──┘     └─────────┬─────────┘
                │  (Whisper/Google STT  │                 │ distress score
                │   + keyword/lexicon)  │                 │ + per-branch detail
                └─────────────────────┘                   ▼
                                                 ┌───────────────────┐
                                                 │ Node/Express API   │
                                                 │  + MySQL           │
                                                 └─────────┬─────────┘
                                                            │ WebSocket push
                                                            ▼
                                                 ┌───────────────────┐
                                                 │ React dashboard    │
                                                 └───────────────────┘
```

- **Late-Fusion Engine**: Independent branch processing combined at the probability/score level.
- **Windowed Inference**: Operates on fixed 3–5 second temporal windows.
- **Triage Aid, Not Diagnostic Tool**: Generates continuous distress scores and proactive alerts for nursing staff.

---

## 2. Repository Structure

```
emo-sis/
├── ai/
│   ├── audio/                 # Audio branch (features, CNN model, training, inference)
│   ├── facial/                # Facial branch (face detection, MobileNetV2, inference)
│   ├── text/                  # Text branch (Whisper, emergency distress scoring)
│   ├── fusion/                # Late fusion algorithm & FastAPI /infer microservice
│   ├── notebooks/             # Exploratory notebooks (audio EDA, facial EDA)
│   ├── models/                # Trained checkpoints & weights (.pt, gitignored)
│   └── requirements.txt
├── backend/                   # Node.js/Express API & WebSocket server
│   ├── src/
│   │   ├── db/                # MySQL connection, queries, and migration runner
│   │   ├── routes/            # REST API (patients, alerts, inferences)
│   │   └── sockets/           # Real-time WebSocket alert broadcasting
│   ├── migrations/            # Schema DDL (001_init.sql)
│   └── package.json
├── dashboard/                 # React/TypeScript nurse station dashboard (Phase 3)
├── docs/                      # Research framing & clinical documentation
└── plans/                     # Development roadmap and phased implementation plans
```

---

## 3. Getting Started

### Prerequisites
- Python 3.11+ (or uv package manager)
- Node.js 18+ & npm
- MySQL Server 8.0+

### Setup AI Pipeline
```bash
# Create virtual environment and install dependencies
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r ai/requirements.txt

# Run inference test
python -m ai.audio.infer <sample.wav>
python -m ai.facial.infer <sample.jpg>

# Start Fusion API
uvicorn ai.fusion.service:app --reload --port 8000
```

### Setup Backend API
```bash
cd backend
npm install
npm run migrate    # Applies database schema
npm run dev        # Starts Express + Socket.IO server on port 5000
```