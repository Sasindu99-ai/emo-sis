"""FastAPI inference microservice exposing POST /infer for emo-sis multimodal distress pipeline."""

import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.facial import infer as facial_branch
from ai.audio import infer as audio_branch
from ai.text import infer as text_branch
from ai.fusion.fuse import fuse_signals

app = FastAPI(
    title="emo-sis Multimodal Fusion & Inference API",
    description="Processes audio and visual window clips to calculate patient distress scores",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferDirectPayload(BaseModel):
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    audio_path: Optional[str] = None
    image_path: Optional[str] = None
    raw_text: Optional[str] = None


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "emo-sis-fusion", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/infer")
async def infer_window(
    audio_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    window_start: Optional[str] = Form(None),
    window_end: Optional[str] = Form(None),
    raw_text: Optional[str] = Form(None)
):
    """Processes multimodal monitoring window input (multipart file upload)."""
    now = datetime.now(timezone.utc)
    start_ts = window_start or now.isoformat()
    end_ts = window_end or now.isoformat()

    facial_res = None
    audio_res = None
    text_res = None

    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Process Audio file
        if audio_file:
            audio_path = os.path.join(temp_dir, audio_file.filename or "clip.wav")
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio_file.file, f)

            audio_res = audio_branch.predict(audio_path)
            text_res = text_branch.predict(audio_path)

        # 2. Process Image / Video frame
        if image_file:
            img_path = os.path.join(temp_dir, image_file.filename or "frame.jpg")
            with open(img_path, "wb") as f:
                shutil.copyfileobj(image_file.file, f)

            facial_res = facial_branch.predict(img_path)

        # Fallback text if explicitly supplied
        if raw_text and not text_res:
            from ai.text.distress_score import score_text_distress
            score, kws = score_text_distress(raw_text)
            text_res = {"transcript": raw_text, "distress_score": score, "keywords": kws}

        # Defaults if branches were absent in test run
        if not facial_res:
            facial_res = {"neutral": 0.8, "pain": 0.1, "panic": 0.05, "agitation": 0.05}
        if not audio_res:
            audio_res = {"neutral": 0.8, "fear": 0.1, "sad": 0.05, "angry": 0.05}
        if not text_res:
            text_res = {"transcript": "", "distress_score": 0.0, "keywords": []}

        # 3. Multimodal Late Fusion
        fused_res = fuse_signals(
            facial_probs=facial_res,
            audio_probs=audio_res,
            text_result=text_res
        )

        return {
            "window_start": start_ts,
            "window_end": end_ts,
            "facial": facial_res,
            "audio": audio_res,
            "text": text_res,
            "fused": {
                "distress_score": fused_res["distress_score"],
                "alert": fused_res["alert"]
            }
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai.fusion.service:app", host="0.0.0.0", port=8000, reload=True)
