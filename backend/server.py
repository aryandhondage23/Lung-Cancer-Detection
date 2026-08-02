"""
SCANLINE — Pulmonary CT Triage — Backend
------------------------------------------
FastAPI app that:
  1. Serves the frontend (static/index.html) at "/"
  2. Serves uploaded images at "/files/*" (for thumbnails)
  3. Exposes a JSON API under "/api/*" for prediction, training, and an
     autonomous "agent loop":

        every prediction    -> image + predicted label is auto-filed into
                                the training set (self-expanding dataset)
        every N new samples -> a retrain is triggered automatically,
                                in the background, no user action needed
        every action        -> written to an activity log the frontend
                                polls, so the autonomy is visible, not silent

Run with:
    uvicorn server:app --reload --port 8000

Then open:
    http://localhost:8000
"""

import os
import json
import shutil
import uuid
import logging
import threading
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# --------------------------------------------------------------------------
# Optional ML stack. The app runs fine (with mock predictions) even if
# TensorFlow / a trained model are not present.
# --------------------------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    from PIL import Image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[INFO] TensorFlow not installed - predictions will be mocked.")

# --------------------------------------------------------------------------
# Paths & setup
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

STATIC_DIR = ROOT_DIR / "static"
MODELS_DIR = ROOT_DIR / "models"
STORAGE_DIR = ROOT_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
TRAIN_DIR = STORAGE_DIR / "train"
TRAIN_CANCER_DIR = TRAIN_DIR / "cancer"
TRAIN_NORMAL_DIR = TRAIN_DIR / "normal"
STATE_FILE = STORAGE_DIR / "agent_state.json"

for d in [MODELS_DIR, UPLOADS_DIR, TRAIN_CANCER_DIR, TRAIN_NORMAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "best_lung_cancer_model.keras"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("scanline")

# --------------------------------------------------------------------------
# Agent state (persisted as JSON): retrain threshold, counters, versioning
# --------------------------------------------------------------------------
DEFAULT_STATE = {
    "samples_since_retrain": 0,
    "retrain_threshold": 8,
    "model_version": 0,
    "last_retrain": None,
    "retraining": False,
}

_state_lock = threading.Lock()


def load_state():
    if STATE_FILE.exists():
        try:
            return {**DEFAULT_STATE, **json.loads(STATE_FILE.read_text())}
        except Exception:
            pass
    return dict(DEFAULT_STATE)


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


agent_state = load_state()


def update_state(**kwargs):
    with _state_lock:
        agent_state.update(kwargs)
        save_state(agent_state)


# --------------------------------------------------------------------------
# Database (SQLite, zero external setup needed)
# --------------------------------------------------------------------------
DATABASE_URL = f"sqlite:///{ROOT_DIR / 'lung_cancer.db'}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ImageDB(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    label = Column(String)         # ground-truth-ish label: cancer / normal / unknown
    pred_label = Column(String)    # model prediction: cancer / normal
    prob = Column(Float)           # prediction probability (0-1)
    corrected = Column(Integer, default=0)  # 1 if a human corrected the label
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AgentLog(Base):
    __tablename__ = "agent_log"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String)   # PREDICT / AUTO-ADD / RETRAIN / CORRECT / INFO
    message = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)


@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_event(event_type: str, message: str):
    with get_db() as db:
        entry = AgentLog(event_type=event_type, message=message)
        db.add(entry)
        db.commit()
    logger.info(f"[{event_type}] {message}")


# --------------------------------------------------------------------------
# Pydantic schemas
# --------------------------------------------------------------------------
class ImageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    filename: str
    label: str
    pred_label: str
    prob: float
    corrected: int
    created_at: datetime


class PredictionResponse(BaseModel):
    id: int
    filename: str
    prob: float
    pred_label: str
    label: str


class StatusResponse(BaseModel):
    tensorflow_available: bool
    model_loaded: bool
    mode: str
    retraining: bool
    model_version: int


class DatasetStats(BaseModel):
    cancer_count: int
    normal_count: int
    total: int
    samples_since_retrain: int
    retrain_threshold: int
    model_version: int
    last_retrain: Optional[datetime]


class LogEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    event_type: str
    message: str
    created_at: datetime


class FeedbackRequest(BaseModel):
    true_label: str  # "cancer" or "normal"


# --------------------------------------------------------------------------
# ML model wrapper
# --------------------------------------------------------------------------
class MLModel:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        if not TF_AVAILABLE:
            return
        if MODEL_PATH.exists():
            try:
                self.model = keras.models.load_model(str(MODEL_PATH), compile=False)
                logger.info(f"Model loaded from {MODEL_PATH}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.model = None
        else:
            logger.info(f"No trained model found at {MODEL_PATH}. Using mock predictions.")

    def preprocess_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((150, 150))
        arr = np.array(img) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, image_path: str):
        if not TF_AVAILABLE or self.model is None:
            import random
            prob = round(random.uniform(0.05, 0.95), 4)
            pred_label = "cancer" if prob > 0.5 else "normal"
            return pred_label, float(prob)
        try:
            arr = self.preprocess_image(image_path)
            prediction = self.model.predict(arr, verbose=0)
            prob = float(prediction[0][0])
            pred_label = "cancer" if prob > 0.5 else "normal"
            return pred_label, prob
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    def retrain(self):
        if not TF_AVAILABLE or self.model is None:
            return {"status": "error", "message": "No base model loaded to fine-tune. Add one at backend/models/best_lung_cancer_model.keras."}

        cancer_images = [p for p in TRAIN_CANCER_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        normal_images = [p for p in TRAIN_NORMAL_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]

        if not cancer_images and not normal_images:
            return {"status": "error", "message": "No labeled training data yet."}

        try:
            X, y = [], []
            for p in cancer_images:
                X.append(self.preprocess_image(str(p))[0])
                y.append(1)
            for p in normal_images:
                X.append(self.preprocess_image(str(p))[0])
                y.append(0)

            X, y = np.array(X), np.array(y)

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            self.model.fit(X, y, epochs=5, batch_size=8, verbose=1)
            self.model.save(str(MODEL_PATH))

            return {
                "status": "success",
                "message": f"retrained on {len(X)} images ({len(cancer_images)} cancer / {len(normal_images)} normal)",
                "samples": {"cancer": len(cancer_images), "normal": len(normal_images)},
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


ml_model = MLModel()
_retrain_lock = threading.Lock()


def run_autonomous_retrain(trigger: str):
    """Runs in a background task. Retrains if possible, always logs what it did."""
    if _retrain_lock.locked():
        return
    with _retrain_lock:
        if not TF_AVAILABLE or ml_model.model is None:
            log_event("RETRAIN", f"skipped auto-retrain (trigger: {trigger}) — no base model loaded yet")
            update_state(samples_since_retrain=0)
            return
        update_state(retraining=True)
        log_event("RETRAIN", f"auto-retrain started (trigger: {trigger})")
        result = ml_model.retrain()
        if result["status"] == "success":
            new_version = agent_state["model_version"] + 1
            update_state(
                samples_since_retrain=0,
                model_version=new_version,
                last_retrain=datetime.now(timezone.utc).isoformat(),
                retraining=False,
            )
            log_event("RETRAIN", f"auto-retrain complete — {result['message']} — model v{new_version}")
        else:
            update_state(retraining=False)
            log_event("RETRAIN", f"auto-retrain failed — {result['message']}")


def auto_file_into_dataset(file_path: Path, unique_filename: str, pred_label: str):
    """Agentic step: every prediction's image is copied into the training set
    under its predicted label, growing the dataset without being asked."""
    target_dir = TRAIN_CANCER_DIR if pred_label == "cancer" else TRAIN_NORMAL_DIR
    dest = target_dir / unique_filename
    try:
        shutil.copy(file_path, dest)
    except Exception as e:
        logger.warning(f"auto-file failed: {e}")
        return

    log_event("AUTO-ADD", f"{unique_filename} filed under '{pred_label}' (self-labeled) — dataset growing")

    new_count = agent_state["samples_since_retrain"] + 1
    update_state(samples_since_retrain=new_count)

    if new_count >= agent_state["retrain_threshold"]:
        run_autonomous_retrain(trigger=f"{new_count} new samples since last retrain")


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------
app = FastAPI(title="SCANLINE — Pulmonary CT Triage")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")


@api_router.get("/")
async def root():
    return {"message": "SCANLINE Pulmonary CT Triage API"}


@api_router.get("/status", response_model=StatusResponse)
async def status():
    return StatusResponse(
        tensorflow_available=TF_AVAILABLE,
        model_loaded=ml_model.model is not None,
        mode="real" if ml_model.model is not None else "mock",
        retraining=agent_state.get("retraining", False),
        model_version=agent_state.get("model_version", 0),
    )


@api_router.get("/dataset-stats", response_model=DatasetStats)
async def dataset_stats():
    cancer_count = len(list(TRAIN_CANCER_DIR.glob("*")))
    normal_count = len(list(TRAIN_NORMAL_DIR.glob("*")))
    last_retrain = agent_state.get("last_retrain")
    return DatasetStats(
        cancer_count=cancer_count,
        normal_count=normal_count,
        total=cancer_count + normal_count,
        samples_since_retrain=agent_state.get("samples_since_retrain", 0),
        retrain_threshold=agent_state.get("retrain_threshold", 8),
        model_version=agent_state.get("model_version", 0),
        last_retrain=datetime.fromisoformat(last_retrain) if last_retrain else None,
    )


@api_router.get("/agent-log", response_model=List[LogEntry])
async def get_agent_log(limit: int = 30):
    with get_db() as db:
        rows = db.query(AgentLog).order_by(AgentLog.created_at.desc()).limit(limit).all()
        return rows


@api_router.post("/predict", response_model=PredictionResponse)
async def predict_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a CT scan image, get a prediction. The image + predicted label
    is then autonomously filed into the training set in the background."""
    allowed_ext = {".png", ".jpg", ".jpeg"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_ext:
        raise HTTPException(status_code=400, detail="Only PNG/JPG images are supported.")

    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOADS_DIR / unique_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pred_label, prob = ml_model.predict(str(file_path))
        log_event("PREDICT", f"{unique_filename} -> {pred_label} (p={prob:.3f})")

        with get_db() as db:
            db_image = ImageDB(filename=unique_filename, label="unknown", pred_label=pred_label, prob=prob)
            db.add(db_image)
            db.commit()
            db.refresh(db_image)
            image_id = db_image.id

        background_tasks.add_task(auto_file_into_dataset, file_path, unique_filename, pred_label)

        return PredictionResponse(
            id=image_id, filename=unique_filename, prob=prob, pred_label=pred_label, label="unknown"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/upload-labeled")
async def upload_labeled_image(background_tasks: BackgroundTasks, file: UploadFile = File(...), label: str = Form(...)):
    """Upload a CT scan with a known label, to grow the training set directly."""
    if label not in ("cancer", "normal"):
        raise HTTPException(status_code=400, detail="Label must be 'cancer' or 'normal'.")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in (".png", ".jpg", ".jpeg"):
        raise HTTPException(status_code=400, detail="Only PNG/JPG images are supported.")

    unique_filename = f"{uuid.uuid4()}{file_ext}"
    target_dir = TRAIN_CANCER_DIR if label == "cancer" else TRAIN_NORMAL_DIR
    file_path = target_dir / unique_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        shutil.copy(file_path, UPLOADS_DIR / unique_filename)

        pred_label, prob = ml_model.predict(str(file_path))

        with get_db() as db:
            db_image = ImageDB(filename=unique_filename, label=label, pred_label=pred_label, prob=prob)
            db.add(db_image)
            db.commit()
            db.refresh(db_image)

        log_event("USER-ADD", f"{unique_filename} labeled '{label}' by user — added to training set")

        new_count = agent_state["samples_since_retrain"] + 1
        update_state(samples_since_retrain=new_count)
        if new_count >= agent_state["retrain_threshold"]:
            background_tasks.add_task(run_autonomous_retrain, f"{new_count} new samples since last retrain")

        return {"id": db_image.id, "filename": unique_filename, "label": label, "message": "Saved for training."}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("upload-labeled failed")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/feedback/{image_id}")
async def submit_feedback(image_id: int, feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """Human correction loop: tell the agent the true label for a past
    prediction. Moves the file into the correct training bucket."""
    if feedback.true_label not in ("cancer", "normal"):
        raise HTTPException(status_code=400, detail="true_label must be 'cancer' or 'normal'.")

    with get_db() as db:
        img = db.query(ImageDB).filter(ImageDB.id == image_id).first()
        if not img:
            raise HTTPException(status_code=404, detail="Image not found.")

        old_pred = img.pred_label
        img.label = feedback.true_label
        img.corrected = 1
        db.commit()
        filename = img.filename

    # Move the file between train buckets if it was auto-filed under the wrong label
    wrong_dir = TRAIN_CANCER_DIR if old_pred == "cancer" else TRAIN_NORMAL_DIR
    right_dir = TRAIN_CANCER_DIR if feedback.true_label == "cancer" else TRAIN_NORMAL_DIR
    wrong_path = wrong_dir / filename
    right_path = right_dir / filename
    if wrong_path.exists() and old_pred != feedback.true_label:
        shutil.move(str(wrong_path), str(right_path))

    log_event("CORRECT", f"{filename} corrected: model said '{old_pred}', true label is '{feedback.true_label}'")

    return {"id": image_id, "label": feedback.true_label, "message": "Correction recorded. Thanks for training the agent."}


@api_router.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Manually trigger a retrain right now (in addition to the automatic one)."""
    if _retrain_lock.locked() or agent_state.get("retraining"):
        raise HTTPException(status_code=409, detail="A retrain is already in progress.")
    background_tasks.add_task(run_autonomous_retrain, "manual trigger")
    return {"status": "started", "message": "Retrain started in the background."}


@api_router.get("/images", response_model=List[ImageResponse])
async def get_images():
    with get_db() as db:
        return db.query(ImageDB).order_by(ImageDB.created_at.desc()).limit(50).all()


app.include_router(api_router)

# --------------------------------------------------------------------------
# Static files: uploaded images (thumbnails) + the frontend itself
# --------------------------------------------------------------------------
app.mount("/files", StaticFiles(directory=str(UPLOADS_DIR)), name="files")
app.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="assets")


@app.get("/")
async def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))
