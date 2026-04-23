from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import tempfile
import logging
import base64
import json
import os
from dotenv import load_dotenv

# load environment variables FROM .env file FIRST
load_dotenv()

from backend.detector import ImageDetector

# configure logger for app
logger = logging.getLogger("ai_image_detector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

# global detector instance
_detector: ImageDetector | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector
    # load model during startup
    _detector = ImageDetector()
    yield
    # cleanup if necessary
    _detector = None

app = FastAPI(
    title="AI Image Detector API",
    description="Detect whether an uploaded image is real or AI generated",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple User Model for Login
class LoginRequest(BaseModel):
    username: str
    password: str

# User Storage Helper
USERS_FILE = Path(__file__).parent / "users.json"

def load_users():
    if not USERS_FILE.exists():
        # Create default admin user if file doesn't exist
        default = {"admin": "admin123"}
        with open(USERS_FILE, "w") as f:
            json.dump(default, f)
        return default
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_user(username, password):
    users = load_users()
    users[username] = password
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

@app.post("/api/register")
async def register(request: LoginRequest):
    users = load_users()
    if request.username in users:
        raise HTTPException(status_code=400, detail="Identifier already in use")
    save_user(request.username, request.password)
    return {"status": "success", "message": "Forensic specialist registered"}

@app.post("/api/login")
async def login(request: LoginRequest):
    users = load_users()
    if request.username in users and users[request.username] == request.password:
        return {"status": "success", "token": f"token-{request.username}"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": _detector is not None,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze uploaded image for AI generation detection and log results."""
    try:
        # Basic validation
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type; please upload an image.")

        # enforce size limit
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large; max 10MB.")

        # save incoming bytes in case of errors
        await file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        logger.info(f"Received file {file.filename} ({len(content)} bytes)")

        try:
            pred, conf, reason, artifacts = _detector.predict(tmp_path)
            confidence_pct = f"{int(conf * 100)}%"
            
            # Use Gemini reason as the message if available
            message = reason if reason else None
            if not message and pred == "Uncertain":
                message = "Analysis inconclusive; confidence below threshold."

            # generate heatmap
            heatmap_path = tmp_path.parent / f"heatmap_{tmp_path.name}"
            heatmap_b64 = None
            try:
                _detector.generate_heatmap(tmp_path, heatmap_path)
                with open(heatmap_path, "rb") as h_file:
                    heatmap_b64 = base64.b64encode(h_file.read()).decode("utf-8")
            except Exception as h_err:
                logger.error(f"Heatmap generation error: {h_err}")
            finally:
                if heatmap_path.exists():
                    heatmap_path.unlink()

            response = {
                "prediction": pred,
                "confidence": confidence_pct,
                "confidence_score": conf,
                "status": "success",
                "heatmap": heatmap_b64,
                "message": message,
                "artifacts": artifacts
            }

            # log prediction details
            logger.info(f"Prediction: {pred} (score={conf:.4f})")
            return response
        except Exception as pred_err:
            # save failing image for inspection
            bad_dir = Path(__file__).parent / "bad_images"
            bad_dir.mkdir(exist_ok=True)
            dest = bad_dir / tmp_path.name
            shutil.copy(tmp_path, dest)
            logger.error(f"Prediction error saved to {dest}: {pred_err}")
            raise
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# mount frontend LAST so API routes take precedence
app.mount("/",
          StaticFiles(directory=Path(__file__).parent.parent / "frontend", html=True),
          name="frontend")
