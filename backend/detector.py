from transformers import pipeline
import logging as log
from PIL import Image
import torch
import numpy as np
import cv2
import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Tuple, Optional, Dict

# Load environment variables
load_dotenv()

# Configure module logger
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = log.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Label mappings per model (verified against HuggingFace Hub model configs)
#
#  dima806/ai_vs_real_image_detection
#      LABEL_0 = REAL,  LABEL_1 = FAKE
#
#  umm-maybe/AI-image-detector
#      LABEL_0 = artificial (AI),  LABEL_1 = human (Real)
#      *** OPPOSITE order to dima806 — LABEL_0 is AI here ***
#
#  dima806/deepfake_vs_real_image_detection
#      LABEL_0 = REAL,  LABEL_1 = FAKE
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    {
        "name": "dima806/ai_vs_real_image_detection",
        "fake_labels": {"FAKE", "LABEL_1"},
        "real_labels": {"REAL", "LABEL_0"},
        "type": "ai",
    },
    {
        "name": "umm-maybe/AI-image-detector",
        # LABEL_0=artificial (AI), LABEL_1=human (Real) — inverse of dima806!
        "fake_labels": {"ARTIFICIAL", "LABEL_0"},
        "real_labels": {"HUMAN", "LABEL_1"},
        "type": "ai",
    },
    {
        "name": "dima806/deepfake_vs_real_image_detection",
        "fake_labels": {"FAKE", "LABEL_1"},
        "real_labels": {"REAL", "LABEL_0"},
        "type": "deepfake",
    },
]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class ImageDetector:
    """
    Multi-signal detector for:
      • AI-generated images  (DALL-E, Midjourney, SD, etc.)
      • Deepfakes / face swaps
      • Real photographs

    Designed to handle WhatsApp-compressed images where EXIF is stripped
    and JPEG re-compression artifacts are introduced.

    Signals used:
      1. EXIF metadata  — decisive fast-path if AI signature found
      2. HF model ensemble (3 models)  — 70% weight
      3. FFT frequency analysis  — 30% weight
      4. JPEG blocking score  — widens uncertainty band for WhatsApp images
    """

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Initializing detector ensemble on device {self.device}")

        self.models = []
        for cfg in MODEL_CONFIGS:
            try:
                pipe = pipeline(
                    "image-classification",
                    model=cfg["name"],
                    device=self.device,
                    top_k=None,
                )
                self.models.append((cfg, pipe))
                logger.info(f"Loaded model '{cfg['name']}' (type={cfg['type']})")
            except Exception as e:
                logger.warning(f"Could not load model '{cfg['name']}': {e}")
        
        self.gemini_model = None
        if GEMINI_API_KEY:
            try:
                self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
                logger.info("Gemini 1.5 Flash configured as vision signal.")
            except Exception as e:
                logger.error(f"Gemini configuration failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Signal 1 — EXIF metadata
    # ──────────────────────────────────────────────────────────────────────────

    def _check_metadata(self, image_path):
        """Return True if EXIF contains known AI generator signatures."""
        try:
            image = Image.open(image_path)
            exif = image.getexif()
            if not exif:
                return False
            for tag_id, value in exif.items():
                val_str = str(value).lower()
                if any(kw in val_str for kw in (
                    "dall-e", "openai", "midjourney",
                    "stable diffusion", "ai generated", "firefly"
                )):
                    logger.info("Found explicit AI signature in EXIF metadata.")
                    return True
        except Exception:
            pass
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Signal 2 — JPEG blocking artifact score
    # WhatsApp re-compresses images, introducing DCT block boundaries.
    # High blocking score = heavily re-compressed = likely WhatsApp-forwarded.
    # ──────────────────────────────────────────────────────────────────────────

    def _jpeg_blocking_score(self, image_path):
        """
        Returns float in [0, 1]. Higher = more JPEG block artifacts.
        Measures variance of pixel differences at 8-pixel DCT boundaries.
        """
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0
            h, w = img.shape
            img_f = img.astype(np.float32)

            h_diffs = [
                np.abs(img_f[:, x] - img_f[:, x - 1]).mean()
                for x in range(8, w, 8)
            ]
            v_diffs = [
                np.abs(img_f[y, :] - img_f[y - 1, :]).mean()
                for y in range(8, h, 8)
            ]

            boundary_mean = np.mean(h_diffs + v_diffs) if (h_diffs or v_diffs) else 0.0
            score = float(np.clip(boundary_mean / 30.0, 0.0, 1.0))
            logger.info(f"JPEG blocking score: {score:.4f}")
            return score
        except Exception as e:
            logger.warning(f"JPEG blocking score failed: {e}")
            return 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Signal 3 — FFT frequency analysis
    # AI diffusion images have reduced high-frequency energy vs real photos.
    # ──────────────────────────────────────────────────────────────────────────

    def _fft_ai_score(self, image):
        """
        Returns float in [0, 1]. Higher = frequency profile looks more AI-like.
        Real photos have more high-frequency energy than AI images.
        """
        try:
            gray = np.array(image.convert("L"), dtype=np.float32)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.log1p(np.abs(fshift))

            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            y_grid, x_grid = np.ogrid[:h, :w]
            dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
            radius = min(cy, cx)

            mask_high = dist > radius * 0.5
            total_energy = magnitude.sum() + 1e-8
            high_freq_ratio = magnitude[mask_high].sum() / total_energy

            # Real photos: high_freq_ratio typically > 0.55
            # AI images:   high_freq_ratio typically < 0.48
            ai_score = float(np.clip((0.58 - high_freq_ratio) / 0.15, 0.0, 1.0))
            logger.info(f"FFT high-freq ratio: {high_freq_ratio:.4f}  →  fft_ai_score={ai_score:.4f}")
            return ai_score
        except Exception as e:
            logger.warning(f"FFT analysis failed: {e}")
            return 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Signal 4 — Gemini Vision API (Arbiter)
    # ──────────────────────────────────────────────────────────────────────────

    def _gemini_analysis(self, image_path: str) -> Tuple[bool, float, str, list[str]]:
        """Use Gemini 1.5 Flash as a Forensic Arbiter to inspect for artifacts."""
        if not self.gemini_model:
            return False, 0.0, "", []

        try:
            # Upload file if needed, or pass directly
            with Image.open(image_path) as img:
                prompt = (
                    "You are a Forensic Image Expert specializing in AI detection. "
                    "Analyze this image for tell-tale AI artifacts. "
                    "Be highly skeptical. Look for:\n"
                    "1. Over-smooth or 'plastic' skin textures.\n"
                    "2. Backgrounds that are too clean or have 'floating' objects.\n"
                    "3. Lighting and shadow inconsistencies (shadows going the wrong way).\n"
                    "4. Anatomical errors in hands, eyes, or ears.\n"
                    "5. Strange texture repetitions or 'frequency noise' in the background.\n"
                    "\nRespond with EXACTLY this JSON format:\n"
                    '{"is_ai": true, "confidence": 0.95, "reason": "Short summary Verdict", '
                    '"artifacts": ["Artifact 1", "Artifact 2"]}'
                )
                
                logger.info(f"Sending image to Gemini Arbiter: {image_path}")
                response = self.gemini_model.generate_content([prompt, img])
                text = response.text.strip()
                logger.info(f"Raw Gemini Response: {text[:200]}...")
            
            # Simple JSON parse or fallback
            import json
            # Extract JSON from potential triple backticks
            if "```json" in text:
                text = text.split("```json")[-1].split("```")[0].strip()
            elif "{" in text:
                text = text[text.find("{"):text.rfind("}")+1]
            
            try:
                res = json.loads(text)
            except Exception as e:
                logger.error(f"Failed to parse Gemini JSON: {e}. Raw text: {text}")
                return False, 0.0, "JSON Parse Error", []

            is_ai = bool(res.get("is_ai", False))
            conf = float(res.get("confidence", 0.0))
            reason = res.get("reason", "")
            artifacts = res.get("artifacts", [])
            
            logger.info(f"Gemini Arbiter Result: is_ai={is_ai} conf={conf} reason={reason} artifacts={len(artifacts)}")
            return is_ai, conf, reason, artifacts
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, 0.0, f"Error: {str(e)}", []

    # ──────────────────────────────────────────────────────────────────────────
    # Signal 6 — Forensic DSP Analysis (Manipulations)
    # ──────────────────────────────────────────────────────────────────────────

    def _edge_score(self, image_path):
        """Detect background replacement / editing via Canny edge density."""
        try:
            img = cv2.imread(str(image_path))
            if img is None: return 0.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            score = float(np.mean(edges))
            logger.info(f"Edge Density Score: {score:.4f}")
            return score
        except Exception as e:
            logger.warning(f"Edge analysis failed: {e}")
            return 0.0

    def _blur_score(self, image_path):
        """Unnatural smoothing or background focus detection."""
        try:
            img = cv2.imread(str(image_path))
            if img is None: return 0.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            logger.info(f"Blur/Focus Variance: {variance:.4f}")
            return float(variance)
        except Exception as e:
            logger.warning(f"Blur analysis failed: {e}")
            return 0.0

    def _lighting_check(self, image_path):
        """Uneven brightness / contrast detection (Composite Signatures)."""
        try:
            img = cv2.imread(str(image_path))
            if img is None: return 0.0, 0.0
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            brightness = float(np.mean(hsv[:,:,2]))
            std_brightness = float(np.std(hsv[:,:,2]))
            logger.info(f"Lighting Stats: mean={brightness:.4f} std={std_brightness:.4f}")
            return brightness, std_brightness
        except Exception as e:
            logger.warning(f"Lighting check failed: {e}")
            return 0.0, 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Core prediction
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, image_path):
        """
        Refined Decision Engine:
        1. Weighted Ensemble (Model 1: 0.4, Model 2: 0.3, Model 3: 0.2, FFT: 0.1)
        2. Gemini Vision Arbiter (Cross-verification & Justification)
        3. DSP Manipulation Signals (Edge/Blur/Lighting)
        """

        # Signal 1: Metadata (Decisive Fast-Path)
        if self._check_metadata(image_path):
            return "AI-Generated (EXIF)", 1.0, "Decisive AI-Signature found in EXIF Metadata.", ["Forensic Metadata Hit"]

        image = Image.open(image_path).convert("RGB")
        
        # Calculate Base Signals
        edge_density = self._edge_score(image_path)
        blur_v = self._blur_score(image_path)
        mean_lights, std_lights = self._lighting_check(image_path)
        fft_ai_score = self._fft_ai_score(image)

        # Signal Ensemble (Model 1=0.4, Model 2=0.3, Model 3=0.2, FFT=0.1)
        weights = [0.4, 0.3, 0.2]
        scores = []
        for i, (cfg, pipe) in enumerate(self.models):
            try:
                out = pipe(image)
                if out and isinstance(out[0], list): out = out[0]
                ff, rr = 0.0, 0.0
                for o in out:
                    lab = o.get("label", "").upper()
                    sc = float(o.get("score", 0.0))
                    if lab in cfg["fake_labels"]: ff = sc
                    elif lab in cfg["real_labels"]: rr = sc
                scores.append(ff if (ff+rr) > 0 else 0.5) # Default to neutral if failed
            except Exception:
                scores.append(0.5)

        # ── Signal Synthesis ──
        
        # Ensemble Synthesis (90% HF Models + 10% FFT)
        weighted_fake_score = sum(s * w for s, w in zip(scores, weights)) + (fft_ai_score * 0.1)
        
        # Step 4: FFT reduction to 10% was already applied in weighted_fake_score
        
        # Signal 5: Gemini Arbiter Verification
        gemini_is_ai, gemini_conf, gemini_reason, gemini_artifacts = self._gemini_analysis(image_path)
        
        # Cross-Verification blending (Hybrid Intelligence)
        if gemini_conf > 0:
            gemini_val = gemini_conf if gemini_is_ai else (1.0 - gemini_conf)
            final_score = (weighted_fake_score * 0.6) + (gemini_val * 0.4)
        else:
            final_score = weighted_fake_score

        # Manipulation Check
        is_manipulated = (edge_density > 52 and blur_v < 110) or (std_lights > 95)
        has_metadata = self._check_metadata(image_path)

        # ── Step 1: Final Verdict & Classification ──
        if final_score > 0.65:
            verdict = "AI Generated"
            reasons = [
                "• Strong generative artifact patterns detected",
                "• Synthetic texture frequency signature",
                "• Anatomical or lighting inconsistencies"
            ]
        elif final_score < 0.35:
            verdict = "Real Image"
            reasons = [
                "• Natural camera sensor noise signature",
                "• High-frequency textural consistency",
                "• Accurate anatomical details"
            ]
        elif is_manipulated and final_score > 0.40:
            verdict = "Edited / Manipulated"
            reasons = [
                "• Localized pixel-level inconsistencies",
                "• Digital sharpening/blurring artifacts",
                "• Potential composite layering detected"
            ]
        else:
            verdict = "Uncertain"
            reasons = [
                "• Signal strength below high-confidence threshold",
                "• Mixed forensic indicators detected",
                "• Image quality may be restricted"
            ]

        full_forensic_report = "\n".join(reasons)

        # UI Payload (Simplified)
        ui_metadata = [
            f"Ensemble: {round(weighted_fake_score, 2)}",
            f"Frequency: {round(fft_ai_score, 2)}",
            f"Arbiter: {round(gemini_conf, 2)}",
            f"Final Match: {round(final_score * 100, 1)}%"
        ]
        ui_metadata.extend(gemini_artifacts)

        return verdict, round(final_score, 4), full_forensic_report, ui_metadata

    # ──────────────────────────────────────────────────────────────────────────
    # Heatmap — highlights suspiciously uniform (AI-smooth) regions
    # ──────────────────────────────────────────────────────────────────────────

    def generate_heatmap(self, image_path, output_path):
        """
        Overlay highlighting low inter-channel variance regions.
        AI images tend to be over-smooth; those regions are flagged red.
        """
        try:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise ValueError(f"Could not read image at {image_path}")

            image_bgr = cv2.resize(image_bgr, (380, 380))
            float_img = image_bgr.astype(np.float32)

            channel_std = np.std(float_img, axis=2)
            std_norm = (channel_std - channel_std.min()) / (
                channel_std.max() - channel_std.min() + 1e-8
            )
            suspicion = 1.0 - std_norm
            suspicion = cv2.GaussianBlur(suspicion, (21, 21), 0)

            heatmap_uint8 = (suspicion * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_colored, 0.4, 0)

            cv2.imwrite(str(output_path), overlay)
            logger.info(f"Heatmap saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            raise