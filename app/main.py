from __future__ import annotations
import base64
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf

from src.trn.trn_inference import load_models, refine
from src.llm.guidance import get_guidance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRN_PATH = PROJECT_ROOT / "srn_unet_model_128.keras"
TRN_PATH = PROJECT_ROOT / "models" / "trn" / "trn_generator.keras"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="LLM-Guided Cross-Domain Image Inpainting", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_srn = None
_generator = None


def _models():
    global _srn, _generator
    if _srn is None or _generator is None:
        if not SRN_PATH.is_file():
            raise FileNotFoundError(f"Missing SRN model: {SRN_PATH}")
        if not TRN_PATH.is_file():
            raise FileNotFoundError(f"Missing TRN generator: {TRN_PATH}")
        _srn, _generator = load_models(SRN_PATH, TRN_PATH)
    return _srn, _generator


def _rgb(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _mask(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("L").resize((128, 128), Image.Resampling.NEAREST)
    return (np.asarray(image, dtype=np.float32) >= 127.5).astype(np.float32)[..., None]


def _png(arr: np.ndarray) -> str:
    arr = np.clip(arr, 0, 1)
    image = Image.fromarray(np.round(arr * 255).astype(np.uint8))
    buf = io.BytesIO(); image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _metrics(gt: np.ndarray | None, pred: np.ndarray, mask: np.ndarray):
    if gt is None:
        return None
    mse = float(np.mean((gt - pred) ** 2))
    psnr = float(tf.image.psnr(gt[None], pred[None], max_val=1.0).numpy()[0])
    ssim = float(tf.image.ssim(gt[None], pred[None], max_val=1.0).numpy()[0])
    missing = np.repeat(mask, 3, axis=2).astype(bool)
    mmse = float(np.mean((gt[missing] - pred[missing]) ** 2)) if missing.any() else 0.0
    mpsnr = float(-10.0 * np.log10(max(mmse, 1e-12))) if mmse > 0 else 99.0
    return {"full_psnr": psnr, "ssim": ssim, "missing_psnr": mpsnr, "full_mse": mse}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "srn_model": SRN_PATH.exists(), "trn_model": TRN_PATH.exists(),
            "llm_enabled": bool(os.getenv("OPENAI_API_KEY"))}


@app.post("/api/reconstruct")
async def reconstruct(masked_image: UploadFile = File(...), mask_image: UploadFile = File(...),
                      domain: str = Form("AUTO") , ground_truth: UploadFile | None = File(None)):
    try:
        masked = _rgb(await masked_image.read())
        mask = _mask(await mask_image.read())
        if mask.mean() <= 0 or mask.mean() >= 1:
            raise HTTPException(400, "Mask must contain both missing and known pixels; white/1 means missing.")
        gt = _rgb(await ground_truth.read()) if ground_truth is not None else None
        domain = domain.upper().strip()
        if domain not in ["AUTO", "X-RAY", "BRAIN-MRI", "AERIAL", "FACES", "PLACES"]:
            raise HTTPException(400, f"Unsupported domain: {domain}")
        guidance = get_guidance(domain, float(mask.mean()), (128, 128))
        srn, generator = _models()
        x = np.concatenate([masked, mask], axis=-1)[None]
        result = refine(srn, generator, x, residual_scale=0.10)
        srn_out = result["srn_output"][0]
        trn = result["trn_candidate"][0]
        final = result["final_output"][0]
        # Known pixels must be exactly the uploaded masked input in the production composite.
        known_delta = float(np.max(np.abs(final * (1-mask) - masked * (1-mask))))
        metrics = {"srn": _metrics(gt, srn_out, mask), "trn_candidate": _metrics(gt, trn, mask), "final": _metrics(gt, final, mask),
                   "known_pixel_max_delta": known_delta}
        return {
            "domain": domain,
            "guidance": guidance,
            "images": {"masked_input": _png(masked), "mask": _png(np.repeat(mask,3,axis=2)),
                       "srn_output": _png(srn_out), "trn_candidate": _png(trn), "final_output": _png(final),
                       "ground_truth": _png(gt) if gt is not None else None},
            "metrics": metrics,
            "model": {"srn": str(SRN_PATH.name), "trn_generator": str(TRN_PATH.relative_to(PROJECT_ROOT)),
                      "residual_scale": 0.10, "known_pixels_preserved": known_delta == 0.0}
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
