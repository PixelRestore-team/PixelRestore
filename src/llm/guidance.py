"""Domain-aware LLM guidance for the frozen epoch-30 TRN inference pipeline.

The LLM is an inference-time controller/advisor. It does not modify or retrain
SRN/TRN weights. If OPENAI_API_KEY is absent, a deterministic fallback keeps
the demo fully usable offline.
"""
from __future__ import annotations
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

try:
    from openai import OpenAI
except Exception:  # optional dependency / offline demo
    OpenAI = None


@dataclass
class Guidance:
    domain: str
    strategy: str
    residual_scale: float
    instructions: list[str]
    rationale: str
    source: str
    model: str



DOMAIN_DEFAULTS = {
    "X-RAY": Guidance("X-RAY", "conservative_structural", 0.10,
        ["Preserve anatomical continuity.", "Avoid artificial high-frequency texture.", "Keep intensity transitions smooth across masked regions."],
        "Medical imagery benefits from conservative refinement that preserves anatomical structure.", "fallback", "deterministic"),

    "BRAIN-MRI": Guidance("BRAIN-MRI", "conservative_structural", 0.10,
        ["Preserve anatomical boundaries.", "Avoid introducing texture not supported by nearby structure.", "Prefer smooth local intensity continuity."],
        "MRI reconstruction should prioritize structural continuity over aggressive texture synthesis.", "fallback", "deterministic"),

    "FACES": Guidance("FACES", "facial_continuity", 0.10,
        ["Preserve facial geometry.", "Maintain skin-tone continuity.", "Avoid sharp halos around reconstructed facial regions."],
        "Faces are highly sensitive to local geometry and boundary artifacts.", "fallback", "deterministic"),

    "AERIAL": Guidance("AERIAL", "scene_continuity", 0.10,
        ["Preserve road and building geometry.", "Maintain long edges and perspective continuity.", "Avoid synthetic texture at mask boundaries."],
        "Aerial imagery contains long geometric structures that should remain continuous.", "fallback", "deterministic"),

    "PLACES": Guidance("PLACES", "texture_conservative", 0.10,
        ["Preserve scene layout.", "Maintain natural texture continuity.", "Avoid over-sharpening reconstructed areas."],
        "Natural scenes benefit from preserving layout while avoiding over-enhanced texture.", "fallback", "deterministic"),
}


SYSTEM_PROMPT = """You are the semantic guidance controller for a cross-domain image inpainting demo.
The SRN and epoch-30 TRN weights are already trained and frozen. You must NOT propose retraining,
architecture changes, or arbitrary numerical tuning. Choose exactly one strategy from:
conservative_structural, facial_continuity, scene_continuity, texture_conservative.
Keep residual_scale at 0.10. Return JSON with keys: strategy, residual_scale, instructions, rationale.
Instructions must be short, practical guidance for reconstruction quality. Never claim the output is
identical to ground truth. Focus on continuity, structure, and avoiding boundary artifacts."""


def _fallback(domain: str) -> Guidance:
    key = domain.upper().strip()
    base = DOMAIN_DEFAULTS.get(key, Guidance(
        key, "conservative_structural", 0.10,
        ["Preserve dominant structures.", "Maintain local continuity.", "Avoid high-frequency boundary artifacts."],
        "Unknown domains use conservative refinement to reduce inference-time artifacts.", "fallback", "deterministic"))
    return base


def _clean_response(domain: str, data: dict[str, Any], model: str) -> Guidance:
    allowed = {"conservative_structural", "facial_continuity", "scene_continuity", "texture_conservative"}
    strategy = str(data.get("strategy", "conservative_structural"))
    if strategy not in allowed:
        strategy = "conservative_structural"
    instructions = data.get("instructions", [])
    if not isinstance(instructions, list):
        instructions = [str(instructions)]
    instructions = [str(x).strip() for x in instructions if str(x).strip()][:4]
    if not instructions:
        instructions = _fallback(domain).instructions
    return Guidance(
        domain=domain.upper().strip(),
        strategy=strategy,
        residual_scale=0.10,
        instructions=instructions,
        rationale=str(data.get("rationale", "Domain-aware conservative guidance.")),
        source="openai",
        model=model,
    )


def get_guidance(domain: str, mask_fraction: float, image_size: tuple[int, int] = (128, 128)) -> dict[str, Any]:
    """Get LLM guidance; automatically falls back if no API key/client is available."""
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return asdict(_fallback(domain)) | {"mask_fraction": float(mask_fraction), "image_size": list(image_size)}
    prompt = (
        f"Domain: {domain}\nMask fraction: {mask_fraction:.4f}\nImage size: {image_size[0]}x{image_size[1]}\n"
        "Give semantic guidance for this inference request. Keep residual_scale exactly 0.10."
    )
    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(model=model, instructions=SYSTEM_PROMPT, input=prompt)
        raw = response.output_text.strip()
        data = json.loads(raw)
        result = _clean_response(domain, data, model)
        return asdict(result) | {"mask_fraction": float(mask_fraction), "image_size": list(image_size)}
    except Exception as exc:
        result = _fallback(domain)
        return asdict(result) | {"mask_fraction": float(mask_fraction), "image_size": list(image_size),
                                 "fallback_reason": f"LLM unavailable; deterministic fallback used ({type(exc).__name__})."}
