"""
llm_client.py — Optional LLM client using standard library urllib.

Calls OpenAI-compatible /chat/completions endpoint to generate scene copy.
Never crashes, never leaks the API key, and strictly validates all output
through fact_checker and length rules.
"""

import os
import json
import re
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

from src.models import PropertyData
from src.fact_checker import extract_allowed_facts, check_text
from src.logger import setup_logger

logger = setup_logger()

# Load environment variables from .env if present
load_dotenv()

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT = 20  # seconds


def _safe_error_msg(exc: Exception) -> str:
    """Format exception message without leaking any potential API keys."""
    msg = str(exc)
    return re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", msg)


def generate_scene_copy(
    prop: PropertyData,
    scene_skeleton: list[dict],
) -> tuple[list[dict] | None, list[str]]:
    """Attempt to generate scene copy using an OpenAI-compatible LLM endpoint.

    Returns:
        (scenes_data | None, warnings)
        If anything fails, scenes_data is None and warnings explain why.
    """
    warnings: list[str] = []

    # ── Check environment variables ────────────────────────────────────────
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        warnings.append(
            "LLM mode requested but LLM_API_KEY is not set. "
            "Falling back to template plan."
        )
        return None, warnings

    model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/") or DEFAULT_BASE_URL
    endpoint = f"{base_url}/chat/completions"

    # ── Build prompt ───────────────────────────────────────────────────────
    expected_count = len(scene_skeleton)
    skeleton_summary = [
        {"scene_index": i + 1, "scene_type": s["scene_type"]}
        for i, s in enumerate(scene_skeleton)
    ]

    facts_dict = {
        "name": prop.name,
        "location": prop.location,
        "price": prop.price,
        "property_type": prop.property_type,
        "features": prop.features,
        "cta_text": prop.cta_text,
        "contact": prop.contact,
        "tagline": prop.tagline,
    }

    system_prompt = (
        "You are an assistant that writes video scripts and captions for real estate promotional videos.\n"
        "STRICT FACTS-ONLY RULE:\n"
        "- Use ONLY the facts provided in the input. Never invent any numbers, amenities, features, distances, or claims.\n"
        "- Do NOT use buzzwords or promotional claims such as 'luxury', 'premium', 'guaranteed', 'best in class', "
        "'RERA approved', 'near metro', 'limited time offer', or 'investment return' unless they appear in the input.\n"
        "- Length limits:\n"
        "  - Caption: maximum 2 lines, maximum 38 characters per line.\n"
        "  - Voiceover: maximum 18 words, short natural single sentence.\n"
        "- Output format: Return strictly valid JSON with no markdown and no extra text:\n"
        '{"scenes": [{"caption": "...", "voiceover": "..."}]}'
    )

    user_prompt = (
        f"Generate copy for exactly {expected_count} scenes based ONLY on these facts:\n"
        f"FACTS:\n{json.dumps(facts_dict, indent=2, ensure_ascii=False)}\n\n"
        f"SCENE TYPES:\n{json.dumps(skeleton_summary, indent=2)}\n\n"
        f"Return JSON with exactly {expected_count} items under 'scenes'."
    )

    req_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    # ── Perform HTTP request ───────────────────────────────────────────────
    req_bytes = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=req_bytes,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            resp_bytes = response.read()
            resp_json = json.loads(resp_bytes.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_msg = _safe_error_msg(exc)
        logger.warning("LLM API HTTP error: %s", err_msg)
        warnings.append(f"LLM request returned HTTP error ({err_msg}). Falling back to template plan.")
        return None, warnings
    except urllib.error.URLError as exc:
        err_msg = _safe_error_msg(exc)
        logger.warning("LLM API network error: %s", err_msg)
        warnings.append(f"LLM network error ({err_msg}). Falling back to template plan.")
        return None, warnings
    except TimeoutError:
        logger.warning("LLM API request timed out after %ds", REQUEST_TIMEOUT)
        warnings.append(f"LLM request timed out after {REQUEST_TIMEOUT}s. Falling back to template plan.")
        return None, warnings
    except Exception as exc:
        err_msg = _safe_error_msg(exc)
        logger.warning("LLM API unexpected error: %s", err_msg)
        warnings.append(f"LLM unexpected error ({err_msg}). Falling back to template plan.")
        return None, warnings

    # ── Parse response ─────────────────────────────────────────────────────
    try:
        content = resp_json["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        warnings.append(f"LLM response missing expected structure: {exc}. Falling back to template plan.")
        return None, warnings

    # Strip code fences if model enclosed JSON in ```json ... ```
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        warnings.append(f"LLM returned malformed JSON: {exc}. Falling back to template plan.")
        return None, warnings

    if not isinstance(parsed, dict) or "scenes" not in parsed:
        warnings.append("LLM JSON missing top-level 'scenes' key. Falling back to template plan.")
        return None, warnings

    raw_scenes = parsed["scenes"]
    if not isinstance(raw_scenes, list) or len(raw_scenes) != expected_count:
        warnings.append(
            f"LLM returned {len(raw_scenes) if isinstance(raw_scenes, list) else 0} scenes; "
            f"expected {expected_count}. Falling back to template plan."
        )
        return None, warnings

    # ── Fact-check and validate length on every scene ─────────────────────
    allowed_facts = extract_allowed_facts(prop)
    validated_scenes: list[dict] = []

    for i, item in enumerate(raw_scenes):
        if not isinstance(item, dict):
            warnings.append(f"LLM scene {i + 1} is not a JSON object. Falling back to template plan.")
            return None, warnings

        caption = str(item.get("caption", "")).strip()
        voiceover = str(item.get("voiceover", "")).strip()

        if not caption or not voiceover:
            warnings.append(
                f"LLM scene {i + 1} produced an empty caption or voiceover. "
                "Falling back to template plan."
            )
            return None, warnings

        # Length validation
        cap_lines = [line.strip() for line in caption.split("\n") if line.strip()]
        if len(cap_lines) > 2 or any(len(line) > 38 for line in cap_lines):
            warnings.append(
                f"LLM scene {i + 1} caption violates length constraints (max 2 lines, 38 chars/line). "
                "Falling back to template plan."
            )
            return None, warnings

        vo_words = voiceover.split()
        if len(vo_words) > 18:
            warnings.append(
                f"LLM scene {i + 1} voiceover has {len(vo_words)} words (max 18). "
                "Falling back to template plan."
            )
            return None, warnings

        # Fact checking
        ok_c, prob_c = check_text(caption, allowed_facts)
        if not ok_c:
            warnings.append(
                f"LLM scene {i + 1} caption failed fact check: {'; '.join(prob_c)}. "
                "Falling back to template plan."
            )
            return None, warnings

        ok_v, prob_v = check_text(voiceover, allowed_facts)
        if not ok_v:
            warnings.append(
                f"LLM scene {i + 1} voiceover failed fact check: {'; '.join(prob_v)}. "
                "Falling back to template plan."
            )
            return None, warnings

        validated_scenes.append(
            {
                "scene_type": scene_skeleton[i]["scene_type"],
                "caption": caption,
                "voiceover": voiceover,
            }
        )

    return validated_scenes, warnings
