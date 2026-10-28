"""
llm_client.py — Optional LLM client using standard library urllib.

Calls OpenAI-compatible /chat/completions endpoint to generate scene copy.
Never crashes, never leaks the API key, and strictly validates all output
per-scene through fact_checker and layout rules with a single targeted retry.
"""

import os
import json
import re
import urllib.request
import urllib.error
from dotenv import load_dotenv

from src.models import PropertyData
from src.fact_checker import extract_allowed_facts, check_text, AllowedFacts
from src.logger import setup_logger

logger = setup_logger()

# Load environment variables from .env if present
load_dotenv()

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT = 20  # seconds
TEMPERATURE = 0.3


def _safe_error_msg(exc: Exception) -> str:
    """Format exception message without leaking any potential API keys."""
    msg = str(exc)
    msg = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", msg)
    for env_var in ("LLM_API_KEY", "GEMINI_API_KEY"):
        key = os.getenv(env_var, "").strip()
        if key and len(key) > 4:
            msg = msg.replace(key, "[REDACTED]")
    return msg


def _format_problem_for_prompt(problem: str) -> str:
    """Convert fact_checker problem string into a clear instruction for LLM retry."""
    m_num = re.search(r"Number '([^']+)' found in text does not match any number in property facts", problem)
    if m_num:
        return f"contains the number {m_num.group(1)} which is not in the input"
    m_amenity = re.search(r"Amenity '([^']+)' mentioned in text was not present in property features/facts", problem)
    if m_amenity:
        return f"mentions unmentioned amenity '{m_amenity.group(1)}'"
    m_banned = re.search(r"Banned promotional claim '([^']+)'", problem)
    if m_banned:
        return f"contains prohibited buzzword '{m_banned.group(1)}'"
    return problem


def _validate_scene_copy(
    caption: str,
    voiceover: str,
    prop: PropertyData,
    allowed_facts: AllowedFacts,
    scene_idx: int,
) -> tuple[bool, list[str]]:
    """Validate a single scene's caption and voiceover against constraints and allowed facts.

    Returns:
        (is_valid, list_of_problem_descriptions)
    """
    problems: list[str] = []

    if not caption or not caption.strip():
        problems.append("caption is empty")
    if not voiceover or not voiceover.strip():
        problems.append("voiceover is empty")

    # Caption layout limits: max 2 lines, max 38 chars per line
    cap_lines = [line.strip() for line in caption.split("\n") if line.strip()]
    if len(cap_lines) > 2:
        problems.append(f"caption has {len(cap_lines)} lines (maximum 2 lines allowed)")
    for l_idx, line in enumerate(cap_lines, 1):
        if len(line) > 38:
            problems.append(f"caption line {l_idx} has {len(line)} characters (maximum 38 characters allowed)")

    # Voiceover layout limits: max 18 words
    vo_words = voiceover.split()
    if len(vo_words) > 18:
        problems.append(f"voiceover has {len(vo_words)} words (maximum 18 words allowed)")

    # Phone number in voiceover check: "never speak the phone number"
    if prop.contact:
        contact_digits = re.sub(r"\D", "", prop.contact)
        vo_digits = re.sub(r"\D", "", voiceover)
        if len(contact_digits) >= 6 and contact_digits[:6] in vo_digits:
            problems.append("voiceover speaks the phone number; phone numbers must not be spoken")
    if re.search(r"\b\d{7,}\b", voiceover):
        problems.append("voiceover speaks a phone number; phone numbers must not be spoken")

    # Fact checks
    ok_c, prob_c = check_text(caption, allowed_facts)
    for p in prob_c:
        problems.append(f"caption {_format_problem_for_prompt(p)}")

    ok_v, prob_v = check_text(voiceover, allowed_facts)
    for p in prob_v:
        problems.append(f"voiceover {_format_problem_for_prompt(p)}")

    return len(problems) == 0, problems


def _send_chat_completion(
    endpoint: str,
    api_key: str,
    messages: list[dict],
    model: str,
    temperature: float = TEMPERATURE,
) -> tuple[dict | None, str | None]:
    """Send HTTP request to OpenAI-compatible chat completions endpoint.

    Returns:
        (response_dict, None) on success
        (None, error_description) on failure
    """
    req_body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
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
            return resp_json, None
    except urllib.error.HTTPError as exc:
        err_msg = _safe_error_msg(exc)
        if exc.code == 429:
            logger.warning("LLM API rate limit exceeded (HTTP 429)")
            return None, "rate limit reached (HTTP 429)"
        logger.warning("LLM API HTTP error %d: %s", exc.code, err_msg)
        return None, f"HTTP error {exc.code} ({err_msg})"
    except (TimeoutError, urllib.error.URLError) as exc:
        if isinstance(exc, TimeoutError) or (isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError)):
            logger.warning("LLM API request timed out after %ds", REQUEST_TIMEOUT)
            return None, f"request timed out after {REQUEST_TIMEOUT}s"
        err_msg = _safe_error_msg(exc)
        logger.warning("LLM API network error: %s", err_msg)
        return None, f"network error ({err_msg})"
    except json.JSONDecodeError as exc:
        logger.warning("LLM API returned non-JSON HTTP body: %s", exc)
        return None, f"malformed HTTP response JSON: {exc}"
    except Exception as exc:
        err_msg = _safe_error_msg(exc)
        logger.warning("LLM API unexpected error: %s", err_msg)
        return None, f"unexpected error ({err_msg})"


def _extract_scenes_from_response(resp_json: dict) -> tuple[list[dict] | None, str | None, str]:
    """Extract list of scene dicts from chat completion response.

    Returns:
        (scenes_list, error_msg, raw_content_str)
    """
    try:
        content = resp_json["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        return None, f"missing choices/message structure ({exc})", ""

    # Strip code fences if model enclosed JSON in ```json ... ```
    clean_content = content
    if clean_content.startswith("```"):
        clean_content = re.sub(r"^```(?:json)?\s*", "", clean_content)
        clean_content = re.sub(r"\s*```$", "", clean_content)
        clean_content = clean_content.strip()

    try:
        parsed = json.loads(clean_content)
    except json.JSONDecodeError as exc:
        return None, f"malformed JSON: {exc}", content

    if not isinstance(parsed, dict) or "scenes" not in parsed:
        return None, "JSON missing top-level 'scenes' key", content

    raw_scenes = parsed["scenes"]
    if not isinstance(raw_scenes, list):
        return None, "'scenes' is not a list", content

    return raw_scenes, None, content


def generate_scene_copy(
    prop: PropertyData,
    scene_skeleton: list[dict],
) -> tuple[list[dict | None] | None, list[str]]:
    """Generate scene copy using an OpenAI-compatible LLM endpoint with per-scene validation and retry.

    Returns:
        (scenes_data | None, warnings)
        where scenes_data is a list of len(scene_skeleton) with:
          - dict for scenes that succeeded (initial or retry)
          - None for scenes that failed and must fall back to template text.
        If the entire call failed (missing key, network/HTTP error, timeout, bad JSON),
        scenes_data is None.
    """
    warnings: list[str] = []

    # ── 1. Check API key ───────────────────────────────────────────────────
    api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        warnings.append(
            "LLM mode requested but LLM_API_KEY is not set. "
            "Falling back to template plan."
        )
        return None, warnings

    is_gemini_key = bool(os.getenv("GEMINI_API_KEY", "").strip()) and not os.getenv("LLM_API_KEY", "").strip()
    default_base = "https://generativelanguage.googleapis.com/v1beta/openai" if is_gemini_key else DEFAULT_BASE_URL
    default_model = "gemini-2.5-flash" if is_gemini_key else DEFAULT_MODEL

    model = os.getenv("LLM_MODEL", default_model).strip() or default_model
    base_url = os.getenv("LLM_BASE_URL", default_base).strip().rstrip("/") or default_base
    endpoint = f"{base_url}/chat/completions"

    # ── 2. Build prompts sending ONLY property facts + scene skeleton ───────
    expected_count = len(scene_skeleton)
    skeleton_summary = []
    for i, s in enumerate(scene_skeleton):
        item = {"scene_index": i + 1, "scene_type": s["scene_type"]}
        if s.get("feature"):
            item["feature"] = s["feature"]
        skeleton_summary.append(item)

    facts_dict: dict[str, object] = {
        "name": prop.name,
        "location": prop.location,
        "price": prop.price,
        "property_type": prop.property_type,
        "features": prop.features,
        "cta_text": prop.cta_text,
    }
    if prop.contact:
        facts_dict["contact"] = prop.contact
    if prop.tagline:
        facts_dict["tagline"] = prop.tagline

    system_prompt = (
        "You are an assistant that writes video scripts and captions for real estate promotional videos.\n\n"
        "STRICT RULES:\n"
        "1. Use ONLY the provided property facts. Never invent any numbers, distances, amenities, features, or promises.\n"
        "2. Absolutely DO NOT use promotional buzzwords such as 'luxury', 'premium', 'guaranteed', 'best in class', "
        "'RERA approved', 'near metro', 'limited time offer', or 'investment return' unless they explicitly appear in the facts.\n"
        "3. Natural spoken English: write concise, conversational narration. Vary the wording between scenes.\n"
        "4. Caption limits: maximum 2 lines, maximum 38 characters per line. Use \\n for line breaks.\n"
        "5. Voiceover limits: maximum 18 words, short natural single sentence.\n"
        "6. Never speak the phone number in the voiceover (refer to contact details on screen instead). "
        "Phone numbers may appear on the on-screen caption.\n"
        "7. Output format: Return strictly valid JSON with no markdown wrapping and no commentary:\n"
        '{"scenes": [{"scene_index": 1, "caption": "...", "voiceover": "..."}]}'
    )

    user_prompt = (
        f"Generate copy for exactly {expected_count} scenes based ONLY on these facts:\n\n"
        f"FACTS:\n{json.dumps(facts_dict, indent=2, ensure_ascii=False)}\n\n"
        f"SCENE SKELETON:\n{json.dumps(skeleton_summary, indent=2)}\n\n"
        f"Return valid JSON with exactly {expected_count} items under 'scenes'."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # ── 3. Initial LLM Call ────────────────────────────────────────────────
    resp_json, err = _send_chat_completion(endpoint, api_key, messages, model, temperature=TEMPERATURE)
    if err:
        warnings.append(f"LLM request returned {err}. Falling back to template plan.")
        return None, warnings

    raw_scenes, parse_err, raw_content = _extract_scenes_from_response(resp_json)
    if parse_err or raw_scenes is None:
        warnings.append(f"LLM returned invalid response: {parse_err}. Falling back to template plan.")
        return None, warnings

    if len(raw_scenes) != expected_count:
        warnings.append(
            f"LLM returned {len(raw_scenes)} scenes; expected {expected_count}. "
            "Falling back to template plan."
        )
        return None, warnings

    # ── 4. Per-Scene Validation ────────────────────────────────────────────
    allowed_facts = extract_allowed_facts(prop)
    results: list[dict | None] = [None] * expected_count
    failed_scenes: list[tuple[int, dict, list[str]]] = []

    for i, item in enumerate(raw_scenes):
        if not isinstance(item, dict):
            failed_scenes.append((i, {}, ["item is not a JSON object"]))
            continue

        caption = str(item.get("caption", "")).strip()
        voiceover = str(item.get("voiceover", "")).strip()

        is_valid, problems = _validate_scene_copy(caption, voiceover, prop, allowed_facts, i + 1)
        if is_valid:
            results[i] = {
                "scene_type": scene_skeleton[i]["scene_type"],
                "caption": caption,
                "voiceover": voiceover,
            }
        else:
            failed_scenes.append((i, item, problems))

    # ── 5. Retry Once If Some Scenes Failed ────────────────────────────────
    if failed_scenes:
        problem_bullets = []
        for idx, _, probs in failed_scenes:
            prob_desc = "; ".join(probs)
            problem_bullets.append(f"- scene {idx + 1} {prob_desc}")

        failed_labels = ", ".join(f"scene {idx + 1}" for idx, _, _ in failed_scenes)
        retry_user_prompt = (
            "The following scenes failed validation:\n"
            + "\n".join(problem_bullets)
            + f"\n\nPlease provide corrected copy for ONLY the failed scenes ({failed_labels}).\n"
            "Rules:\n"
            "- Use ONLY the provided property facts (no invented numbers, distances, amenities, or promises).\n"
            "- Absolutely no prohibited buzzwords (luxury, premium, best in class, guaranteed).\n"
            "- Captions max 2 lines x 38 chars; voiceover max 18 words; never speak the phone number.\n"
            "Return strictly valid JSON with corrected scenes:\n"
            '{"scenes": [{"scene_index": <int>, "caption": "...", "voiceover": "..."}]}'
        )

        retry_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_content},
            {"role": "user", "content": retry_user_prompt},
        ]

        logger.info("Retrying %d failed LLM scene(s)...", len(failed_scenes))
        retry_resp, retry_err = _send_chat_completion(
            endpoint, api_key, retry_messages, model, temperature=TEMPERATURE
        )

        if retry_err:
            warnings.append(f"LLM retry failed ({retry_err}).")
            for idx, _, probs in failed_scenes:
                warnings.append(
                    f"Scene {idx + 1} LLM copy failed validation ({'; '.join(probs)}). "
                    "Replaced with template text."
                )
        else:
            retry_raw, retry_parse_err, _ = _extract_scenes_from_response(retry_resp)
            if retry_parse_err or not retry_raw:
                warnings.append(f"LLM retry returned invalid response: {retry_parse_err}.")
                for idx, _, probs in failed_scenes:
                    warnings.append(
                        f"Scene {idx + 1} LLM copy failed validation ({'; '.join(probs)}). "
                        "Replaced with template text."
                    )
            else:
                failed_indices = [idx for idx, _, _ in failed_scenes]
                fixed_indices = set()

                for r_idx, r_item in enumerate(retry_raw):
                    if not isinstance(r_item, dict):
                        continue

                    target_idx = None
                    if "scene_index" in r_item:
                        try:
                            s_num = int(r_item["scene_index"])
                            if (s_num - 1) in failed_indices:
                                target_idx = s_num - 1
                            elif s_num in failed_indices:
                                target_idx = s_num
                        except (ValueError, TypeError):
                            pass

                    if target_idx is None and r_idx < len(failed_indices):
                        target_idx = failed_indices[r_idx]

                    if target_idx is not None and target_idx in failed_indices:
                        r_cap = str(r_item.get("caption", "")).strip()
                        r_vo = str(r_item.get("voiceover", "")).strip()
                        r_ok, r_probs = _validate_scene_copy(
                            r_cap, r_vo, prop, allowed_facts, target_idx + 1
                        )
                        if r_ok:
                            results[target_idx] = {
                                "scene_type": scene_skeleton[target_idx]["scene_type"],
                                "caption": r_cap,
                                "voiceover": r_vo,
                            }
                            fixed_indices.add(target_idx)
                        else:
                            warnings.append(
                                f"Scene {target_idx + 1} LLM copy failed validation after retry ({'; '.join(r_probs)}). "
                                "Replaced with template text."
                            )

                for idx, _, probs in failed_scenes:
                    if idx not in fixed_indices and results[idx] is None:
                        if not any(f"Scene {idx + 1} LLM copy failed" in w for w in warnings):
                            warnings.append(
                                f"Scene {idx + 1} LLM copy failed validation ({'; '.join(probs)}). "
                                "Replaced with template text."
                            )

    # ── 6. If all scenes failed, return None so entire plan falls back ────
    if all(r is None for r in results):
        return None, warnings

    return results, warnings
