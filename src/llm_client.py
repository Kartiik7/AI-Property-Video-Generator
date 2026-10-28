"""
llm_client.py — Optional LLM client using standard library urllib.

Calls OpenAI-compatible /chat/completions endpoint to generate scene copy.
Never crashes, never leaks the API key, and strictly validates all output
per-scene through fact_checker and layout rules with a single targeted retry.
"""

import os
import json
import re
import time
import random
from dataclasses import dataclass
from urllib.parse import urlparse
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

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
AUTH_ERROR_CODES = {401, 403}


@dataclass
class LLMProvider:
    name: str                 # "primary" or "fallback: openrouter"
    api_key: str
    base_url: str
    model: str
    key_env_var: str          # "LLM_API_KEY" or "LLM_FALLBACK_API_KEY"
    is_primary: bool


def _get_provider_label(base_url: str, is_primary: bool) -> str:
    """Derive human-friendly provider name for plan source tracking."""
    if is_primary:
        return "primary"
    if not base_url:
        return "fallback"
    try:
        netloc = urlparse(base_url).netloc.lower() or base_url.lower()
        host = netloc.split(":")[0]
        ignore = {"api", "www", "com", "ai", "io", "org", "co", "dev", "net", "app", "v1", "http", "https"}
        parts = [p for p in host.split(".") if p not in ignore]
        if parts:
            return f"fallback: {parts[0]}"
        return "fallback"
    except Exception:
        return "fallback"


def _get_configured_providers() -> list[LLMProvider]:
    """Return ordered list of configured providers: primary first, then fallback."""
    providers: list[LLMProvider] = []

    # Primary provider
    primary_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if primary_key:
        is_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip()) and not os.getenv("LLM_API_KEY", "").strip()
        default_base = "https://generativelanguage.googleapis.com/v1beta/openai" if is_gemini else DEFAULT_BASE_URL
        default_model = "gemini-2.5-flash" if is_gemini else DEFAULT_MODEL
        base_url = os.getenv("LLM_BASE_URL", default_base).strip().rstrip("/") or default_base
        model = os.getenv("LLM_MODEL", default_model).strip() or default_model
        key_var = "GEMINI_API_KEY" if is_gemini else "LLM_API_KEY"
        providers.append(
            LLMProvider(
                name="primary",
                api_key=primary_key,
                base_url=base_url,
                model=model,
                key_env_var=key_var,
                is_primary=True,
            )
        )

    # Fallback provider
    fallback_key = os.getenv("LLM_FALLBACK_API_KEY", "").strip()
    if fallback_key:
        fallback_base = os.getenv("LLM_FALLBACK_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/") or DEFAULT_BASE_URL
        fallback_model = os.getenv("LLM_FALLBACK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        label = _get_provider_label(fallback_base, is_primary=False)
        providers.append(
            LLMProvider(
                name=label,
                api_key=fallback_key,
                base_url=fallback_base,
                model=fallback_model,
                key_env_var="LLM_FALLBACK_API_KEY",
                is_primary=False,
            )
        )

    return providers


def _safe_error_msg(exc: Exception) -> str:
    """Format exception message without leaking any potential API keys."""
    msg = str(exc)
    msg = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", msg)
    for env_var in ("LLM_API_KEY", "GEMINI_API_KEY", "LLM_FALLBACK_API_KEY"):
        key = os.getenv(env_var, "").strip()
        if key and len(key) >= 4:
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


def _send_single_http_request(
    endpoint: str,
    api_key: str,
    messages: list[dict],
    model: str,
    temperature: float = TEMPERATURE,
) -> tuple[dict | None, int | None, str | None]:
    """Execute a single HTTP request to an OpenAI-compatible endpoint.

    Returns:
        (response_dict, status_code, error_message)
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
            return resp_json, 200, None
    except urllib.error.HTTPError as exc:
        err_msg = _safe_error_msg(exc)
        if exc.code == 429:
            logger.warning("LLM API rate limit exceeded (HTTP 429)")
            return None, 429, "rate limit reached (HTTP 429)"
        logger.warning("LLM API HTTP error %d: %s", exc.code, err_msg)
        return None, exc.code, f"HTTP error {exc.code} ({err_msg})"
    except (TimeoutError, urllib.error.URLError) as exc:
        if isinstance(exc, TimeoutError) or (isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError)):
            logger.warning("LLM API request timed out after %ds", REQUEST_TIMEOUT)
            return None, None, f"request timed out after {REQUEST_TIMEOUT}s"
        err_msg = _safe_error_msg(exc)
        logger.warning("LLM API network error: %s", err_msg)
        return None, None, f"network error ({err_msg})"
    except json.JSONDecodeError as exc:
        logger.warning("LLM API returned non-JSON HTTP body: %s", exc)
        return None, None, f"malformed HTTP response JSON: {exc}"
    except Exception as exc:
        err_msg = _safe_error_msg(exc)
        logger.warning("LLM API unexpected error: %s", err_msg)
        return None, None, f"unexpected error ({err_msg})"


def _send_chat_completion(
    endpoint: str,
    api_key: str,
    messages: list[dict],
    model: str,
    temperature: float = TEMPERATURE,
) -> tuple[dict | None, str | None]:
    """Single request helper returning (response_dict, error_msg)."""
    resp_json, _, err_msg = _send_single_http_request(
        endpoint, api_key, messages, model, temperature
    )
    return resp_json, err_msg


def _call_provider_with_retry(
    provider: LLMProvider,
    messages: list[dict],
    warnings: list[str],
) -> tuple[dict | None, str | None]:
    """Call a provider with at most ONE retry on 429, 500, 502, 503, 504, timeout, or network error.

    Do not retry on 401/403 (bad key); move on and warn "check LLM_API_KEY".
    """
    endpoint = f"{provider.base_url}/chat/completions"

    resp_json, status_code, err_msg = _send_single_http_request(
        endpoint, provider.api_key, messages, provider.model, TEMPERATURE
    )

    if resp_json is not None:
        return resp_json, None

    # 401 / 403: Bad key. Do NOT retry! Move on and warn "check LLM_API_KEY" (or key_env_var)
    if status_code in AUTH_ERROR_CODES:
        auth_warning = f"LLM API authentication failed ({status_code}); check {provider.key_env_var}."
        logger.warning(auth_warning)
        warnings.append(auth_warning)
        return None, auth_warning

    # Check if retryable: 429, 500, 502, 503, 504, timeout, or network error
    is_retryable = False
    if status_code in RETRYABLE_STATUS_CODES:
        is_retryable = True
    elif status_code is None and err_msg and ("timed out" in err_msg or "network error" in err_msg):
        is_retryable = True

    if not is_retryable:
        # Other non-retryable error
        err_desc = f"LLM API returned {err_msg}."
        logger.warning(err_desc)
        warnings.append(err_desc)
        return None, err_desc

    # Retry the same provider ONCE after a short backoff (2 s, plus a small random jitter)
    backoff = 2.0 + random.uniform(0.1, 0.5)
    logger.warning("LLM API %s returned %s. Retrying once after %.2fs...", provider.name, err_msg, backoff)
    time.sleep(backoff)

    resp_json_retry, status_code_retry, err_msg_retry = _send_single_http_request(
        endpoint, provider.api_key, messages, provider.model, TEMPERATURE
    )

    if resp_json_retry is not None:
        logger.info("LLM API %s retry succeeded.", provider.name)
        return resp_json_retry, None

    retry_failure = f"LLM API {provider.name} failed after retry ({err_msg_retry or err_msg})."
    logger.warning(retry_failure)
    warnings.append(retry_failure)
    return None, retry_failure


def _clean_and_extract_json(content: str) -> tuple[dict | None, str | None]:
    """Strip markdown code fences, leading thinking text, and extract first {...} block."""
    if not content or not content.strip():
        return None, "empty content"

    text = content.strip()

    # 1. Strip thinking tags: <think>...</think>, <thought>...</thought>
    text = re.sub(r"(?is)<think(?:ing)?>.*?</think(?:ing)?>", "", text).strip()
    text = re.sub(r"(?is)<thought>.*?</thought>", "", text).strip()

    # 2. Extract content from markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        fenced_text = fence_match.group(1).strip()
        try:
            parsed = json.loads(fenced_text)
            if isinstance(parsed, dict):
                return parsed, None
        except json.JSONDecodeError:
            text = fenced_text

    # 3. Direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, None
    except json.JSONDecodeError:
        pass

    # 4. Extract first balanced {...} block
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed, None
                        except json.JSONDecodeError as exc:
                            return None, f"extracted JSON object is invalid ({exc})"
                        break

    return None, "no valid JSON object found in response"


def _extract_scenes_from_response(resp_json: dict) -> tuple[list[dict] | None, str | None, str]:
    """Extract list of scene dicts from chat completion response.

    Returns:
        (scenes_list, error_msg, raw_content_str)
    """
    try:
        content = resp_json["choices"][0]["message"]["content"]
        if content is None:
            content = ""
        content = str(content).strip()
    except (KeyError, IndexError, TypeError) as exc:
        return None, f"missing choices/message structure ({exc})", ""

    parsed, parse_err = _clean_and_extract_json(content)
    if parse_err or parsed is None:
        return None, f"malformed JSON: {parse_err}", content

    if "scenes" not in parsed:
        return None, "JSON missing top-level 'scenes' key", content

    raw_scenes = parsed["scenes"]
    if not isinstance(raw_scenes, list):
        return None, "'scenes' is not a list", content

    return raw_scenes, None, content

def generate_scene_copy(
    prop: PropertyData,
    scene_skeleton: list[dict],
) -> tuple[list[dict | None] | None, list[str], str | None]:
    """Generate scene copy using OpenAI-compatible LLM endpoint(s) with per-scene validation and retry.

    Call order: primary provider -> fallback provider -> template.
    Returns:
        (scenes_data | None, warnings, answering_provider_name | None)
        where scenes_data is a list of len(scene_skeleton) with:
          - dict for scenes that succeeded (initial or retry)
          - None for scenes that failed and must fall back to template text.
        If all providers fail, scenes_data is None.
    """
    warnings: list[str] = []

    # ── 1. Check configured providers ──────────────────────────────────────
    providers = _get_configured_providers()
    if not providers:
        warnings.append(
            "LLM mode requested but LLM_API_KEY is not set. "
            "Falling back to template plan."
        )
        return None, warnings, None

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

    # ── 3. Call order: primary provider -> fallback provider -> template ──
    answering_provider: LLMProvider | None = None
    raw_scenes: list[dict] | None = None
    raw_content: str = ""

    for provider in providers:
        resp_json, call_err = _call_provider_with_retry(provider, messages, warnings)
        if not resp_json:
            continue

        scenes_list, parse_err, content_str = _extract_scenes_from_response(resp_json)
        if parse_err or scenes_list is None:
            msg = f"LLM {provider.name} provider returned invalid response: {parse_err}."
            logger.warning(msg)
            warnings.append(msg)
            continue

        if len(scenes_list) != expected_count:
            msg = f"LLM {provider.name} provider returned {len(scenes_list)} scenes; expected {expected_count}."
            logger.warning(msg)
            warnings.append(msg)
            continue

        # Provider answered successfully!
        answering_provider = provider
        raw_scenes = scenes_list
        raw_content = content_str
        break

    if answering_provider is None or raw_scenes is None:
        warnings.append("All LLM providers failed. Falling back to template plan.")
        return None, warnings, None

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

    # ── 5. Retry Once If Some Scenes Failed (Using answering provider) ─────
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

        logger.info("Retrying %d failed LLM scene(s) using %s...", len(failed_scenes), answering_provider.name)
        retry_endpoint = f"{answering_provider.base_url}/chat/completions"
        retry_resp, retry_err = _send_chat_completion(
            retry_endpoint,
            answering_provider.api_key,
            retry_messages,
            answering_provider.model,
            temperature=TEMPERATURE,
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
        return None, warnings, None

    return results, warnings, answering_provider.name

