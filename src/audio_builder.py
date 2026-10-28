"""
audio_builder.py — TTS voiceover narration, audio sync, and background music.

Features:
  - edge-tts synthesis with thread-safe event loop execution (Streamlit & sync safe)
  - Voice verification with fallback: en-IN-NeerjaNeural -> en-IN fallback -> en-US-JennyNeural
  - Spoken text normalization (24/7, &, BHK, sq ft, rupee symbols, phone number stripping)
  - SHA256 disk caching in output/.cache/tts for rapid rebuilds
  - Timeline sync: scene_duration = max(planned, narration + 0.7s)
  - Speed-up optimization (+12% rate limit) and pause trimming when timeline exceeds 40s
  - Background music mixing: looping/trimming, 1s fade-in, 1.5s fade-out, ducking / volume scaling
  - Graceful degradation: narration or music failures log clear warnings and continue
"""

import os
import re
import shutil
import hashlib
import asyncio
import concurrent.futures
import base64
import json
import wave
import time
import random
import urllib.request
import urllib.error
from pathlib import Path
from typing import Callable, Any
from dotenv import load_dotenv

import edge_tts
from moviepy import (
    AudioClip,
    AudioFileClip,
    CompositeAudioClip,
    afx,
)

from src.models import ScenePlan, Scene
from src.logger import setup_logger

load_dotenv()
logger = setup_logger()

DEFAULT_EDGE_VOICE = "en-IN-NeerjaNeural"
DEFAULT_VOICE = DEFAULT_EDGE_VOICE
DEFAULT_FALLBACK_VOICE = "en-US-JennyNeural"

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash-lite-tts"
DEFAULT_GEMINI_VOICE = "Kore"
DEFAULT_GEMINI_STYLE = "warm, clear, professional real-estate narrator, moderate pace"

TTS_TIMEOUT = 20  # seconds
CACHE_DIR = Path("output/.cache/tts")

_AVAILABLE_VOICES_CACHE: list[str] | None = None


def _safe_error_msg(exc: Exception) -> str:
    """Format exception message without leaking any potential API keys."""
    msg = str(exc)
    msg = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", msg)
    for env_var in ("GEMINI_API_KEY", "LLM_API_KEY", "LLM_FALLBACK_API_KEY"):
        key = os.getenv(env_var, "").strip()
        if key and len(key) >= 4:
            msg = msg.replace(key, "[REDACTED]")
    return msg


def get_gemini_api_key() -> str:
    """Resolve Gemini API key for TTS.

    Uses GEMINI_API_KEY if present. If missing, reuses LLM_API_KEY
    only when LLM_BASE_URL contains 'generativelanguage.googleapis.com'.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    base_url = os.getenv("LLM_BASE_URL", "").strip().lower()
    if "generativelanguage.googleapis.com" in base_url:
        llm_key = os.getenv("LLM_API_KEY", "").strip()
        if llm_key:
            return llm_key
    return ""


def resolve_tts_provider(requested_provider: str | None = None) -> str:
    """Resolve TTS provider: 'gemini' or 'edge'.

    In 'auto' mode (default): 'gemini' if a Gemini key exists, else 'edge'.
    """
    prov = (requested_provider or os.getenv("TTS_PROVIDER", "auto")).strip().lower()
    if prov == "gemini":
        return "gemini"
    if prov == "edge":
        return "edge"
    # Auto mode:
    return "gemini" if bool(get_gemini_api_key()) else "edge"


def get_tts_cache_key(provider: str, model: str, voice: str, style: str, text: str) -> str:
    """Generate SHA256 cache key from provider, model, voice, style, and text."""
    raw = f"{provider}|{model}|{voice}|{style}|{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ── Text Normalization for TTS ─────────────────────────────────────────────

def prepare_spoken_text(text: str) -> str:
    """Normalize text for speech synthesis according to TTS pronunciation rules.

    Transformations:
      - Never speak phone numbers (sequences of 8+ digits / contact formats stripped)
      - "24/7" -> "24 by 7"
      - "&" -> "and"
      - "BHK" -> "B H K" (e.g. "2BHK" -> "2 B H K")
      - "sq ft" / "sqft" / "sq.ft." -> "square feet"
      - "₹" / "Rs." -> "rupees"
      - Collapse whitespace
    """
    if not text:
        return ""

    s = text

    # 1. Strip phone numbers (formatted with country codes/dashes or raw 8+ digits)
    s = re.sub(r"\+?\d[\d\s\-]{6,}\d", "", s)
    s = re.sub(r"\b\d{8,}\b", "", s)

    # 2. "24/7" -> "24 by 7"
    s = re.sub(r"\b24/7\b", "24 by 7", s, flags=re.IGNORECASE)

    # 3. "&" -> "and"
    s = re.sub(r"&", " and ", s)

    # 4. BHK expansions (e.g. 2BHK -> 2 B H K, 3 BHK -> 3 B H K, BHK -> B H K)
    s = re.sub(r"(\d+)\s*BHK\b", r"\1 B H K", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBHK\b", "B H K", s, flags=re.IGNORECASE)

    # 5. Square feet expansions
    s = re.sub(r"\bsq\.?\s*ft(?:\.|\b)", "square feet", s, flags=re.IGNORECASE)
    s = re.sub(r"\bsqft\b", "square feet", s, flags=re.IGNORECASE)

    # 6. Currency symbols to spoken words
    s = re.sub(r"₹\s*", "rupees ", s)
    s = re.sub(r"\bRs\.?\s*", "rupees ", s, flags=re.IGNORECASE)

    # 7. Collapse multi-spaces and strip edges
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── Thread-Safe Async Runner ───────────────────────────────────────────────

def _run_async_in_thread(coro_fn: Callable[[], Any], timeout: float = 25.0) -> Any:
    """Run an async coroutine inside a dedicated thread with a fresh event loop.

    Safe to call from sync CLI commands and from within Streamlit / Jupyter.
    """
    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro_fn())
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_worker)
        return future.result(timeout=timeout)


# ── Voice List Resolution ──────────────────────────────────────────────────

def get_available_voices() -> list[str]:
    """Retrieve list of available voices from edge-tts, cached in memory."""
    global _AVAILABLE_VOICES_CACHE
    if _AVAILABLE_VOICES_CACHE is not None:
        return _AVAILABLE_VOICES_CACHE

    try:
        voices_data = _run_async_in_thread(lambda: edge_tts.list_voices(), timeout=10.0)
        _AVAILABLE_VOICES_CACHE = [v["ShortName"] for v in voices_data if "ShortName" in v]
    except Exception as exc:
        logger.warning("Could not fetch edge-tts voice list: %s; using default fallbacks.", exc)
        _AVAILABLE_VOICES_CACHE = [DEFAULT_VOICE, "en-IN-PrabhatNeural", DEFAULT_FALLBACK_VOICE]

    return _AVAILABLE_VOICES_CACHE


def resolve_voice(requested_voice: str | None = None) -> str:
    """Verify requested voice exists; fall back to en-IN voice, then en-US-JennyNeural."""
    target = (requested_voice or DEFAULT_VOICE).strip()
    available = get_available_voices()

    if target in available:
        return target

    logger.warning("Voice '%s' not found in available voices.", target)

    # Fallback to another en-IN voice
    for v in available:
        if v.startswith("en-IN-"):
            logger.info("Falling back to voice '%s'", v)
            return v

    # Fallback to en-US-JennyNeural
    if DEFAULT_FALLBACK_VOICE in available:
        logger.info("Falling back to '%s'", DEFAULT_FALLBACK_VOICE)
        return DEFAULT_FALLBACK_VOICE

    return available[0] if available else DEFAULT_VOICE


# ── TTS Synthesis with Caching ─────────────────────────────────────────────

def _format_rate_str(rate_percent: int) -> str:
    """Format integer percentage into edge-tts rate string (e.g. +12% -> '+12%')."""
    if rate_percent >= 0:
        return f"+{rate_percent}%"
    return f"{rate_percent}%"


def synthesize_gemini_voiceover(
    text: str,
    out_path: Path | str,
    voice: str | None = None,
    model: str | None = None,
    style: str = DEFAULT_GEMINI_STYLE,
    api_key: str | None = None,
    use_cache: bool = True,
) -> Path:
    """Synthesize voiceover audio using Google Gemini TTS endpoint via urllib.

    Args:
        text: Raw or normalized voiceover text (treated as verbatim transcript).
        out_path: Destination file path (saved as .wav).
        voice: Gemini voice name (default: env TTS_GEMINI_VOICE or Kore).
        model: Gemini model ID (default: env TTS_GEMINI_MODEL or gemini-3.8-flash-lite-tts).
        style: Speech delivery style for speech_metadata/style field.
        api_key: Explicit API key or resolved from env.
        use_cache: Whether to check/store output/.cache/tts.

    Returns:
        Path to output WAV file.
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spoken_text = prepare_spoken_text(text)
    if not spoken_text:
        raise ValueError("Cannot synthesize voiceover for empty text.")

    key = api_key or get_gemini_api_key()
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    model_name = (model or os.getenv("TTS_GEMINI_MODEL", DEFAULT_GEMINI_MODEL)).strip() or DEFAULT_GEMINI_MODEL
    voice_name = (voice or os.getenv("TTS_GEMINI_VOICE", DEFAULT_GEMINI_VOICE)).strip() or DEFAULT_GEMINI_VOICE

    # 1. Check cache by hash(provider + model + voice + style + text)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_key = get_tts_cache_key("gemini", model_name, voice_name, style, spoken_text)
        cached_file = CACHE_DIR / f"{cache_key}.wav"
        if cached_file.exists() and cached_file.stat().st_size > 0:
            shutil.copyfile(cached_file, out_path)
            return out_path

    # 2. Build JSON payload exactly as documented:
    # Verbatim text in parts[].text; delivery style in parts[].speech_metadata.style
    req_body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": spoken_text,
                        "speech_metadata": {
                            "style": style,
                        },
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice_name,
                    }
                }
            },
        },
    }
    req_bytes = json.dumps(req_body).encode("utf-8")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    # 3. HTTP Request with single retry on transient error (429, 5xx, timeout, network error)
    last_err: Exception | None = None
    for attempt in range(2):
        req = urllib.request.Request(
            endpoint,
            data=req_bytes,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=TTS_TIMEOUT) as response:
                resp_bytes = response.read()
                resp_json = json.loads(resp_bytes.decode("utf-8"))

            candidates = resp_json.get("candidates", [])
            if not candidates:
                raise ValueError("Gemini TTS response missing candidates.")
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError("Gemini TTS candidate missing content parts.")

            inline_data = parts[0].get("inlineData") or parts[0].get("inline_data")
            if not inline_data or "data" not in inline_data:
                raise ValueError("Gemini TTS response missing inlineData audio.")

            b64_audio = inline_data["data"]
            pcm_bytes = base64.b64decode(b64_audio)
            if not pcm_bytes:
                raise ValueError("Gemini TTS returned empty audio payload.")

            mime_type = inline_data.get("mimeType", "")
            sample_rate = 24000
            m_rate = re.search(r"rate=(\d+)", mime_type)
            if m_rate:
                try:
                    sample_rate = int(m_rate.group(1))
                except ValueError:
                    pass

            if pcm_bytes.startswith(b"RIFF"):
                out_path.write_bytes(pcm_bytes)
            else:
                with wave.open(str(out_path), "wb") as wf:
                    wf.setnchannels(1)      # mono
                    wf.setsampwidth(2)      # 16-bit = 2 bytes
                    wf.setframerate(sample_rate)
                    wf.writeframes(pcm_bytes)

            if out_path.exists() and out_path.stat().st_size > 0:
                if use_cache:
                    shutil.copyfile(out_path, cached_file)
                return out_path
            raise ValueError("Gemini TTS produced 0 bytes audio file.")

        except urllib.error.HTTPError as exc:
            safe_err = _safe_error_msg(exc)
            last_err = RuntimeError(f"HTTP {exc.code}: {safe_err}")
            if exc.code in (401, 403):
                logger.warning("Gemini TTS authentication failed (HTTP %d); check GEMINI_API_KEY.", exc.code)
                break
            if attempt == 0 and (exc.code == 429 or exc.code >= 500):
                backoff = 2.0 + random.uniform(0.1, 0.5)
                logger.warning("Gemini TTS transient HTTP %d error. Retrying once after %.2fs...", exc.code, backoff)
                time.sleep(backoff)
                continue
            break
        except (TimeoutError, urllib.error.URLError) as exc:
            safe_err = _safe_error_msg(exc)
            last_err = RuntimeError(f"Network error: {safe_err}")
            if attempt == 0:
                backoff = 2.0 + random.uniform(0.1, 0.5)
                logger.warning("Gemini TTS network timeout/error. Retrying once after %.2fs...", backoff)
                time.sleep(backoff)
                continue
            break
        except Exception as exc:
            safe_err = _safe_error_msg(exc)
            last_err = RuntimeError(safe_err)
            if attempt == 0:
                backoff = 2.0 + random.uniform(0.1, 0.5)
                logger.warning("Gemini TTS error: %s. Retrying once after %.2fs...", safe_err, backoff)
                time.sleep(backoff)
                continue
            break

    raise RuntimeError(f"Gemini TTS synthesis failed: {last_err}")


def synthesize_edge_voiceover(
    text: str,
    out_path: Path | str,
    voice: str = DEFAULT_VOICE,
    rate: int = 0,
    use_cache: bool = True,
) -> Path:
    """Synthesize voiceover audio using edge-tts.

    Args:
        text: Raw or normalized voiceover text.
        out_path: Destination file path (e.g. .mp3).
        voice: edge-tts voice ShortName.
        rate: Speed percentage change (-50 to +50).
        use_cache: Check and save to output/.cache/tts.

    Returns:
        Path to generated audio file.
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spoken_text = prepare_spoken_text(text)
    if not spoken_text:
        raise ValueError("Cannot synthesize voiceover for empty text.")

    rate_str = _format_rate_str(rate)
    resolved_v = resolve_voice(voice)

    # 1. Check cache by hash(provider + model + voice + style + text)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_key = get_tts_cache_key("edge", "edge-tts", resolved_v, rate_str, spoken_text)
        cached_file = CACHE_DIR / f"{cache_key}.mp3"

        if cached_file.exists() and cached_file.stat().st_size > 0:
            shutil.copyfile(cached_file, out_path)
            return out_path

    # 2. Async synthesis with 20s timeout and 1 retry
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            async def _do_synth():
                communicate = edge_tts.Communicate(spoken_text, voice=resolved_v, rate=rate_str)
                await communicate.save(str(out_path))

            _run_async_in_thread(_do_synth, timeout=TTS_TIMEOUT)

            if out_path.exists() and out_path.stat().st_size > 0:
                if use_cache:
                    shutil.copyfile(out_path, cached_file)
                return out_path
        except Exception as exc:
            last_err = exc
            logger.warning("TTS attempt %d failed for voice '%s': %s", attempt + 1, resolved_v, exc)

    raise RuntimeError(f"edge-tts synthesis failed after retry: {last_err}")


def synthesize_voiceover(
    text: str,
    out_path: Path | str,
    voice: str | None = None,
    rate: int = 0,
    use_cache: bool = True,
    provider: str = "edge",
    model: str | None = None,
    style: str = DEFAULT_GEMINI_STYLE,
) -> Path:
    """General dispatcher for voiceover synthesis."""
    if provider == "gemini":
        return synthesize_gemini_voiceover(
            text=text,
            out_path=out_path,
            voice=voice,
            model=model,
            style=style,
            use_cache=use_cache,
        )
    return synthesize_edge_voiceover(
        text=text,
        out_path=out_path,
        voice=voice or DEFAULT_VOICE,
        rate=rate,
        use_cache=use_cache,
    )


# ── Scene Timeline Recalculation & Narration Assembly ──────────────────────

def build_audio_tracks(
    plan: ScenePlan,
    temp_dir: Path,
    voice: str | None = None,
    tts_provider: str | None = None,
    base_rate: int = 0,
    music_path: Path | str | None = None,
    no_audio: bool = False,
    no_music: bool = False,
    no_narration: bool = False,
    warnings: list[str] | None = None,
) -> tuple[AudioClip | None, list[float]]:
    """Synthesize voiceovers for all scenes, sync timeline durations, and mix music.

    Maintains voice consistency: never mixes voices within one video.
    If Gemini fails for ANY scene, retries once with backoff; if it still fails,
    regenerates ALL scenes with edge-tts and records ONE warning.
    If both fail, falls back to silent video.

    Returns:
        (composite_audio_or_none, updated_scene_durations)
    """
    if warnings is None:
        warnings = plan.warnings

    num_scenes = len(plan.scenes)
    original_durations = [s.duration for s in plan.scenes]

    temp_dir.mkdir(parents=True, exist_ok=True)
    narration_files: list[Path | None] = [None] * num_scenes
    narration_durations: list[float] = [0.0] * num_scenes

    chosen_provider = resolve_tts_provider(tts_provider)
    actual_provider = "none"
    actual_voice = "none"
    actual_model = "none"

    if no_audio or (no_music and no_narration):
        plan.metadata["tts_provider"] = "none"
        plan.metadata["tts_voice"] = "none"
        plan.metadata["tts_model"] = "none"
        return None, original_durations

    tts_failed = False

    # ── 1. Synthesize narration for each scene ──────────────────────────────
    if not no_narration:
        if chosen_provider == "gemini":
            gemini_model = os.getenv("TTS_GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
            gemini_voice = voice or os.getenv("TTS_GEMINI_VOICE", DEFAULT_GEMINI_VOICE).strip() or DEFAULT_GEMINI_VOICE
            gemini_failed = False
            gemini_err_str = ""

            for i, scene in enumerate(plan.scenes):
                if not scene.voiceover.strip():
                    continue
                audio_file = temp_dir / f"scene_{i+1}_vo_gemini.wav"
                try:
                    synthesize_gemini_voiceover(
                        text=scene.voiceover,
                        out_path=audio_file,
                        voice=gemini_voice,
                        model=gemini_model,
                    )
                    with AudioFileClip(str(audio_file)) as clip:
                        dur = float(clip.duration)
                    narration_files[i] = audio_file
                    narration_durations[i] = dur
                except Exception as exc:
                    gemini_failed = True
                    gemini_err_str = _safe_error_msg(exc)
                    logger.warning("Gemini TTS failed for scene %d: %s. Falling back to edge-tts for all scenes.", i + 1, gemini_err_str)
                    break

            if gemini_failed:
                # Add ONE warning for Gemini failure and regenerate ALL scenes with edge-tts
                warn_msg = f"Gemini TTS failed ({gemini_err_str}); regenerated all narration with edge-tts."
                if not any("Gemini TTS failed" in w for w in warnings):
                    warnings.append(warn_msg)

                narration_files = [None] * num_scenes
                narration_durations = [0.0] * num_scenes

                edge_voice = resolve_voice(voice if (voice and voice.startswith("en-")) else None)
                edge_failed = False
                for i, scene in enumerate(plan.scenes):
                    if not scene.voiceover.strip():
                        continue
                    audio_file = temp_dir / f"scene_{i+1}_vo_edge.mp3"
                    try:
                        synthesize_voiceover(
                            text=scene.voiceover,
                            out_path=audio_file,
                            voice=edge_voice,
                            rate=base_rate,
                        )
                        with AudioFileClip(str(audio_file)) as clip:
                            dur = float(clip.duration)
                        narration_files[i] = audio_file
                        narration_durations[i] = dur
                    except Exception as exc:
                        edge_failed = True
                        logger.warning("edge-tts fallback failed for scene %d: %s", i + 1, exc)
                        break

                if edge_failed:
                    warn_silent = "Narration unavailable; exporting without narration."
                    if warn_silent not in warnings:
                        warnings.append(warn_silent)
                    narration_files = [None] * num_scenes
                    narration_durations = [0.0] * num_scenes
                    tts_failed = True
                    actual_provider = "none"
                    actual_voice = "none"
                    actual_model = "none"
                else:
                    actual_provider = "edge"
                    actual_voice = edge_voice
                    actual_model = "edge-tts"
            else:
                actual_provider = "gemini"
                actual_voice = gemini_voice
                actual_model = gemini_model

        else:  # chosen_provider == "edge"
            edge_voice = resolve_voice(voice)
            edge_failed = False
            for i, scene in enumerate(plan.scenes):
                if not scene.voiceover.strip():
                    continue
                audio_file = temp_dir / f"scene_{i+1}_vo_edge.mp3"
                try:
                    synthesize_voiceover(
                        text=scene.voiceover,
                        out_path=audio_file,
                        voice=edge_voice,
                        rate=base_rate,
                    )
                    with AudioFileClip(str(audio_file)) as clip:
                        dur = float(clip.duration)
                    narration_files[i] = audio_file
                    narration_durations[i] = dur
                except Exception as exc:
                    edge_failed = True
                    logger.warning("edge-tts generation failed for scene %d: %s", i + 1, exc)
                    break

            if edge_failed:
                warn_msg = "Narration unavailable; exporting without narration."
                if warn_msg not in warnings:
                    warnings.append(warn_msg)
                narration_files = [None] * num_scenes
                narration_durations = [0.0] * num_scenes
                tts_failed = True
                actual_provider = "none"
                actual_voice = "none"
                actual_model = "none"
            else:
                actual_provider = "edge"
                actual_voice = edge_voice
                actual_model = "edge-tts"

    plan.metadata["tts_provider"] = actual_provider
    plan.metadata["tts_voice"] = actual_voice
    plan.metadata["tts_model"] = actual_model

    # ── 2. Sync scene durations ─────────────────────────────────────────────
    # Base formula: scene duration = max(planned duration, narration length + 0.7 s)
    scene_durations: list[float] = []
    for i, scene in enumerate(plan.scenes):
        vo_len = narration_durations[i]
        if vo_len > 0:
            d = max(scene.duration, vo_len + 0.7)
        else:
            d = scene.duration
        scene_durations.append(round(d, 2))

    total_dur = sum(scene_durations)

    # ── 3. Duration Budget Management (Hard Cap 45s, Target <= 40s) ────────
    if total_dur > 40.0 and any(d > 0 for d in narration_durations):
        # Step A: Speed up TTS rate up to +12%
        extra_speed = min(12, int(round((total_dur - 40.0) / total_dur * 100)) + 4)
        target_rate = min(base_rate + extra_speed, 12)

        if target_rate > base_rate and not tts_failed:
            logger.info("Total duration %.1fs exceeds 40s. Re-synthesizing TTS at rate +%d%%", total_dur, target_rate)
            for i, scene in enumerate(plan.scenes):
                if narration_durations[i] > 0:
                    audio_file = temp_dir / f"scene_{i+1}_vo_fast.mp3"
                    try:
                        synthesize_voiceover(
                            text=scene.voiceover,
                            out_path=audio_file,
                            voice=voice,
                            rate=target_rate,
                        )
                        with AudioFileClip(str(audio_file)) as clip:
                            fast_dur = float(clip.duration)
                        narration_files[i] = audio_file
                        narration_durations[i] = fast_dur
                    except Exception:
                        pass

            # Recalculate scene durations after speed-up
            scene_durations = [
                round(max(plan.scenes[i].duration, narration_durations[i] + 0.7), 2)
                if narration_durations[i] > 0 else round(plan.scenes[i].duration, 2)
                for i in range(num_scenes)
            ]
            total_dur = sum(scene_durations)

        # Step B: Trim pauses if still > 40s (never cut speech: min padding narration + 0.4s)
        if total_dur > 40.0:
            excess = total_dur - 40.0
            new_durations = []
            for i in range(num_scenes):
                vo_len = narration_durations[i]
                current_d = scene_durations[i]
                if vo_len > 0:
                    min_allowed = vo_len + 0.4
                    can_trim = max(0.0, current_d - min_allowed)
                    trim_amt = min(can_trim, excess / num_scenes)
                    new_durations.append(round(current_d - trim_amt, 2))
                else:
                    new_durations.append(current_d)
            scene_durations = new_durations
            total_dur = sum(scene_durations)

        # Step C: Hard cap at 45.0s with warning
        if total_dur > 45.0:
            warnings.append(f"Total video duration ({total_dur:.1f}s) exceeded 45s limit; capped at 45.0s.")
            scale = 45.0 / total_dur
            scene_durations = [round(d * scale, 2) for d in scene_durations]
            total_dur = sum(scene_durations)

    # Update plan.scenes durations in place
    for i, s in enumerate(plan.scenes):
        s.duration = scene_durations[i]

    # ── 4. Build Narration Composite Audio Clip ─────────────────────────────
    audio_clips_to_mix: list[AudioClip] = []
    current_time = 0.0

    has_narration = False
    for i, afile in enumerate(narration_files):
        sc_dur = scene_durations[i]
        if afile is not None and afile.exists():
            try:
                # Narration starts 0.3s after scene's visible start
                vo_clip = AudioFileClip(str(afile))
                start_offset = current_time + 0.3
                vo_clip = vo_clip.with_start(start_offset)
                audio_clips_to_mix.append(vo_clip)
                has_narration = True
            except Exception as exc:
                logger.warning("Could not load narration clip %s: %s", afile, exc)
        current_time += sc_dur

    narration_track: AudioClip | None = None
    if audio_clips_to_mix:
        narration_track = CompositeAudioClip(audio_clips_to_mix)

    # ── 5. Background Music Mixing ──────────────────────────────────────────
    music_track: AudioClip | None = None
    if not no_music:
        resolved_music = _resolve_music_file(music_path, warnings)
        if resolved_music:
            music_track = load_background_music(
                music_file=resolved_music,
                target_duration=plan.total_duration,
                has_narration=has_narration,
                warnings=warnings,
            )

    # ── 6. Combine Narration + Music ────────────────────────────────────────
    final_audio: AudioClip | None = None
    if narration_track and music_track:
        final_audio = CompositeAudioClip([music_track, narration_track])
    elif narration_track:
        final_audio = narration_track
    elif music_track:
        final_audio = music_track

    return final_audio, scene_durations


def load_background_music(
    music_file: Path | str,
    target_duration: float,
    has_narration: bool = False,
    warnings: list[str] | None = None,
) -> AudioClip | None:
    """Load, loop or trim, duck, and fade background music safely.

    Returns AudioClip or None on failure, appending a clean warning.
    """
    if warnings is None:
        warnings = []
    try:
        raw_music = AudioFileClip(str(music_file))
        vol_factor = 0.10 if has_narration else 0.18

        if raw_music.duration < target_duration:
            looped = raw_music.with_effects([afx.AudioLoop(duration=target_duration)])
        else:
            looped = raw_music.subclipped(0, target_duration)

        return looped.with_effects([
            afx.AudioFadeIn(1.0),
            afx.AudioFadeOut(1.5),
            afx.MultiplyVolume(vol_factor),
        ])
    except Exception as exc:
        logger.warning("Could not process background music file '%s': %s", music_file, exc)
        warn_msg = f"Could not load background music ({exc}); continuing without music."
        if warn_msg not in warnings:
            warnings.append(warn_msg)
        return None


# ── Music File Resolver ────────────────────────────────────────────────────

def _resolve_music_file(custom_path: Path | str | None, warnings: list[str]) -> Path | None:
    """Find background music file from custom path or default assets/music folder."""
    if custom_path:
        p = Path(custom_path)
        if p.exists() and p.is_file():
            return p
        warnings.append(f"Specified music file '{custom_path}' does not exist; skipping music.")
        return None

    # Search default assets/music folder
    assets_dir = Path("assets/music")
    if not assets_dir.exists():
        return None

    music_exts = {".mp3", ".wav", ".m4a"}
    for f in assets_dir.iterdir():
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in music_exts:
            return f

    return None
