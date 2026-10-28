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
from pathlib import Path
from typing import Callable, Any

import edge_tts
from moviepy import (
    AudioClip,
    AudioFileClip,
    CompositeAudioClip,
    afx,
)

from src.models import ScenePlan, Scene
from src.logger import setup_logger

logger = setup_logger()

DEFAULT_VOICE = "en-IN-NeerjaNeural"
DEFAULT_FALLBACK_VOICE = "en-US-JennyNeural"
TTS_TIMEOUT = 20  # seconds
CACHE_DIR = Path("output/.cache/tts")

_AVAILABLE_VOICES_CACHE: list[str] | None = None


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


def synthesize_voiceover(
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
    voice = resolve_voice(voice)

    # 1. Check cache
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key_raw = f"{spoken_text}|{voice}|{rate_str}".encode("utf-8")
        cache_key = hashlib.sha256(key_raw).hexdigest()
        cached_file = CACHE_DIR / f"{cache_key}.mp3"

        if cached_file.exists() and cached_file.stat().st_size > 0:
            shutil.copyfile(cached_file, out_path)
            return out_path

    # 2. Async synthesis with 20s timeout and 1 retry
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            async def _do_synth():
                communicate = edge_tts.Communicate(spoken_text, voice=voice, rate=rate_str)
                await communicate.save(str(out_path))

            _run_async_in_thread(_do_synth, timeout=TTS_TIMEOUT)

            if out_path.exists() and out_path.stat().st_size > 0:
                # Cache successful output
                if use_cache:
                    shutil.copyfile(out_path, cached_file)
                return out_path
        except Exception as exc:
            last_err = exc
            logger.warning("TTS attempt %d failed for voice '%s': %s", attempt + 1, voice, exc)

    raise RuntimeError(f"edge-tts synthesis failed after retry: {last_err}")


# ── Scene Timeline Recalculation & Narration Assembly ──────────────────────

def build_audio_tracks(
    plan: ScenePlan,
    temp_dir: Path,
    voice: str = DEFAULT_VOICE,
    base_rate: int = 0,
    music_path: Path | str | None = None,
    no_audio: bool = False,
    no_music: bool = False,
    no_narration: bool = False,
    warnings: list[str] | None = None,
) -> tuple[AudioClip | None, list[float]]:
    """Synthesize voiceovers for all scenes, sync timeline durations, and mix music.

    Args:
        plan: ScenePlan with scenes and voiceover texts.
        temp_dir: Directory for temporary audio clips.
        voice: Requested TTS voice.
        base_rate: Requested initial TTS speed rate.
        music_path: Optional explicit background music file.
        no_audio: If True, returns None without synthesizing.
        no_music: If True, skips background music.
        no_narration: If True, skips voiceover synthesis.
        warnings: Optional list to append warnings to.

    Returns:
        (composite_audio_or_none, updated_scene_durations)
    """
    if warnings is None:
        warnings = plan.warnings

    num_scenes = len(plan.scenes)
    original_durations = [s.duration for s in plan.scenes]

    if no_audio or (no_music and no_narration):
        return None, original_durations

    temp_dir.mkdir(parents=True, exist_ok=True)
    narration_files: list[Path | None] = [None] * num_scenes
    narration_durations: list[float] = [0.0] * num_scenes

    # ── 1. Synthesize narration for each scene ──────────────────────────────
    tts_failed = False
    if not no_narration:
        for i, scene in enumerate(plan.scenes):
            if not scene.voiceover.strip():
                continue

            audio_file = temp_dir / f"scene_{i+1}_vo.mp3"
            try:
                synthesize_voiceover(
                    text=scene.voiceover,
                    out_path=audio_file,
                    voice=voice,
                    rate=base_rate,
                )
                # Measure actual narration duration
                with AudioFileClip(str(audio_file)) as clip:
                    dur = float(clip.duration)

                narration_files[i] = audio_file
                narration_durations[i] = dur
            except Exception as exc:
                logger.warning("TTS generation failed for scene %d: %s", i + 1, exc)
                tts_failed = True
                break

    if tts_failed:
        warn_msg = "Narration unavailable; exporting without narration."
        if warn_msg not in warnings:
            warnings.append(warn_msg)
        narration_files = [None] * num_scenes
        narration_durations = [0.0] * num_scenes

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
            try:
                raw_music = AudioFileClip(str(resolved_music))
                video_total = plan.total_duration

                # Volume level: 0.10 when narration exists, 0.18 when no narration
                vol_factor = 0.10 if has_narration else 0.18

                # Loop or trim to match total video duration
                if raw_music.duration < video_total:
                    looped = raw_music.with_effects([afx.AudioLoop(duration=video_total)])
                else:
                    looped = raw_music.subclipped(0, video_total)

                music_track = looped.with_effects([
                    afx.AudioFadeIn(1.0),
                    afx.AudioFadeOut(1.5),
                    afx.MultiplyVolume(vol_factor),
                ])
            except Exception as exc:
                logger.warning("Could not process background music file: %s", exc)
                warnings.append(f"Could not load background music ({exc}); continuing without music.")

    # ── 6. Combine Narration + Music ────────────────────────────────────────
    final_audio: AudioClip | None = None
    if narration_track and music_track:
        final_audio = CompositeAudioClip([music_track, narration_track])
    elif narration_track:
        final_audio = narration_track
    elif music_track:
        final_audio = music_track

    return final_audio, scene_durations


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
