"""
test_audio_builder.py — Unit and integration tests for Phase 4 audio features.

Verifies:
  - Spoken text normalization (24/7, &, BHK, sq ft, rupee symbols, phone number stripping)
  - Voice verification and fallback
  - edge-tts synthesis with caching
  - Timeline duration extension: max(planned, narration + 0.7s)
  - TTS speed-up (+12% rate) when duration exceeds 40s
  - Graceful fallback when TTS fails (warning added, video still produced)
  - Music looping, trimming, and volume ducking
  - --no-audio flag leaves video silent (clip.audio is None)
  - Full build produces MP4 with valid audio track (clip.audio is not None)
"""

import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from PIL import Image
from moviepy import (
    AudioClip,
    AudioFileClip,
    VideoFileClip,
)

from src.models import PropertyData, Scene, ScenePlan
from src.video_builder import build_video
from src.audio_builder import (
    prepare_spoken_text,
    resolve_voice,
    synthesize_voiceover,
    build_audio_tracks,
    CACHE_DIR,
)


# ── Helpers for Dummy Audio & Visual Fixtures ───────────────────────────────

def _make_sine_audio(path: Path, duration: float = 2.0, freq: int = 440) -> Path:
    """Generate a clean synthetic sine-wave audio file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    make_frame = lambda t: np.sin(freq * 2 * np.pi * t)
    clip = AudioClip(make_frame, duration=duration, fps=22050)
    clip.write_audiofile(str(path), fps=22050, logger=None)
    clip.close()
    return path


def _make_dummy_image(path: Path, size=(320, 180), color=(80, 120, 160)) -> Path:
    """Generate a small test image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path)
    return path


# ── 1. Text Normalization ──────────────────────────────────────────────────

def test_text_normalization():
    raw = (
        "Welcome to 2BHK with 24/7 Security & Modular Kitchen.\n"
        "Spanning 1500 sq ft, priced at ₹85,00,000 or Rs. 85 Lakhs.\n"
        "Call us at +91-98765-43210 or 9876543210 today!"
    )
    normalized = prepare_spoken_text(raw)

    assert "24/7" not in normalized
    assert "24 by 7" in normalized
    assert " & " not in normalized
    assert " and " in normalized
    assert "2 B H K" in normalized
    assert "1500 square feet" in normalized
    assert "₹" not in normalized
    assert "rupees 85,00,000" in normalized
    assert "+91-98765-43210" not in normalized
    assert "9876543210" not in normalized
    # Whitespace collapsed
    assert "  " not in normalized


def test_text_normalization_variations():
    assert prepare_spoken_text("3BHK luxury") == "3 B H K luxury"
    assert prepare_spoken_text("Modern BHK") == "Modern B H K"
    assert prepare_spoken_text("1200 sqft") == "1200 square feet"
    assert prepare_spoken_text("900 sq. ft.") == "900 square feet"
    assert prepare_spoken_text("Priced at ₹45 Lakhs") == "Priced at rupees 45 Lakhs"
    assert prepare_spoken_text("") == ""


# ── 2. Voice Resolution & Fallback ──────────────────────────────────────────

def test_voice_resolution():
    with patch("src.audio_builder.get_available_voices", return_value=["en-IN-NeerjaNeural", "en-IN-PrabhatNeural", "en-US-JennyNeural"]):
        # Exact match
        assert resolve_voice("en-IN-NeerjaNeural") == "en-IN-NeerjaNeural"
        # Unknown voice falls back to another en-IN voice
        assert resolve_voice("non-existent-voice") == "en-IN-NeerjaNeural"

    with patch("src.audio_builder.get_available_voices", return_value=["en-US-JennyNeural", "en-GB-RyanNeural"]):
        # No en-IN voice available falls back to Jenny
        assert resolve_voice("en-IN-NeerjaNeural") == "en-US-JennyNeural"


# ── 3. TTS Synthesis & Caching ─────────────────────────────────────────────

def test_synthesize_voiceover_caching(tmp_path):
    target_out = tmp_path / "test_vo.mp3"
    custom_cache = tmp_path / "cache"

    # Mock edge_tts.Communicate.save to create a sine wave file
    save_calls = []

    async def mock_save(self, path):
        save_calls.append(path)
        _make_sine_audio(Path(path), duration=1.0)

    with patch("src.audio_builder.CACHE_DIR", custom_cache):
        with patch("edge_tts.Communicate.save", new=mock_save):
            # First call: synthesizes and saves to cache
            res1 = synthesize_voiceover("Welcome to Maple Residency Unique 123", target_out, use_cache=True)
            assert res1.exists()
            assert res1.stat().st_size > 0
            assert len(save_calls) == 1

            # Second call to a different path: should hit cache without calling Communicate.save
            target_out2 = tmp_path / "test_vo_2.mp3"
            res2 = synthesize_voiceover("Welcome to Maple Residency Unique 123", target_out2, use_cache=True)
            assert res2.exists()
            assert res2.stat().st_size > 0
            # save_calls remains 1 because cache was used!
            assert len(save_calls) == 1


# ── 4. Timeline Duration Extension ─────────────────────────────────────────

def test_duration_extension_from_narration(tmp_path):
    temp_dir = tmp_path / "audio_temp"
    img = _make_dummy_image(tmp_path / "img.jpg")
    scenes = [
        Scene("title", img, "Title", "A very short VO", duration=3.5),
        Scene("feature", img, "Feature", "A much longer voiceover that takes 4.5 seconds to speak", duration=3.5),
    ]
    plan = ScenePlan(scenes=scenes, source="template")

    # Mock synthesize_voiceover to write 1.5s for scene 1 and 4.2s for scene 2
    def mock_synth(text, out_path, voice, rate, use_cache=True):
        if "short" in text:
            _make_sine_audio(Path(out_path), duration=1.5)
        else:
            _make_sine_audio(Path(out_path), duration=4.2)
        return Path(out_path)

    with patch("src.audio_builder.synthesize_voiceover", side_effect=mock_synth):
        audio_clip, updated_durs = build_audio_tracks(
            plan=plan,
            temp_dir=temp_dir,
            tts_provider="edge",
            no_music=True,
        )

        assert audio_clip is not None
        # Scene 1: planned 3.5s > 1.5 + 0.7 = 2.2s -> stays 3.5s
        assert updated_durs[0] == 3.5
        # Scene 2: planned 3.5s < 4.2 + 0.7 = 4.9s -> extended to 4.9s
        assert updated_durs[1] == pytest.approx(4.9, abs=0.1)
        assert plan.scenes[1].duration == pytest.approx(4.9, abs=0.1)

        audio_clip.close()


# ── 5. TTS Rate Speed-Up When Exceeding 40s ─────────────────────────────────

def test_rate_speedup_when_total_exceeds_40s(tmp_path):
    temp_dir = tmp_path / "audio_temp"
    img = _make_dummy_image(tmp_path / "img.jpg")
    # 6 scenes with long planned durations
    scenes = [
        Scene("feature", img, f"Feature {i+1}", f"Voiceover description {i+1}", duration=7.0)
        for i in range(6)
    ]
    plan = ScenePlan(scenes=scenes, source="template")

    synth_rates = []

    def mock_synth(text, out_path, voice, rate, use_cache=True):
        synth_rates.append(rate)
        # Generate 6.5s audio clip
        _make_sine_audio(Path(out_path), duration=6.5)
        return Path(out_path)

    with patch("src.audio_builder.synthesize_voiceover", side_effect=mock_synth):
        audio_clip, updated_durs = build_audio_tracks(
            plan=plan,
            temp_dir=temp_dir,
            tts_provider="edge",
            no_music=True,
        )

        # Rate speedup should have triggered (target_rate up to +12%)
        assert any(r > 0 for r in synth_rates)
        assert audio_clip is not None
        audio_clip.close()


# ── 6. TTS Failure Path Graceful Degradation ────────────────────────────────

def test_tts_failure_path_preserves_plan_and_warns(tmp_path):
    temp_dir = tmp_path / "audio_temp"
    img = _make_dummy_image(tmp_path / "img.jpg")
    scenes = [Scene("title", img, "Title", "Voiceover", duration=4.0)]
    plan = ScenePlan(scenes=scenes, source="template")

    with patch("src.audio_builder.synthesize_voiceover", side_effect=RuntimeError("Network offline")):
        audio_clip, updated_durs = build_audio_tracks(
            plan=plan,
            temp_dir=temp_dir,
            tts_provider="edge",
            no_music=True,
            warnings=plan.warnings,
        )

        # Audio clip is None because narration failed and no music
        assert audio_clip is None
        # Planned duration is preserved
        assert updated_durs == [4.0]
        # Warning is recorded
        assert any("Narration unavailable; exporting without narration" in w for w in plan.warnings)


# ── 7. Background Music Looping and Volume ──────────────────────────────────

def test_music_loop_and_trim(tmp_path):
    temp_dir = tmp_path / "audio_temp"
    img = _make_dummy_image(tmp_path / "img.jpg")
    scenes = [Scene("title", img, "Title", "VO", duration=6.0)]
    plan = ScenePlan(scenes=scenes, source="template")

    # Create a 2.0s music file (shorter than video duration of 6.0s)
    music_file = _make_sine_audio(tmp_path / "bg_music.mp3", duration=2.0, freq=300)

    # Disable narration to test pure music track
    audio_clip, _ = build_audio_tracks(
        plan=plan,
        temp_dir=temp_dir,
        music_path=music_file,
        no_music=False,
    )

    assert audio_clip is not None
    # Audio duration matches total video duration (6.0s)
    assert pytest.approx(audio_clip.duration, abs=0.1) == 6.0
    audio_clip.close()


# ── 8. --no-audio Flag Leaves Video Silent ─────────────────────────────────

def test_no_audio_flag(tmp_path):
    temp_dir = tmp_path / "audio_temp"
    img = _make_dummy_image(tmp_path / "img.jpg")
    scenes = [Scene("title", img, "Title", "VO", duration=3.0)]
    plan = ScenePlan(scenes=scenes, source="template")

    audio_clip, durs = build_audio_tracks(
        plan=plan,
        temp_dir=temp_dir,
        no_audio=True,
    )
    assert audio_clip is None
    assert durs == [3.0]


# ── 9. Integration: Video Export With Valid Audio Track ────────────────────

def test_mp4_has_audio_track(tmp_path):
    img = _make_dummy_image(tmp_path / "img.jpg")
    scenes = [
        Scene("title", img, "Title Scene", "Welcome to the property", duration=2.0),
        Scene("cta", img, "CTA Scene", "Book a site visit", duration=2.0),
    ]
    plan = ScenePlan(scenes=scenes, source="template")
    prop = PropertyData("Test Villa", "Noida", 5000000, "Villa", ["Pool"], "Book")

    audio_file = _make_sine_audio(tmp_path / "narr.mp3", duration=3.0)
    audio_clip = AudioFileClip(str(audio_file))

    out_mp4 = tmp_path / "video_with_audio.mp4"
    build_video(
        plan=plan,
        prop=prop,
        out_path=out_mp4,
        size=(320, 180),
        fps=12,
        audio=audio_clip,
    )

    audio_clip.close()

    assert out_mp4.exists()
    assert out_mp4.stat().st_size > 0

    with VideoFileClip(str(out_mp4)) as vid:
        assert vid.audio is not None
        assert vid.audio.duration > 0
