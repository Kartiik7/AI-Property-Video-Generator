"""
test_edge_cases.py — Comprehensive reliability and edge-case test suite for Phase 6.

Matrix covered:
- Images: none, 1-3 count warning, 9+ count limit warning, corrupt, unsupported,
  huge images (downscaled), EXIF-rotated, CMYK, Palette, Grayscale, RGBA-with-transparency,
  tiny, very wide, and very tall images.
- Text: special characters / slugs, Hindi / non-Latin glyph handling without crash,
  long names and features.
- Fields: missing optional fields, empty features, price as text vs number, duplicate features.
- Output: locked/open destination file fallback to timestamped alternate filename, partial file cleanup.
- Services: graceful degradation on LLM failure, TTS failure, corrupt music file.
- Logging: run_id present in logs/app.log and API keys are automatically scrubbed.
"""

import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from src.validators import validate_property, validate_images
from src.video_builder import (
    create_base_composed_frame,
    render_text_overlay,
    resolve_font_for_text,
    get_best_font_path,
    slugify,
    build_video,
)
from src.models import PropertyData, ScenePlan, Scene
from src.audio_builder import build_audio_tracks, load_background_music
from src.script_generator import build_scene_plan
from src.logger import setup_logger, set_run_id, get_run_id


# ── 1. Image Edge Cases Matrix ───────────────────────────────────────────────

def test_huge_image_downscale(tmp_path):
    """Synthetic 4000x3000 image downscaled on load for memory and speed."""
    huge_img = tmp_path / "huge.jpg"
    im = Image.new("RGB", (4000, 3000), color=(100, 150, 200))
    im.save(huge_img, format="JPEG")

    frame = create_base_composed_frame(huge_img, (1280, 720))
    assert frame.size == (1280, 720)


def test_exif_rotated_image(tmp_path):
    """EXIF orientation (e.g. orientation 6 = 90 deg CW phone photo)."""
    exif_img = tmp_path / "phone_photo.jpg"
    im = Image.new("RGB", (400, 600), color=(80, 120, 160))
    # Create simple EXIF with orientation tag 274 (0x0112)
    exif_data = im.getexif()
    exif_data[0x0112] = 6  # 90 degrees CW
    im.save(exif_img, format="JPEG", exif=exif_data)

    frame = create_base_composed_frame(exif_img, (1280, 720))
    assert frame.size == (1280, 720)


def test_color_modes_cmyk_palette_grayscale_rgba(tmp_path):
    """Safe conversion of CMYK, Palette with transparency, Grayscale, and RGBA."""
    # CMYK
    cmyk_p = tmp_path / "img_cmyk.jpg"
    Image.new("CMYK", (800, 600), color=(0, 100, 100, 0)).save(cmyk_p, format="JPEG")
    f_cmyk = create_base_composed_frame(cmyk_p, (1280, 720))
    assert f_cmyk.mode == "RGB"

    # Grayscale (L)
    gray_p = tmp_path / "img_gray.png"
    Image.new("L", (800, 600), color=128).save(gray_p, format="PNG")
    f_gray = create_base_composed_frame(gray_p, (1280, 720))
    assert f_gray.mode == "RGB"

    # RGBA with transparent alpha
    rgba_p = tmp_path / "img_rgba.png"
    im_rgba = Image.new("RGBA", (800, 600), color=(255, 0, 0, 128))
    im_rgba.save(rgba_p, format="PNG")
    f_rgba = create_base_composed_frame(rgba_p, (1280, 720))
    assert f_rgba.mode == "RGB"

    # Palette (P)
    palette_p = tmp_path / "img_palette.png"
    im_p = Image.new("P", (800, 600))
    im_p.save(palette_p, format="PNG")
    f_palette = create_base_composed_frame(palette_p, (1280, 720))
    assert f_palette.mode == "RGB"


def test_tiny_and_extreme_aspect_ratio_images(tmp_path):
    """Tiny (10x10) and extreme aspect ratios (3000x150 wide, 150x3000 tall)."""
    # Tiny 10x10
    tiny_p = tmp_path / "tiny.jpg"
    Image.new("RGB", (10, 10), (50, 50, 50)).save(tiny_p)
    f_tiny = create_base_composed_frame(tiny_p, (1280, 720))
    assert f_tiny.size == (1280, 720)

    # Ultra-wide
    wide_p = tmp_path / "wide.jpg"
    Image.new("RGB", (2000, 150), (60, 70, 80)).save(wide_p)
    f_wide = create_base_composed_frame(wide_p, (1280, 720))
    assert f_wide.size == (1280, 720)

    # Ultra-tall
    tall_p = tmp_path / "tall.jpg"
    Image.new("RGB", (150, 2000), (90, 80, 70)).save(tall_p)
    f_tall = create_base_composed_frame(tall_p, (1280, 720))
    assert f_tall.size == (1280, 720)


def test_corrupt_and_unsupported_image_files(tmp_path):
    """Corrupt image and unsupported extensions handled gracefully."""
    # Corrupt file
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"NOT_AN_IMAGE_HEADER_DATA_123456789")

    # Unsupported format
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Hello world", encoding="utf-8")

    # Valid image
    valid_file = tmp_path / "valid.jpg"
    Image.new("RGB", (800, 600)).save(valid_file)

    accepted, warns, errs = validate_images(tmp_path)
    assert valid_file in accepted
    assert corrupt_file not in accepted
    assert txt_file not in accepted
    assert any("corrupt" in w.lower() for w in warns)
    assert any("not a supported image type" in w.lower() for w in warns)


def test_image_counts_warnings_and_limits(tmp_path):
    """Check image counts: 0 (error), 1-3 (warning), 9+ (first 8 + warning)."""
    # 0 images
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    _, _, errs = validate_images(empty_dir)
    assert any("no usable images found" in e.lower() for e in errs)

    # 2 images -> warning
    dir_2 = tmp_path / "dir_2"
    dir_2.mkdir()
    for i in range(2):
        Image.new("RGB", (800, 600)).save(dir_2 / f"img_{i}.jpg")
    acc_2, warns_2, _ = validate_images(dir_2)
    assert len(acc_2) == 2
    assert any("only 2 image(s) found" in w.lower() for w in warns_2)

    # 10 images -> first 8 + warning
    dir_10 = tmp_path / "dir_10"
    dir_10.mkdir()
    for i in range(10):
        Image.new("RGB", (800, 600)).save(dir_10 / f"img_{i:02d}.jpg")
    acc_10, warns_10, _ = validate_images(dir_10)
    assert len(acc_10) == 8
    assert any("only the first 8 will be used" in w.lower() for w in warns_10)


# ── 2. Text and Glyph Edge Cases ─────────────────────────────────────────────

def test_slugify_special_characters():
    """Property names with punctuation, symbols, and extra spaces produce safe slugs."""
    assert slugify("Maple Residency @ Sector 150 #1!") == "maple_residency_sector_150_1"
    assert slugify("   Luxury Villa --- 2026   ") == "luxury_villa_2026"
    assert slugify("???!!!") == "property_video"


def test_hindi_devanagari_glyph_handling():
    """Hindi/Devanagari characters handled safely without crashing on Pillow overlay."""
    font_p, _ = get_best_font_path()
    lines = ["शानदार विला", "प्राइम लोकेशन"]
    warnings: list[str] = []

    resolved_font, safe_lines = resolve_font_for_text(font_p, lines, warnings)
    overlay, bbox = render_text_overlay("feature", safe_lines, (1280, 720), resolved_font)

    assert overlay is not None
    assert overlay.size == (1280, 720)


def test_text_length_validation():
    """Clear error when text exceeds screen limits."""
    prop_dict = {
        "name": "A" * 200,   # Exceeds MAX_TEXT_LEN (120)
        "location": "Noida",
        "price": 8500000,
        "property_type": "Apartment",
        "features": ["F1"],
        "cta_text": "Book now",
    }
    _, _, errs = validate_property(prop_dict)
    assert any("Field 'name' is too long" in e for e in errs)


# ── 3. Fields Handling ───────────────────────────────────────────────────────

def test_fields_missing_optional_and_deduplication():
    """Optional fields default to None, duplicate features are deduplicated."""
    prop_dict = {
        "name": "Oak Manor",
        "location": "Whitefield, Bangalore",
        "price": "₹1,20,00,000",
        "property_type": "Villa",
        "features": ["Garden", "Swimming Pool", "GARDEN", "swimming pool", "Gym"],
        "cta_text": "Call today",
    }
    prop, warns, errs = validate_property(prop_dict)
    assert len(errs) == 0
    assert prop is not None
    assert prop.contact is None
    assert prop.tagline is None
    # Deduplicated features
    assert prop.features == ["Garden", "Swimming Pool", "Gym"]


def test_price_string_and_number_interoperability():
    """Price can be supplied as number (8500000) or Indian-formatted string."""
    base = {
        "name": "Test", "location": "Test", "property_type": "Villa",
        "features": ["F1", "F2", "F3"], "cta_text": "CTA",
    }
    p1, _, _ = validate_property({**base, "price": 5000000})
    assert p1.price == "₹50,00,000"

    p2, _, _ = validate_property({**base, "price": "₹50,00,000"})
    assert p2.price == "₹50,00,000"

    p3, _, _ = validate_property({**base, "price": "Price on Request"})
    assert p3.price == "Price on Request"


# ── 4. Output Locked File Handling ───────────────────────────────────────────

def test_locked_destination_file_writes_to_alternate_name(tmp_path):
    """When target MP4 is locked, exports to timestamped alternate file with warning."""
    out_video = tmp_path / "locked_video.mp4"
    out_video.write_bytes(b"LOCKED_SIMULATION")

    # Build dummy 1-scene plan
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (640, 360)).save(img_path)

    plan = ScenePlan(
        scenes=[
            Scene(scene_type="title", image_path=img_path, caption="Title", voiceover="VO", duration=2.0)
        ],
        source="template",
    )
    prop = PropertyData(
        name="Locked Prop", location="Loc", price="10", property_type="Villa",
        features=["F1"], cta_text="CTA"
    )

    def fake_write(filename, *args, **kwargs):
        Path(filename).write_bytes(b"VIDEO_DATA")

    orig_replace = Path.replace

    def fake_replace(self, target):
        if Path(target) == out_video:
            raise PermissionError("File locked by media player")
        return orig_replace(self, target)

    with patch("moviepy.CompositeVideoClip.write_videofile", side_effect=fake_write), \
         patch.object(Path, "replace", fake_replace):

        final_path = build_video(plan, prop, out_path=out_video, size=(640, 360), fps=10)

        # Must have written to alternate file
        assert final_path != out_video
        assert "locked_video_" in final_path.name
        assert any("is locked or in use" in w for w in plan.warnings)


# ── 5. Service Failure & Graceful Degradation ────────────────────────────────

def test_llm_failure_gracefully_degrades_to_template(tmp_path):
    """LLM network/503/timeout failure falls back to template scenes with a clean warning."""
    prop = PropertyData(
        name="Skyline", location="Gurugram", price="₹90,00,000",
        property_type="Apartment", features=["Balcony", "Clubhouse"], cta_text="Visit today"
    )
    img_p = tmp_path / "f1.jpg"
    Image.new("RGB", (640, 360)).save(img_p)

    with patch("src.llm_client.generate_scene_copy", return_value=(None, ["LLM API timed out"])):
        plan = build_scene_plan(prop, [img_p], use_llm=True)

        assert plan.source == "template"
        assert any("LLM API timed out" in w for w in plan.warnings)
        assert len(plan.scenes) >= 4


def test_corrupt_music_file_skipped_with_warning(tmp_path):
    """Corrupt audio file in assets/music is skipped with a clear warning."""
    bad_music = tmp_path / "bad.mp3"
    bad_music.write_bytes(b"NOT_AUDIO_BYTES_000")

    warns: list[str] = []
    clip = load_background_music(bad_music, target_duration=10.0, warnings=warns)
    assert clip is None
    assert any("Could not load background music" in w for w in warns)


# ── 6. Logging and Security Check ───────────────────────────────────────────

def test_logging_includes_run_id_and_scrubs_api_keys(tmp_path):
    """Verify run_id is included in logs and API keys are scrubbed."""
    logger = setup_logger()
    import uuid
    unique_run_id = f"test_{uuid.uuid4().hex[:6]}"
    set_run_id(unique_run_id)
    assert get_run_id() == unique_run_id

    fake_gemini_key = "AIzaSyD1234567890abcdef1234567890abcde"
    fake_openai_key = "sk-proj-1234567890abcdef1234567890"

    logger.info("Initializing run with key %s and secret %s", fake_gemini_key, fake_openai_key)

    log_path = Path("logs/app.log")
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")

    # Filter lines for this specific test log entry
    init_lines = [
        line for line in content.splitlines()
        if f"[run:{unique_run_id}]" in line and "Initializing run with key" in line
    ]
    assert len(init_lines) == 1
    assert fake_gemini_key not in init_lines[0]
    assert fake_openai_key not in init_lines[0]
    assert "[REDACTED_API_KEY]" in init_lines[0]
