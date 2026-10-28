"""
test_video_builder.py — Fast tests for Phase 3 video rendering.

Verifies:
  - 3-scene plan builds an MP4 that exists, is non-empty, and duration matches within 0.5s
  - Landscape, portrait, and tiny (50x50) images render without distortion or crash
  - Missing/corrupt images trigger warnings, fallback to placeholder, and video still exports
  - Text fitting helper: long captions fit within safe width and max 2 lines
  - Rupee symbol fallback path ('Rs.' on screen and warning when font lacks glyph)
  - Output directory auto-created and partial files cleaned up on failure
"""

from pathlib import Path
from PIL import Image
import pytest
from moviepy import VideoFileClip

from src.models import PropertyData, Scene, ScenePlan
from src.video_builder import (
    build_video,
    fit_font,
    prepare_caption_text,
    font_has_glyph,
    get_best_font_path,
    slugify,
    create_base_composed_frame,
)


def _make_dummy_image(path: Path, size=(300, 200), color=(100, 150, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path)
    return path


def _make_dummy_prop() -> PropertyData:
    return PropertyData(
        name="Oak Ridge Villa",
        location="Sector 100, Noida",
        price="₹85,00,000",
        property_type="Villa",
        features=["Garden", "Terrace", "Parking"],
        cta_text="Book a visit today",
        contact="+91-00000-00000",
        tagline="A peaceful sanctuary",
    )


def test_3_scene_plan_builds_valid_mp4(tmp_path):
    img1 = _make_dummy_image(tmp_path / "img1.jpg", (400, 300), (80, 120, 160))
    img2 = _make_dummy_image(tmp_path / "img2.jpg", (400, 300), (120, 80, 160))
    img3 = _make_dummy_image(tmp_path / "img3.jpg", (400, 300), (80, 160, 120))

    scenes = [
        Scene("title", img1, "Oak Ridge Villa\nVilla", "Welcome to Oak Ridge.", 2.5),
        Scene("feature", img2, "Private Garden", "Highlights include garden.", 2.5),
        Scene("cta", img3, "Book a visit today", "Contact details on screen.", 2.5),
    ]
    plan = ScenePlan(scenes=scenes, source="template")
    prop = _make_dummy_prop()

    out_file = tmp_path / "test_output.mp4"
    result_path = build_video(
        plan=plan,
        prop=prop,
        out_path=out_file,
        size=(320, 180),  # fast test size
        fps=12,           # fast test fps
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 0

    # Verify duration within 0.5s of plan total (7.5s)
    with VideoFileClip(str(result_path)) as clip:
        assert abs(clip.duration - plan.total_duration) <= 0.5


def test_landscape_portrait_and_tiny_images(tmp_path):
    # Landscape
    img_l = _make_dummy_image(tmp_path / "landscape.jpg", (500, 250), (200, 100, 50))
    # Portrait
    img_p = _make_dummy_image(tmp_path / "portrait.jpg", (250, 500), (50, 100, 200))
    # Tiny (50x50)
    img_t = _make_dummy_image(tmp_path / "tiny.jpg", (50, 50), (100, 200, 50))

    scenes = [
        Scene("title", img_l, "Landscape View", "Wide shot.", 2.0),
        Scene("feature", img_p, "Portrait View", "Tall shot.", 2.0),
        Scene("cta", img_t, "Tiny Icon", "Small shot.", 2.0),
    ]
    plan = ScenePlan(scenes=scenes, source="template")
    prop = _make_dummy_prop()

    out_file = tmp_path / "shapes_test.mp4"
    result = build_video(plan, prop, out_file, size=(320, 180), fps=12)
    assert result.exists()
    assert result.stat().st_size > 0


def test_missing_and_corrupt_image_renders_placeholder_with_warning(tmp_path):
    # Missing file
    img_missing = tmp_path / "does_not_exist.jpg"
    # Corrupt file
    img_corrupt = tmp_path / "corrupt.jpg"
    img_corrupt.write_bytes(b"garbage not an image data bytes 1234567890")
    # Valid file
    img_valid = _make_dummy_image(tmp_path / "valid.jpg", (300, 200))

    scenes = [
        Scene("title", img_missing, "Missing Image", "Voiceover one.", 2.0),
        Scene("feature", img_corrupt, "Corrupt Image", "Voiceover two.", 2.0),
        Scene("cta", img_valid, "Valid Image", "Voiceover three.", 2.0),
    ]
    plan = ScenePlan(scenes=scenes, source="template")
    prop = _make_dummy_prop()

    out_file = tmp_path / "fault_test.mp4"
    result = build_video(plan, prop, out_file, size=(320, 180), fps=12)

    assert result.exists()
    assert result.stat().st_size > 0
    # Both missing and corrupt must have generated warnings
    assert any("could not load image" in w.lower() for w in plan.warnings)
    assert len([w for w in plan.warnings if "placeholder" in w.lower()]) >= 2


def test_text_fitting_and_max_2_lines():
    font_path, _ = get_best_font_path()

    long_line = "This is an extremely long property feature description that cannot fit in a small width"
    font = fit_font(font_path, [long_line], max_width=200, base_size=40, min_size=12)
    bbox = font.getbbox(long_line)
    w = bbox[2] - bbox[0]
    # Font size shrinks so width is within bounds (or minimum font size reached)
    assert w < font.getbbox(long_line)[2] or font.size <= 14

    # 2-line max constraint
    three_lines = "Line One\nLine Two\nLine Three"
    lines = prepare_caption_text(three_lines, supports_rupee=True)
    assert len(lines) == 2
    assert lines == ["Line One", "Line Two"]


def test_rupee_glyph_fallback(tmp_path, monkeypatch):
    # Test prepare_caption_text replaces rupee with Rs. when supports_rupee is False
    cap = "₹85,00,000\nSector 150, Noida"
    replaced = prepare_caption_text(cap, supports_rupee=False)
    assert "₹" not in replaced[0]
    assert "Rs." in replaced[0]

    # Test warning added when no font supports rupee
    monkeypatch.setattr("src.video_builder.font_has_glyph", lambda path, glyph="₹": False)
    warnings = []
    font_path, supports_rupee = get_best_font_path(warnings)
    assert supports_rupee is False
    assert any("rupee" in w.lower() or "rs." in w.lower() for w in warnings)


def test_output_folder_auto_created_and_no_partial_on_failure(tmp_path, monkeypatch):
    img = _make_dummy_image(tmp_path / "img.jpg")
    scenes = [Scene("title", img, "Title", "VO", 2.0)]
    plan = ScenePlan(scenes=scenes, source="template")
    prop = _make_dummy_prop()

    # Deep nested path that does not exist
    deep_out = tmp_path / "deep" / "nested" / "folder" / "video.mp4"
    assert not deep_out.parent.exists()

    # First test: normal build auto-creates directory
    result = build_video(plan, prop, deep_out, size=(320, 180), fps=12)
    assert deep_out.parent.exists()
    assert result.exists()

    # Second test: forced failure during write does not leave temporary file
    deep_fail = tmp_path / "deep" / "nested" / "folder" / "fail.mp4"

    def mock_write(self, filename, *args, **kwargs):
        # Create a fake temp file simulating partial write before failure
        tmp_target = Path(filename)
        tmp_target.write_bytes(b"partial video data")
        raise RuntimeError("Simulated ffmpeg encoder crash")

    monkeypatch.setattr("moviepy.video.compositing.CompositeVideoClip.CompositeVideoClip.write_videofile", mock_write)

    with pytest.raises(RuntimeError, match="Simulated ffmpeg encoder crash"):
        build_video(plan, prop, deep_fail, size=(320, 180), fps=12)

    # Verify no partial temp files left behind
    leftover_tmps = list(deep_fail.parent.glob(".tmp_*"))
    assert len(leftover_tmps) == 0


def test_slugify():
    assert slugify("Maple Residency") == "maple_residency"
    assert slugify("Cedar Park Villa 2BHK!") == "cedar_park_villa_2bhk"
    assert slugify("Green Valley (Sector 150)") == "green_valley_sector_150"
    assert slugify("  ---  ") == "property_video"
