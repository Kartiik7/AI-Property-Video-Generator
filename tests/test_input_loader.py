"""
test_input_loader.py — pytest tests for src/input_loader.py

Covers: JSON loading (valid, missing, bad syntax, array), interactive
input (valid flow, Ctrl+C), and image path handling.
"""

import json
import builtins
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from src.input_loader import load_from_json, load_interactive


# ── Helpers ────────────────────────────────────────────────────────────────

def _img(path: Path, size: tuple = (800, 600)) -> None:
    Image.new("RGB", size, (80, 120, 200)).save(path, format="JPEG")


def _make_images(folder: Path, count: int = 4) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        _img(folder / f"img{i + 1}.jpg")


VALID_DICT = {
    "name": "Sunrise 2BHK",
    "location": "Pune",
    "price": "₹50,00,000",
    "property_type": "Apartment",
    "features": ["Pool", "Parking", "Gym", "Security"],
    "cta_text": "Book now",
    "contact": "+91-00000-00000",
}


# ── load_from_json ─────────────────────────────────────────────────────────

def test_valid_json_loads_ok(tmp_path):
    jf = tmp_path / "prop.json"
    jf.write_text(json.dumps(VALID_DICT), encoding="utf-8")
    img_dir = tmp_path / "images"
    _make_images(img_dir)

    result = load_from_json(jf, img_dir)

    assert result.ok
    assert result.property is not None
    assert result.property.name == "Sunrise 2BHK"
    assert len(result.images) == 4


def test_missing_json_file(tmp_path):
    result = load_from_json(tmp_path / "missing.json", tmp_path / "images")
    assert not result.ok
    assert any("not found" in e.lower() for e in result.errors)


def test_invalid_json_syntax_shows_line_col(tmp_path):
    jf = tmp_path / "bad.json"
    jf.write_text('{ "name": oops }', encoding="utf-8")

    result = load_from_json(jf, tmp_path / "images")

    assert not result.ok
    assert any("line" in e.lower() for e in result.errors)
    assert any("column" in e.lower() for e in result.errors)


def test_json_array_rejected(tmp_path):
    jf = tmp_path / "arr.json"
    jf.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    result = load_from_json(jf, tmp_path / "images")

    assert not result.ok
    assert any("object" in e.lower() for e in result.errors)


def test_json_null_rejected(tmp_path):
    jf = tmp_path / "null.json"
    jf.write_text("null", encoding="utf-8")

    result = load_from_json(jf, tmp_path / "images")

    assert not result.ok
    assert any("object" in e.lower() for e in result.errors)


def test_utf8_bom_handled(tmp_path):
    """Files saved with BOM (UTF-8 sig) by some Windows editors must load."""
    jf = tmp_path / "bom.json"
    # utf-8-sig adds a BOM prefix
    jf.write_bytes(b"\xef\xbb\xbf" + json.dumps(VALID_DICT).encode("utf-8"))
    img_dir = tmp_path / "images"
    _make_images(img_dir)

    result = load_from_json(jf, img_dir)
    assert result.ok


def test_missing_images_dir_gives_error(tmp_path):
    jf = tmp_path / "prop.json"
    jf.write_text(json.dumps(VALID_DICT), encoding="utf-8")

    result = load_from_json(jf, tmp_path / "no_such_dir")
    assert not result.ok
    assert any("image" in e.lower() or "folder" in e.lower() for e in result.errors)


def test_json_with_property_errors_accumulates_both(tmp_path):
    """Property errors and image errors should both appear in result."""
    bad = {**VALID_DICT}
    del bad["name"]              # property error
    jf = tmp_path / "prop.json"
    jf.write_text(json.dumps(bad), encoding="utf-8")
    # images dir does not exist -> image error

    result = load_from_json(jf, tmp_path / "images")
    assert not result.ok
    assert len(result.errors) >= 2   # at least one property + one image error


def test_numeric_price_formatted_from_json(tmp_path):
    data = {**VALID_DICT, "price": 8500000}
    jf = tmp_path / "prop.json"
    jf.write_text(json.dumps(data), encoding="utf-8")
    img_dir = tmp_path / "images"
    _make_images(img_dir)

    result = load_from_json(jf, img_dir)
    assert result.ok
    assert result.property.price == "₹85,00,000"


# ── load_interactive ───────────────────────────────────────────────────────

def _run_interactive(inputs: list[str], img_dir: Path):
    """Helper: patch builtins.input with *inputs* and call load_interactive."""
    with patch("builtins.input", side_effect=iter(inputs)):
        return load_interactive(img_dir)


def test_interactive_valid_flow(tmp_path):
    img_dir = tmp_path / "images"
    _make_images(img_dir)

    result = _run_interactive(
        [
            "Sunrise 2BHK",          # name
            "Mumbai",                # location
            "₹50,00,000",            # price (string, kept as-is)
            "Apartment",             # property_type
            "Book now",              # cta_text
            "Pool",                  # feature 1
            "Parking",               # feature 2
            "",                      # blank → end features
            "+91-00000-00000",       # contact
            "",                      # no tagline
        ],
        img_dir,
    )

    assert result.ok, f"Unexpected errors: {result.errors}"
    assert result.property.name == "Sunrise 2BHK"
    assert result.property.price == "₹50,00,000"
    assert result.property.features == ["Pool", "Parking"]
    assert len(result.images) == 4


def test_interactive_re_asks_empty_required_field(tmp_path):
    """Entering blank for a required field should prompt again."""
    img_dir = tmp_path / "images"
    _make_images(img_dir)

    result = _run_interactive(
        [
            "",             # name — blank, should re-ask
            "My Property",  # name — second attempt succeeds
            "Mumbai",
            "₹30,00,000",
            "Villa",
            "Call us",
            "Garden",
            "",             # end features
            "",             # no contact
            "",             # no tagline
        ],
        img_dir,
    )

    assert result.property is not None
    assert result.property.name == "My Property"


def test_interactive_ctrl_c_returns_clean_result(tmp_path):
    """KeyboardInterrupt should not propagate; result has a clear error."""
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        result = load_interactive(tmp_path / "images")

    assert not result.ok
    assert any("cancel" in e.lower() for e in result.errors)


def test_interactive_feature_blank_re_asks(tmp_path):
    """Blank feature list should prompt again before accepting."""
    img_dir = tmp_path / "images"
    _make_images(img_dir)

    result = _run_interactive(
        [
            "Home",
            "Delhi",
            "₹20,00,000",
            "Apartment",
            "Visit today",
            "",            # blank without any feature → should re-ask
            "Garden",      # now enter one feature
            "",            # blank → end features
            "",            # no contact
            "",            # no tagline
        ],
        img_dir,
    )

    assert result.property is not None
    assert result.property.features == ["Garden"]


def test_interactive_no_images_gives_error(tmp_path):
    """Interactive mode with an empty images dir should still report an error."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()   # exists but empty

    result = _run_interactive(
        [
            "Home", "Delhi", "₹20,00,000", "Apartment", "Visit today",
            "Garden", "", "", "",
        ],
        img_dir,
    )

    assert not result.ok
    assert any("image" in e.lower() or "no usable" in e.lower() for e in result.errors)
