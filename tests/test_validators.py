"""
test_validators.py — pytest tests for src/validators.py

Images are generated on the fly with Pillow inside tmp_path.
No real property images are required.
"""

import pytest
from pathlib import Path
from PIL import Image

from src.validators import validate_property, validate_images, _format_indian_price


# ── Helpers ────────────────────────────────────────────────────────────────

def _img(path: Path, size: tuple = (800, 600), fmt: str = "JPEG") -> None:
    """Create a minimal valid image at *path*."""
    Image.new("RGB", size, (80, 120, 200)).save(path, format=fmt)


VALID = {
    "name": "Sunrise 2BHK",
    "location": "Andheri West, Mumbai",
    "price": "₹85,00,000",
    "property_type": "Apartment",
    "features": ["Swimming Pool", "Parking", "24/7 Security", "Garden"],
    "cta_text": "Book a site visit today",
    "contact": "+91-00000-00000",
}


# ── validate_property — happy path ─────────────────────────────────────────

def test_valid_input_passes():
    prop, warnings, errors = validate_property(VALID)
    assert errors == []
    assert prop is not None
    assert prop.name == "Sunrise 2BHK"


def test_optional_contact_included():
    prop, _, errors = validate_property(VALID)
    assert errors == []
    assert prop.contact == "+91-00000-00000"


def test_tagline_accepted():
    data = {**VALID, "tagline": "Live the dream"}
    prop, _, errors = validate_property(data)
    assert errors == []
    assert prop.tagline == "Live the dream"


# ── Required field checks ──────────────────────────────────────────────────

@pytest.mark.parametrize("field", ["name", "location", "property_type", "cta_text"])
def test_missing_required_text_field(field):
    data = {k: v for k, v in VALID.items() if k != field}
    _, _, errors = validate_property(data)
    assert any(field in e for e in errors), f"Expected error mentioning '{field}'"


@pytest.mark.parametrize("field", ["name", "location", "property_type", "cta_text"])
def test_empty_required_text_field(field):
    data = {**VALID, field: "   "}
    _, _, errors = validate_property(data)
    assert any(field in e for e in errors)


def test_missing_price():
    data = {k: v for k, v in VALID.items() if k != "price"}
    _, _, errors = validate_property(data)
    assert any("price" in e for e in errors)


def test_empty_price_string():
    data = {**VALID, "price": "  "}
    _, _, errors = validate_property(data)
    assert any("price" in e for e in errors)


def test_missing_features():
    data = {k: v for k, v in VALID.items() if k != "features"}
    _, _, errors = validate_property(data)
    assert any("features" in e for e in errors)


# ── Feature rules ──────────────────────────────────────────────────────────

def test_features_must_be_list():
    data = {**VALID, "features": "Swimming Pool"}
    _, _, errors = validate_property(data)
    assert any("features" in e for e in errors)


def test_features_empty_list_error():
    data = {**VALID, "features": []}
    _, _, errors = validate_property(data)
    assert any("features" in e for e in errors)


def test_features_all_blanks_error():
    data = {**VALID, "features": ["", "  ", "\t"]}
    _, _, errors = validate_property(data)
    assert any("features" in e for e in errors)


def test_features_blanks_dropped_silently():
    data = {**VALID, "features": ["Pool", "", "  ", "Parking"]}
    prop, _, errors = validate_property(data)
    assert errors == []
    assert prop.features == ["Pool", "Parking"]


def test_features_duplicates_removed():
    data = {**VALID, "features": ["Pool", "pool", "POOL", "Parking"]}
    prop, _, errors = validate_property(data)
    assert errors == []
    assert len(prop.features) == 2   # Pool + Parking


def test_features_1_gives_warning():
    data = {**VALID, "features": ["Pool"]}
    _, warnings, errors = validate_property(data)
    assert errors == []
    assert any("short" in w.lower() for w in warnings)


def test_features_2_gives_warning():
    data = {**VALID, "features": ["Pool", "Parking"]}
    _, warnings, errors = validate_property(data)
    assert errors == []
    assert any("short" in w.lower() for w in warnings)


def test_features_3_no_short_warning():
    data = {**VALID, "features": ["Pool", "Parking", "Garden"]}
    _, warnings, errors = validate_property(data)
    assert errors == []
    assert not any("short" in w.lower() for w in warnings)


def test_features_too_many_truncated_to_6():
    data = {**VALID, "features": [f"Feature {i}" for i in range(10)]}
    prop, warnings, errors = validate_property(data)
    assert errors == []
    assert len(prop.features) == 6
    assert any("6" in w for w in warnings)


def test_feature_item_too_long_is_error():
    long_feat = "A" * 81
    data = {**VALID, "features": [long_feat, "Pool"]}
    _, _, errors = validate_property(data)
    assert len(errors) >= 1


# ── Price formatting ───────────────────────────────────────────────────────

def test_numeric_price_8500000():
    assert _format_indian_price(8500000) == "₹85,00,000"


def test_numeric_price_100000():
    assert _format_indian_price(100000) == "₹1,00,000"


def test_numeric_price_10000000():
    assert _format_indian_price(10000000) == "₹1,00,00,000"


def test_numeric_price_999():
    assert _format_indian_price(999) == "₹999"


def test_validate_numeric_price_formatted():
    data = {**VALID, "price": 8500000}
    prop, _, errors = validate_property(data)
    assert errors == []
    assert prop.price == "₹85,00,000"


def test_validate_float_price_rounded():
    data = {**VALID, "price": 4500000.0}
    prop, _, errors = validate_property(data)
    assert errors == []
    assert prop.price == "₹45,00,000"


def test_zero_price_error():
    data = {**VALID, "price": 0}
    _, _, errors = validate_property(data)
    assert any("price" in e.lower() for e in errors)


def test_negative_price_error():
    data = {**VALID, "price": -500000}
    _, _, errors = validate_property(data)
    assert any("price" in e.lower() for e in errors)


def test_string_price_kept_as_is():
    data = {**VALID, "price": "Call for price"}
    prop, _, errors = validate_property(data)
    assert errors == []
    assert prop.price == "Call for price"


def test_numeric_string_price_2000000_formatted():
    data = {**VALID, "price": "2000000"}
    prop, _, errors = validate_property(data)
    assert errors == []
    assert prop.price == "₹20,00,000"


def test_numeric_string_price_formatted_with_rupee_symbol():
    data = {**VALID, "price": "₹20,00,000"}
    prop, _, errors = validate_property(data)
    assert errors == []
    assert prop.price == "₹20,00,000"


def test_textual_price_45_lakhs_kept_as_is():
    data = {**VALID, "price": "45 Lakhs"}
    prop, _, errors = validate_property(data)
    assert errors == []
    assert prop.price == "45 Lakhs"


def test_textual_price_rupee_45_lakhs_kept_as_is():
    data = {**VALID, "price": "₹45 Lakhs"}
    prop, _, errors = validate_property(data)
    assert errors == []
    assert prop.price == "₹45 Lakhs"


def test_textual_price_on_request_kept_as_is():
    data = {**VALID, "price": "On request"}
    prop, _, errors = validate_property(data)
    assert errors == []
    assert prop.price == "On request"


# ── Optional / extra field handling ───────────────────────────────────────

def test_missing_contact_gives_warning_not_error():
    data = {k: v for k, v in VALID.items() if k != "contact"}
    _, warnings, errors = validate_property(data)
    assert errors == []
    assert any("contact" in w.lower() for w in warnings)


def test_unknown_field_gives_warning():
    data = {**VALID, "bedrooms": 2, "floors": 5}
    _, warnings, errors = validate_property(data)
    assert any("bedrooms" in w for w in warnings)
    assert any("floors" in w for w in warnings)


def test_field_too_long_120_chars():
    data = {**VALID, "name": "N" * 121}
    _, _, errors = validate_property(data)
    assert any("name" in e for e in errors)


def test_field_exactly_120_chars_ok():
    data = {**VALID, "name": "N" * 120}
    prop, _, errors = validate_property(data)
    assert errors == []
    assert len(prop.name) == 120


# ── validate_images ────────────────────────────────────────────────────────

def test_valid_images_accepted(tmp_path):
    for i in range(4):
        _img(tmp_path / f"img{i + 1}.jpg")
    paths, warnings, errors = validate_images(tmp_path)
    assert errors == []
    assert len(paths) == 4


def test_missing_folder(tmp_path):
    _, _, errors = validate_images(tmp_path / "nonexistent")
    assert len(errors) >= 1
    assert any(str(tmp_path / "nonexistent") in e for e in errors)


def test_zero_images_error(tmp_path):
    _, _, errors = validate_images(tmp_path)
    assert len(errors) >= 1
    assert any("no" in e.lower() for e in errors)


def test_one_image_warning(tmp_path):
    _img(tmp_path / "single.jpg")
    paths, warnings, errors = validate_images(tmp_path)
    assert errors == []
    assert len(paths) == 1
    assert any("recommend" in w.lower() or "shorter" in w.lower() for w in warnings)


def test_two_images_warning(tmp_path):
    for i in range(2):
        _img(tmp_path / f"img{i + 1}.jpg")
    paths, warnings, errors = validate_images(tmp_path)
    assert errors == []
    assert any("recommend" in w.lower() or "shorter" in w.lower() for w in warnings)


def test_four_images_no_count_warning(tmp_path):
    for i in range(4):
        _img(tmp_path / f"img{i + 1}.jpg")
    _, warnings, errors = validate_images(tmp_path)
    assert errors == []
    assert not any("recommend" in w.lower() for w in warnings)


def test_ten_images_truncated_to_8(tmp_path):
    for i in range(10):
        _img(tmp_path / f"img{i + 1:02d}.jpg")
    paths, warnings, errors = validate_images(tmp_path)
    assert errors == []
    assert len(paths) == 8
    assert any("8" in w for w in warnings)


def test_corrupt_image_skipped_with_warning(tmp_path):
    _img(tmp_path / "good.jpg")
    (tmp_path / "corrupt.jpg").write_bytes(b"this is not a jpeg file")
    paths, warnings, errors = validate_images(tmp_path)
    assert len(paths) == 1
    assert any("corrupt.jpg" in w for w in warnings)


def test_txt_file_ignored_with_warning(tmp_path):
    _img(tmp_path / "photo.jpg")
    (tmp_path / "notes.txt").write_text("hello")
    paths, warnings, errors = validate_images(tmp_path)
    assert errors == []
    assert len(paths) == 1
    assert any("notes.txt" in w for w in warnings)


def test_hidden_file_silently_ignored(tmp_path):
    for i in range(4):
        _img(tmp_path / f"img{i + 1}.jpg")
    (tmp_path / ".gitkeep").write_text("")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x00")
    paths, warnings, errors = validate_images(tmp_path)
    assert errors == []
    assert len(paths) == 4
    assert not any(".gitkeep" in w for w in warnings)
    assert not any(".DS_Store" in w for w in warnings)


def test_png_accepted(tmp_path):
    _img(tmp_path / "photo.png", fmt="PNG")
    paths, _, errors = validate_images(tmp_path)
    assert errors == []
    assert len(paths) == 1


def test_low_res_image_accepted_with_warning(tmp_path):
    _img(tmp_path / "tiny.jpg", size=(320, 240))
    paths, warnings, errors = validate_images(tmp_path)
    assert errors == []
    assert len(paths) == 1
    assert any("low" in w.lower() or "resolution" in w.lower() for w in warnings)


def test_natural_sort_order(tmp_path):
    for name in ["img10.jpg", "img2.jpg", "img1.jpg"]:
        _img(tmp_path / name)
    paths, _, errors = validate_images(tmp_path)
    assert errors == []
    assert [p.name for p in paths] == ["img1.jpg", "img2.jpg", "img10.jpg"]


def test_mixed_extensions(tmp_path):
    _img(tmp_path / "a.jpg")
    _img(tmp_path / "b.png", fmt="PNG")
    (tmp_path / "c.bmp").write_bytes(b"\x00" * 10)   # unsupported
    paths, warnings, errors = validate_images(tmp_path)
    assert errors == []
    assert len(paths) == 2
    assert any("c.bmp" in w for w in warnings)


def test_custom_script_known_field_no_warning():
    data = {
        **VALID,
        "custom_script": {
            "title": {"caption": "Custom", "voiceover": "Custom VO"}
        },
    }
    prop, warnings, errors = validate_property(data)
    assert errors == []
    assert not any("custom_script" in w for w in warnings)
    assert prop.custom_script == data["custom_script"]

