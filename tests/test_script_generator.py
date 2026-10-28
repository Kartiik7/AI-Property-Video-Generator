"""
test_script_generator.py — Tests for template-based script and caption generation.

Verifies:
  - 4 images / 3 features
  - 5 images / 5 features (features grouped/merged)
  - 8 images / 6 features
  - 2 images / 3 features (image reuse warning)
  - Total duration always in [20s, 40s] and every scene >= 3.0s
  - Captions max 2 lines x 38 chars; no empty caption or voiceover
  - Determinism (same input -> identical output)
  - Every feature from input is included in the plan
  - Spoken price conversions ("45 lakh rupees", "1 crore 20 lakh rupees", "On request")
"""

import pytest
from pathlib import Path

from src.models import PropertyData
from src.script_generator import (
    build_scene_plan,
    format_spoken_price,
    wrap_caption,
    group_features,
)


# ── Spoken price formatting tests ──────────────────────────────────────────

def test_spoken_price_45_lakhs():
    assert format_spoken_price("4500000") == "45 lakh rupees"
    assert format_spoken_price("₹45,00,000") == "45 lakh rupees"


def test_spoken_price_1_crore_20_lakhs():
    assert format_spoken_price("12000000") == "1 crore 20 lakh rupees"
    assert format_spoken_price("₹1,20,00,000") == "1 crore 20 lakh rupees"


def test_spoken_price_crore_round():
    assert format_spoken_price("10000000") == "1 crore rupees"
    assert format_spoken_price("₹2,00,00,000") == "2 crore rupees"


def test_spoken_price_plain_text_unchanged():
    assert format_spoken_price("On request") == "On request"
    assert format_spoken_price("Call for price") == "Call for price"


# ── Caption wrapping tests ─────────────────────────────────────────────────

def test_wrap_caption_within_limits():
    text = "Short Caption"
    wrapped = wrap_caption(text, max_lines=2, max_line_len=38)
    assert wrapped == "Short Caption"


def test_wrap_caption_long_text_splits_cleanly():
    long_text = "Rooftop Swimming Pool With Heated Water And Panoramic City View"
    wrapped = wrap_caption(long_text, max_lines=2, max_line_len=38)
    lines = wrapped.split("\n")
    assert len(lines) <= 2
    for line in lines:
        assert len(line) <= 38


# ── Feature grouping tests ─────────────────────────────────────────────────

def test_group_features_under_limit():
    feats = ["Pool", "Parking", "Gym"]
    assert group_features(feats, max_scenes=4) == ["Pool", "Parking", "Gym"]


def test_group_features_5_items_merged():
    feats = ["Pool", "Parking", "Gym", "Security", "Garden"]
    grouped = group_features(feats, max_scenes=4)
    assert len(grouped) == 4
    assert grouped[0] == "Pool"
    assert grouped[1] == "Parking"
    assert grouped[2] == "Gym"
    assert grouped[3] == "Security & Garden"


def test_group_features_6_items_merged():
    feats = ["F1", "F2", "F3", "F4", "F5", "F6"]
    grouped = group_features(feats, max_scenes=4)
    assert len(grouped) == 4
    assert grouped[3] == "F4 & F5 & F6"


# ── Full Plan Building Tests ───────────────────────────────────────────────

def _make_prop(num_features: int = 3, price: str = "₹45,00,000") -> PropertyData:
    feats = [f"Feature {i+1}" for i in range(num_features)]
    return PropertyData(
        name="Sunny Meadows 2BHK",
        location="Baner, Pune",
        price=price,
        property_type="Apartment",
        features=feats,
        cta_text="Book a site visit today",
        contact="+91-12345-67890",
        tagline="Your dream abode",
    )


def test_plan_4_images_3_features():
    prop = _make_prop(num_features=3)
    images = [Path(f"img_{i}.jpg") for i in range(4)]
    # Scenes = 1 title + 3 features + 1 price_loc + 1 cta = 6 scenes
    plan = build_scene_plan(prop, images)

    assert plan.source == "template"
    assert len(plan.scenes) == 6
    # 4 images for 6 scenes -> reuse warning
    assert any("reused" in w.lower() for w in plan.warnings)

    # Durations
    assert 20.0 <= plan.total_duration <= 40.0
    for s in plan.scenes:
        assert s.duration >= 3.0


def test_plan_5_images_5_features():
    prop = _make_prop(num_features=5)
    images = [Path(f"img_{i}.jpg") for i in range(5)]
    # Features grouped to 4 scenes -> Total scenes = 1 + 4 + 1 + 1 = 7 scenes
    plan = build_scene_plan(prop, images)

    assert len(plan.scenes) == 7
    assert 20.0 <= plan.total_duration <= 40.0
    for s in plan.scenes:
        assert s.duration >= 3.0

    # Every feature from the input must appear in captions or voiceovers
    combined_text = " ".join(s.caption + " " + s.voiceover for s in plan.scenes)
    for feat in prop.features:
        assert feat in combined_text


def test_plan_8_images_6_features():
    prop = _make_prop(num_features=6)
    images = [Path(f"img_{i}.jpg") for i in range(8)]
    # Features grouped to 4 scenes -> Total scenes = 1 + 4 + 1 + 1 = 7 scenes
    plan = build_scene_plan(prop, images)

    assert len(plan.scenes) == 7
    # 8 images available for 7 scenes -> no reuse warning
    assert not any("reused" in w.lower() for w in plan.warnings)
    assert 20.0 <= plan.total_duration <= 40.0


def test_plan_2_images_3_features_reuse_warning():
    prop = _make_prop(num_features=3)
    images = [Path("img_a.jpg"), Path("img_b.jpg")]
    plan = build_scene_plan(prop, images)

    assert len(plan.scenes) == 6
    assert any("reused" in w.lower() for w in plan.warnings)


def test_caption_and_voiceover_constraints():
    prop = _make_prop(num_features=4)
    images = [Path(f"img_{i}.jpg") for i in range(6)]
    plan = build_scene_plan(prop, images)

    for i, s in enumerate(plan.scenes, 1):
        assert s.caption.strip(), f"Scene {i} has empty caption"
        assert s.voiceover.strip(), f"Scene {i} has empty voiceover"

        lines = s.caption.split("\n")
        assert len(lines) <= 2, f"Scene {i} caption has {len(lines)} lines (max 2)"
        for line in lines:
            assert len(line) <= 38, f"Scene {i} line '{line}' is {len(line)} chars (max 38)"

        words = s.voiceover.split()
        assert len(words) <= 18, f"Scene {i} voiceover has {len(words)} words (max 18)"


def test_deterministic_output():
    prop = _make_prop(num_features=3)
    images = [Path(f"img_{i}.jpg") for i in range(5)]

    plan1 = build_scene_plan(prop, images)
    plan2 = build_scene_plan(prop, images)

    assert plan1.total_duration == plan2.total_duration
    for s1, s2 in zip(plan1.scenes, plan2.scenes):
        assert s1.scene_type == s2.scene_type
        assert s1.caption == s2.caption
        assert s1.voiceover == s2.voiceover
        assert s1.duration == s2.duration
        assert s1.image_path == s2.image_path
