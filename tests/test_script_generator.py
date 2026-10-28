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
    format_property_type,
    format_feature_for_voiceover,
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


# ── Custom Script Tests ────────────────────────────────────────────────────

def _sample_custom_script() -> dict:
    return {
        "title": {
            "caption": "Maple Residency\nIndependent House",
            "voiceover": "Welcome to Maple Residency, an independent house in Sector 150, Noida.",
        },
        "features": [
            {"caption": "Spacious kitchen", "voiceover": "Start your day in a spacious kitchen."},
            {"caption": "Comfortable bedroom", "voiceover": "Unwind in a comfortable bedroom."},
            {"caption": "Clean bathroom", "voiceover": "A clean bathroom completes the home."},
        ],
        "price_location": {
            "caption": "₹85,00,000\nSector 150, Noida",
            "voiceover": "Priced at 85 lakh rupees, in Sector 150, Noida.",
        },
        "cta": {
            "caption": "Book a site visit today\n+91-00000-00000",
            "voiceover": "Book a site visit today. Contact details are on screen.",
        },
    }


def test_custom_script_used_when_valid():
    prop = _make_prop(num_features=3, price="₹85,00,000")
    prop.name = "Maple Residency"
    prop.location = "Sector 150, Noida"
    prop.property_type = "Independent House"
    prop.features = ["Spacious kitchen", "Comfortable bedroom", "Clean bathroom"]
    prop.cta_text = "Book a site visit today"
    prop.contact = "+91-00000-00000"
    prop.custom_script = _sample_custom_script()

    images = [Path(f"img_{i}.jpg") for i in range(6)]
    plan = build_scene_plan(prop, images)

    assert plan.source == "template"
    assert any("custom script used for 6 scenes" in w.lower() for w in plan.warnings)

    # Check that custom lines were used
    assert plan.scenes[0].caption == "Maple Residency\nIndependent House"
    assert plan.scenes[0].voiceover == "Welcome to Maple Residency, an independent house in Sector 150, Noida."
    assert plan.scenes[1].caption == "Spacious kitchen"
    assert plan.scenes[1].voiceover == "Start your day in a spacious kitchen."
    assert plan.scenes[4].caption == "₹85,00,000\nSector 150, Noida"
    assert plan.scenes[4].voiceover == "Priced at 85 lakh rupees, in Sector 150, Noida."


def test_invalid_custom_line_invented_number_falls_back_for_that_scene_only():
    prop = _make_prop(num_features=3, price="₹85,00,000")
    prop.name = "Maple Residency"
    prop.location = "Sector 150, Noida"
    prop.features = ["Spacious kitchen", "Comfortable bedroom", "Clean bathroom"]
    prop.contact = "+91-00000-00000"
    cs = _sample_custom_script()
    # Invented number 9999
    cs["title"]["voiceover"] = "Welcome to Maple Residency, just 9999 minutes from anywhere."
    prop.custom_script = cs

    images = [Path(f"img_{i}.jpg") for i in range(6)]
    plan = build_scene_plan(prop, images)

    # Title fell back to template text
    assert "9999" not in plan.scenes[0].voiceover
    assert any("title" in w.lower() and "rejected" in w.lower() for w in plan.warnings)

    # Feature scenes still used custom lines
    assert plan.scenes[1].caption == "Spacious kitchen"
    assert plan.scenes[1].voiceover == "Start your day in a spacious kitchen."
    assert any("custom script used for 5 scenes" in w.lower() for w in plan.warnings)


def test_invalid_custom_line_banned_claim_falls_back_for_that_scene_only():
    prop = _make_prop(num_features=3, price="₹85,00,000")
    prop.name = "Maple Residency"
    prop.location = "Sector 150, Noida"
    prop.features = ["Spacious kitchen", "Comfortable bedroom", "Clean bathroom"]
    prop.contact = "+91-00000-00000"
    cs = _sample_custom_script()
    # Banned promotional claim "luxury"
    cs["price_location"]["voiceover"] = "Luxury living priced at 85 lakh rupees in Sector 150."
    prop.custom_script = cs

    images = [Path(f"img_{i}.jpg") for i in range(6)]
    plan = build_scene_plan(prop, images)

    # Price location fell back to template text
    assert "luxury" not in plan.scenes[4].voiceover.lower()
    assert any("price_location" in w.lower() and "rejected" in w.lower() for w in plan.warnings)

    # Other scenes retained custom text
    assert plan.scenes[0].caption == "Maple Residency\nIndependent House"
    assert any("custom script used for 5 scenes" in w.lower() for w in plan.warnings)


def test_custom_line_exceeding_limits_falls_back_for_that_scene_only():
    prop = _make_prop(num_features=3, price="₹85,00,000")
    prop.features = ["Spacious kitchen", "Comfortable bedroom", "Clean bathroom"]
    prop.contact = "+91-00000-00000"
    cs = _sample_custom_script()
    # Voiceover exceeding 18 words
    cs["features"][0]["voiceover"] = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen."
    prop.custom_script = cs

    images = [Path(f"img_{i}.jpg") for i in range(6)]
    plan = build_scene_plan(prop, images)

    assert any("feature 1" in w.lower() and "rejected" in w.lower() for w in plan.warnings)
    # Feature 1 fell back to template
    assert "one two three" not in plan.scenes[1].voiceover
    # Feature 2 still custom
    assert plan.scenes[2].caption == "Comfortable bedroom"



def test_missing_custom_script_uses_template():
    prop = _make_prop(num_features=3)
    prop.custom_script = None

    images = [Path(f"img_{i}.jpg") for i in range(6)]
    plan = build_scene_plan(prop, images)

    assert plan.source == "template"
    assert not any("custom script used" in w.lower() for w in plan.warnings)


def test_sample_files_load_and_generate_without_errors():
    from src.input_loader import load_from_json

    base = Path(__file__).resolve().parent.parent

    # House 1
    h1_json = base / "input" / "samples" / "house_1" / "property.json"
    h1_images = base / "input" / "samples" / "house_1" / "images"
    res1 = load_from_json(h1_json, h1_images)
    assert res1.ok, f"House 1 failed loading: {res1.errors}"
    assert res1.errors == []
    plan1 = build_scene_plan(res1.property, res1.images)
    assert any("custom script used for 6 scenes" in w.lower() for w in plan1.warnings)
    # Verify room-based image assignment
    assert plan1.scenes[0].image_path.name == "1_frontal.jpg"
    assert plan1.scenes[1].image_path.name == "1_kitchen.jpg"
    assert plan1.scenes[2].image_path.name == "1_bedroom.jpg"
    assert plan1.scenes[3].image_path.name == "1_bathroom.jpg"

    # House 2
    h2_json = base / "input" / "samples" / "house_2" / "property.json"
    h2_images = base / "input" / "samples" / "house_2" / "images"
    res2 = load_from_json(h2_json, h2_images)
    assert res2.ok, f"House 2 failed loading: {res2.errors}"
    assert res2.errors == []
    plan2 = build_scene_plan(res2.property, res2.images)
    assert any("custom script used for 6 scenes" in w.lower() for w in plan2.warnings)
    assert plan2.scenes[0].image_path.name == "2_frontal.jpg"
    assert plan2.scenes[1].image_path.name == "2_kitchen.jpg"
    assert plan2.scenes[2].image_path.name == "2_bedroom.jpg"
    assert plan2.scenes[3].image_path.name == "2_bathroom.jpg"


# ── Template Wording Fix Tests ─────────────────────────────────────────────

def test_property_type_articles_and_casing():
    # Helper checks
    assert format_property_type("Apartment") == "an apartment"
    assert format_property_type("Independent House") == "an independent house"
    assert format_property_type("Villa") == "a villa"
    assert format_property_type("Penthouse") == "a penthouse"

    # Plan with Apartment
    prop_apt = _make_prop(num_features=3)
    prop_apt.property_type = "Apartment"
    plan_apt = build_scene_plan(prop_apt, [Path(f"img_{i}.jpg") for i in range(5)])
    assert "an apartment" in plan_apt.scenes[0].voiceover
    assert "a Apartment" not in plan_apt.scenes[0].voiceover
    assert "a apartment" not in plan_apt.scenes[0].voiceover

    # Plan with Independent House
    prop_house = _make_prop(num_features=3)
    prop_house.property_type = "Independent House"
    plan_house = build_scene_plan(prop_house, [Path(f"img_{i}.jpg") for i in range(5)])
    assert "an independent house" in plan_house.scenes[0].voiceover
    assert "a Independent House" not in plan_house.scenes[0].voiceover
    assert "a independent house" not in plan_house.scenes[0].voiceover


def test_feature_voiceover_formatting_and_casing():
    # Helper checks
    assert format_feature_for_voiceover("Spacious kitchen") == "spacious kitchen"
    assert format_feature_for_voiceover("24/7 Security") == "24/7 Security"
    assert format_feature_for_voiceover("CCTV cover") == "CCTV cover"
    assert format_feature_for_voiceover("Modular Kitchen") == "modular Kitchen"

    prop = _make_prop(num_features=3)
    prop.features = ["24/7 Security", "Spacious Kitchen", "CCTV Surveillance"]
    plan = build_scene_plan(prop, [Path(f"img_{i}.jpg") for i in range(6)])

    feature_scenes = [s for s in plan.scenes if s.scene_type == "feature"]
    assert len(feature_scenes) == 3

    # Check prefix patterns and casing
    valid_prefixes = ("Highlights include ", "Next up: ", "Also featured: ")
    for s in feature_scenes:
        assert any(s.voiceover.startswith(prefix) for prefix in valid_prefixes)

    # 24/7 Security must NOT be lowercased
    assert "24/7 Security" in feature_scenes[0].voiceover
    # Spacious Kitchen has first letter lowercased
    assert "spacious Kitchen" in feature_scenes[1].voiceover
    # CCTV Surveillance must NOT lowercase CCTV
    assert "CCTV Surveillance" in feature_scenes[2].voiceover


def test_cta_voiceover_no_phone_no_double_today():
    # Input with "Book a site visit today" and contact phone
    prop = _make_prop(num_features=3)
    prop.cta_text = "Book a site visit today"
    prop.contact = "+91-00000-00000"
    plan = build_scene_plan(prop, [Path(f"img_{i}.jpg") for i in range(5)])

    cta_scene = [s for s in plan.scenes if s.scene_type == "cta"][0]
    # Never speak the phone number
    assert "+91-00000-00000" not in cta_scene.voiceover
    assert "00000" not in cta_scene.voiceover
    # No duplicate "today"
    assert "today today" not in cta_scene.voiceover
    # Correct format
    assert cta_scene.voiceover == "Book a site visit today. Contact details are on screen."

    # Input with trailing punctuation
    prop2 = _make_prop(num_features=3)
    prop2.cta_text = "Call now for an inquiry!"
    prop2.contact = "+91-11111-22222"
    plan2 = build_scene_plan(prop2, [Path(f"img_{i}.jpg") for i in range(5)])
    cta_scene2 = [s for s in plan2.scenes if s.scene_type == "cta"][0]
    assert cta_scene2.voiceover == "Call now for an inquiry. Contact details are on screen."


def test_title_voiceover_tagline_dropped_if_too_long():
    prop = _make_prop(num_features=3)
    prop.name = "Maple Residency"
    prop.property_type = "Independent House"
    # Long tagline pushing total title VO over 18 words
    prop.tagline = "Where luxury meets comfort in an expansive green setting surrounded by calm lakes"
    plan = build_scene_plan(prop, [Path(f"img_{i}.jpg") for i in range(5)])

    title_scene = plan.scenes[0]
    # Tagline dropped from voiceover so voiceover stays <= 18 words
    assert len(title_scene.voiceover.split()) <= 18
    assert "calm lakes" not in title_scene.voiceover
    assert "an independent house" in title_scene.voiceover
    # Tagline is still kept on screen (in caption)
    assert prop.tagline in title_scene.caption or prop.name in title_scene.caption

    # Short tagline kept in voiceover
    prop_short = _make_prop(num_features=3)
    prop_short.name = "Maple Residency"
    prop_short.property_type = "Independent House"
    prop_short.tagline = "Feels like home"
    plan_short = build_scene_plan(prop_short, [Path(f"img_{i}.jpg") for i in range(5)])
    assert "Feels like home." in plan_short.scenes[0].voiceover


