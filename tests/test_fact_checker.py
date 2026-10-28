"""
test_fact_checker.py — Tests for pure fact checking functions.

Verifies:
  - Allowed facts extraction
  - Foreign numbers flagging
  - Unmentioned amenities flagging
  - Banned promotional claims flagging
  - Legitimate words passing
  - Banned word permitted if explicitly present in input
  - Template output self-check passes
"""

import pytest
from src.models import PropertyData
from src.fact_checker import extract_allowed_facts, check_text
from src.script_generator import build_scene_plan
from pathlib import Path


@pytest.fixture
def sample_prop() -> PropertyData:
    return PropertyData(
        name="Green Valley 2BHK",
        location="Baner, Pune",
        price="₹45,00,000",
        property_type="Apartment",
        features=["Covered Parking", "Modular Kitchen", "Power Backup"],
        cta_text="Book a site visit today",
        contact="+91-98765-43210",
        tagline="A serene home for modern living",
    )


def test_clean_text_passes(sample_prop):
    allowed = extract_allowed_facts(sample_prop)
    text = "Covered Parking and Modular Kitchen available at Green Valley 2BHK."
    ok, problems = check_text(text, allowed)
    assert ok
    assert problems == []


def test_invented_number_flagged(sample_prop):
    allowed = extract_allowed_facts(sample_prop)
    # 5 is not in the facts (facts have 2 from 2BHK, 45, 9876543210)
    text = "Only 5 minutes away from highway."
    ok, problems = check_text(text, allowed)
    assert not ok
    assert any("5" in p for p in problems)


def test_invented_amenity_flagged(sample_prop):
    allowed = extract_allowed_facts(sample_prop)
    # "gym" is not in sample_prop features
    text = "Enjoy the state of the art gym."
    ok, problems = check_text(text, allowed)
    assert not ok
    assert any("gym" in p.lower() for p in problems)


def test_swimming_pool_flagged_when_not_in_input(sample_prop):
    allowed = extract_allowed_facts(sample_prop)
    text = "Includes swimming pool for residents."
    ok, problems = check_text(text, allowed)
    assert not ok
    assert any("swimming pool" in p.lower() or "pool" in p.lower() for p in problems)


def test_allowed_amenity_passes(sample_prop):
    allowed = extract_allowed_facts(sample_prop)
    text = "Includes covered parking and modular kitchen."
    ok, problems = check_text(text, allowed)
    assert ok


def test_banned_claim_luxury_flagged(sample_prop):
    allowed = extract_allowed_facts(sample_prop)
    text = "Experience luxury living in Baner."
    ok, problems = check_text(text, allowed)
    assert not ok
    assert any("luxury" in p.lower() for p in problems)


def test_banned_claim_rera_flagged(sample_prop):
    allowed = extract_allowed_facts(sample_prop)
    text = "RERA approved apartment in Pune."
    ok, problems = check_text(text, allowed)
    assert not ok
    assert any("rera" in p.lower() for p in problems)


def test_banned_claim_allowed_if_in_input():
    prop = PropertyData(
        name="Grand Heights",
        location="Mumbai",
        price="₹1,00,00,000",
        property_type="Apartment",
        features=["Gym"],
        cta_text="Call now",
        tagline="Affordable luxury in the heart of Mumbai",  # "luxury" is in the input
    )
    allowed = extract_allowed_facts(prop)
    text = "Affordable luxury apartment in Mumbai."
    ok, problems = check_text(text, allowed)
    assert ok, f"Expected luxury to be allowed when in input, got problems: {problems}"


def test_template_output_always_passes_fact_check(sample_prop):
    images = [Path(f"img{i}.jpg") for i in range(5)]
    plan = build_scene_plan(sample_prop, images)

    allowed = extract_allowed_facts(sample_prop)
    for i, scene in enumerate(plan.scenes):
        ok_cap, prob_cap = check_text(scene.caption, allowed)
        assert ok_cap, f"Scene {i+1} caption failed fact check: {prob_cap}"

        ok_vo, prob_vo = check_text(scene.voiceover, allowed)
        assert ok_vo, f"Scene {i+1} voiceover failed fact check: {prob_vo}"
