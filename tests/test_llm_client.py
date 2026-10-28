"""
test_llm_client.py — Tests for the optional LLM client using mocked HTTP layer.

Verifies:
  - Missing API key falls back to template with clear warning
  - Valid LLM response is accepted and used in scene plan
  - Hallucinated facts (e.g. invented gym, luxury) are rejected and fall back
  - Malformed JSON falls back cleanly
  - Wrong scene count falls back cleanly
  - Network/HTTP error falls back cleanly
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import urllib.error

from src.models import PropertyData
from src.llm_client import generate_scene_copy
from src.script_generator import build_scene_plan


@pytest.fixture
def sample_prop() -> PropertyData:
    return PropertyData(
        name="Green Valley 2BHK",
        location="Baner, Pune",
        price="₹45,00,000",
        property_type="Apartment",
        features=["Covered Parking", "Modular Kitchen"],
        cta_text="Book a site visit today",
        contact="+91-00000-00000",
        tagline="A serene home for modern living",
    )


@pytest.fixture
def skeleton() -> list[dict]:
    return [
        {"scene_type": "title", "caption": "Title", "voiceover": "Welcome."},
        {"scene_type": "feature", "caption": "Feat 1", "voiceover": "Feat 1."},
        {"scene_type": "feature", "caption": "Feat 2", "voiceover": "Feat 2."},
        {"scene_type": "price_location", "caption": "Price", "voiceover": "Price."},
        {"scene_type": "cta", "caption": "CTA", "voiceover": "CTA."},
    ]


def _mock_http_response(content_str: str):
    """Create a mock HTTP response context manager returning JSON with *content_str*."""
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content_str,
                }
            }
        ]
    }
    raw_bytes = json.dumps(response_data).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw_bytes
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


def test_missing_api_key_falls_back(sample_prop, skeleton):
    with patch.dict("os.environ", {"LLM_API_KEY": ""}):
        scenes, warnings = generate_scene_copy(sample_prop, skeleton)
        assert scenes is None
        assert any("LLM_API_KEY is not set" in w for w in warnings)


def test_valid_llm_response_accepted(sample_prop, skeleton):
    valid_content = json.dumps({
        "scenes": [
            {"caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
            {"caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
            {"caption": "Modular Kitchen", "voiceover": "Features a modern modular kitchen."},
            {"caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
            {"caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today at +91-00000-00000."},
        ]
    })

    with patch.dict("os.environ", {"LLM_API_KEY": "fake_test_key"}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(valid_content)):
            scenes, warnings = generate_scene_copy(sample_prop, skeleton)
            assert scenes is not None
            assert len(scenes) == len(skeleton)
            assert scenes[0]["caption"] == "Green Valley 2BHK\nApartment"


def test_hallucinated_amenity_rejected_and_falls_back(sample_prop, skeleton):
    # LLM hallucinates "Swimming pool and gym" which are NOT in sample_prop
    hallucinated_content = json.dumps({
        "scenes": [
            {"caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK."},
            {"caption": "Covered Parking", "voiceover": "Includes covered parking."},
            {"caption": "Gym and Pool", "voiceover": "Includes world class gym and swimming pool."},
            {"caption": "₹45,00,000\nBaner", "voiceover": "Priced at 45 lakh rupees."},
            {"caption": "Book now", "voiceover": "Book a site visit today."},
        ]
    })

    with patch.dict("os.environ", {"LLM_API_KEY": "fake_test_key"}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(hallucinated_content)):
            scenes, warnings = generate_scene_copy(sample_prop, skeleton)
            assert scenes is None
            assert any("fact check" in w.lower() for w in warnings)


def test_banned_claim_rejected_and_falls_back(sample_prop, skeleton):
    # LLM adds "luxury" which is NOT in sample_prop
    banned_content = json.dumps({
        "scenes": [
            {"caption": "Luxury Green Valley\nApartment", "voiceover": "Welcome to luxury living."},
            {"caption": "Covered Parking", "voiceover": "Includes covered parking."},
            {"caption": "Modular Kitchen", "voiceover": "Includes modular kitchen."},
            {"caption": "₹45,00,000\nBaner", "voiceover": "Priced at 45 lakh rupees."},
            {"caption": "Book now", "voiceover": "Book a site visit today."},
        ]
    })

    with patch.dict("os.environ", {"LLM_API_KEY": "fake_test_key"}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(banned_content)):
            scenes, warnings = generate_scene_copy(sample_prop, skeleton)
            assert scenes is None
            assert any("fact check" in w.lower() or "banned" in w.lower() for w in warnings)


def test_malformed_json_falls_back(sample_prop, skeleton):
    with patch.dict("os.environ", {"LLM_API_KEY": "fake_test_key"}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response("Not a JSON string at all")):
            scenes, warnings = generate_scene_copy(sample_prop, skeleton)
            assert scenes is None
            assert any("malformed" in w.lower() or "json" in w.lower() for w in warnings)


def test_wrong_scene_count_falls_back(sample_prop, skeleton):
    # Returns only 2 scenes when 5 are expected
    short_content = json.dumps({
        "scenes": [
            {"caption": "Scene 1", "voiceover": "Welcome."},
            {"caption": "Scene 2", "voiceover": "Goodbye."},
        ]
    })

    with patch.dict("os.environ", {"LLM_API_KEY": "fake_test_key"}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(short_content)):
            scenes, warnings = generate_scene_copy(sample_prop, skeleton)
            assert scenes is None
            assert any("expected 5" in w.lower() for w in warnings)


def test_network_error_falls_back(sample_prop, skeleton):
    with patch.dict("os.environ", {"LLM_API_KEY": "fake_test_key"}):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            scenes, warnings = generate_scene_copy(sample_prop, skeleton)
            assert scenes is None
            assert any("network error" in w.lower() for w in warnings)


def test_build_scene_plan_with_llm_flag_uses_llm(sample_prop):
    valid_content = json.dumps({
        "scenes": [
            {"caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
            {"caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
            {"caption": "Modular Kitchen", "voiceover": "Features a modern modular kitchen."},
            {"caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
            {"caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today at +91-00000-00000."},
        ]
    })

    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with patch.dict("os.environ", {"LLM_API_KEY": "fake_test_key"}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(valid_content)):
            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.source == "llm"
            assert plan.scenes[0].caption == "Green Valley 2BHK\nApartment"
