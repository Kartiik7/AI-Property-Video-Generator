"""
test_llm_client.py — Tests for the LLM client with mocked HTTP layer.

Verifies:
  1. all-valid LLM response (all scenes used, source="llm")
  2. one bad scene (others kept, source="mixed (N of M scenes from LLM)")
  3. retry fixing a bad scene (2 calls, problem description sent, source="llm")
  4. retry still failing (2 calls, bad scene replaced with template)
  5. 429 rate limit error (clear warning, source="template")
  6. timeout error (20s timeout caught cleanly, source="template")
  7. malformed JSON (clear warning, source="template")
  8. no key (LLM_API_KEY not set, clean fallback)
  9. default env behavior & --no-llm precedence
  10. custom_script takes priority for scenes it covers
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
        {"scene_index": 1, "scene_type": "title", "caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome."},
        {"scene_index": 2, "scene_type": "feature", "feature": "Covered Parking", "caption": "Covered Parking", "voiceover": "Parking."},
        {"scene_index": 3, "scene_type": "feature", "feature": "Modular Kitchen", "caption": "Modular Kitchen", "voiceover": "Kitchen."},
        {"scene_index": 4, "scene_type": "price_location", "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Price."},
        {"scene_index": 5, "scene_type": "cta", "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "CTA."},
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


# ── 1. No API Key ───────────────────────────────────────────────────────────

def test_no_key_falls_back_to_template(sample_prop, skeleton):
    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with patch.dict("os.environ", {"LLM_API_KEY": ""}):
        scenes, warnings = generate_scene_copy(sample_prop, skeleton)
        assert scenes is None
        assert any("LLM_API_KEY is not set" in w for w in warnings)

        plan = build_scene_plan(sample_prop, images, use_llm=True)
        assert plan.source == "template"
        assert len(plan.scenes) == 5


# ── 2. All-Valid LLM Response ───────────────────────────────────────────────

def test_all_valid_llm_response(sample_prop, skeleton):
    valid_content = json.dumps({
        "scenes": [
            {"scene_index": 1, "caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
            {"scene_index": 2, "caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
            {"scene_index": 3, "caption": "Modular Kitchen", "voiceover": "Features a modern modular kitchen."},
            {"scene_index": 4, "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
            {"scene_index": 5, "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today, contact details on screen."},
        ]
    })

    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(valid_content)):
            scenes, warnings = generate_scene_copy(sample_prop, skeleton)
            assert scenes is not None
            assert len(scenes) == 5
            assert all(s is not None for s in scenes)
            assert scenes[0]["caption"] == "Green Valley 2BHK\nApartment"

            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.source == "llm"
            assert plan.scenes[0].caption == "Green Valley 2BHK\nApartment"


# ── 3. One Bad Scene (Others Kept) ──────────────────────────────────────────

def test_one_bad_scene_others_kept(sample_prop, skeleton):
    # Scene 3 has an unmentioned number 500 and retry also fails to fix it
    bad_initial = json.dumps({
        "scenes": [
            {"scene_index": 1, "caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
            {"scene_index": 2, "caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
            {"scene_index": 3, "caption": "Kitchen 500 sq ft", "voiceover": "Features a 500 sq ft modular kitchen."},
            {"scene_index": 4, "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
            {"scene_index": 5, "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today, details on screen."},
        ]
    })
    bad_retry = json.dumps({
        "scenes": [
            {"scene_index": 3, "caption": "Still 500 sq ft", "voiceover": "Still has number 500."},
        ]
    })

    images = [Path(f"img_{i}.jpg") for i in range(5)]
    mock_responses = [_mock_http_response(bad_initial), _mock_http_response(bad_retry)]

    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        with patch("urllib.request.urlopen", side_effect=mock_responses):
            scenes, warnings = generate_scene_copy(sample_prop, skeleton)
            assert scenes is not None
            assert scenes[0] is not None  # Kept
            assert scenes[1] is not None  # Kept
            assert scenes[2] is None      # Bad scene replaced
            assert scenes[3] is not None  # Kept
            assert scenes[4] is not None  # Kept

            # Verify warning for the replaced scene
            assert any("Scene 3 LLM copy failed" in w for w in warnings)

    # In build_scene_plan: plan source is "mixed (4 of 5 scenes from LLM)"
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        with patch("urllib.request.urlopen", side_effect=[_mock_http_response(bad_initial), _mock_http_response(bad_retry)]):
            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.source == "mixed (4 of 5 scenes from LLM)"
            # Scene 3 fell back to template text
            assert plan.scenes[2].caption == "Modular Kitchen"
            # Scene 1, 2, 4, 5 are from LLM
            assert plan.scenes[0].caption == "Green Valley 2BHK\nApartment"


# ── 4. Retry Fixing a Bad Scene ─────────────────────────────────────────────

def test_retry_fixing_a_bad_scene(sample_prop, skeleton):
    # Initial: scene 3 has foreign number 500
    bad_initial = json.dumps({
        "scenes": [
            {"scene_index": 1, "caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
            {"scene_index": 2, "caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
            {"scene_index": 3, "caption": "Kitchen 500 sq ft", "voiceover": "Features a 500 sq ft modular kitchen."},
            {"scene_index": 4, "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
            {"scene_index": 5, "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today, details on screen."},
        ]
    })
    # Retry: correctly fixed scene 3
    fixed_retry = json.dumps({
        "scenes": [
            {"scene_index": 3, "caption": "Modular Kitchen", "voiceover": "Features a modular kitchen."},
        ]
    })

    recorded_requests = []

    def fake_urlopen(req, timeout=20):
        recorded_requests.append(req)
        if len(recorded_requests) == 1:
            return _mock_http_response(bad_initial)
        return _mock_http_response(fixed_retry)

    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            scenes, warnings = generate_scene_copy(sample_prop, skeleton)
            # 2 calls were made
            assert len(recorded_requests) == 2

            # Check that the retry prompt listed the problem with scene 3 and number 500
            retry_body = json.loads(recorded_requests[1].data.decode("utf-8"))
            retry_prompt = retry_body["messages"][-1]["content"]
            assert "scene 3" in retry_prompt.lower()
            assert "500" in retry_prompt

            # All 5 scenes are now valid
            assert scenes is not None
            assert all(s is not None for s in scenes)
            assert scenes[2]["caption"] == "Modular Kitchen"
            assert scenes[2]["voiceover"] == "Features a modular kitchen."

    # Test through build_scene_plan gives plan.source == "llm"
    recorded_requests.clear()
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.source == "llm"
            assert plan.scenes[2].caption == "Modular Kitchen"


# ── 5. Retry Still Failing ──────────────────────────────────────────────────

def test_retry_still_failing(sample_prop, skeleton):
    bad_initial = json.dumps({
        "scenes": [
            {"scene_index": 1, "caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
            {"scene_index": 2, "caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
            {"scene_index": 3, "caption": "Kitchen 500 sq ft", "voiceover": "Features 500 sq ft."},
            {"scene_index": 4, "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
            {"scene_index": 5, "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today, details on screen."},
        ]
    })
    # Retry still contains luxury buzzword
    still_bad_retry = json.dumps({
        "scenes": [
            {"scene_index": 3, "caption": "Luxury Kitchen", "voiceover": "Features a luxury kitchen."},
        ]
    })

    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        with patch("urllib.request.urlopen", side_effect=[_mock_http_response(bad_initial), _mock_http_response(still_bad_retry)]):
            scenes, warnings = generate_scene_copy(sample_prop, skeleton)
            assert scenes is not None
            assert scenes[2] is None
            assert any("after retry" in w for w in warnings)


# ── 6. 429 Rate Limit Error ─────────────────────────────────────────────────

def test_rate_limit_429_error(sample_prop, skeleton):
    mock_err = urllib.error.HTTPError(
        url="https://api.openai.com/v1/chat/completions",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=None,
    )

    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        with patch("urllib.request.urlopen", side_effect=mock_err):
            scenes, warnings = generate_scene_copy(sample_prop, skeleton)
            assert scenes is None
            assert any("429" in w or "rate limit" in w.lower() for w in warnings)

            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.source == "template"


# ── 7. Timeout Error ────────────────────────────────────────────────────────

def test_timeout_error(sample_prop, skeleton):
    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")):
            scenes, warnings = generate_scene_copy(sample_prop, skeleton)
            assert scenes is None
            assert any("timed out" in w.lower() for w in warnings)

            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.source == "template"


# ── 8. Malformed JSON ───────────────────────────────────────────────────────

def test_malformed_json_falls_back(sample_prop, skeleton):
    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response("Not JSON at all! {invalid")):
            scenes, warnings = generate_scene_copy(sample_prop, skeleton)
            assert scenes is None
            assert any("malformed" in w.lower() or "json" in w.lower() for w in warnings)

            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.source == "template"


# ── 9. Default Environment Behavior & Precedence ────────────────────────────

def test_default_env_uses_llm_when_key_set(sample_prop):
    valid_content = json.dumps({
        "scenes": [
            {"scene_index": 1, "caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
            {"scene_index": 2, "caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
            {"scene_index": 3, "caption": "Modular Kitchen", "voiceover": "Features a modern modular kitchen."},
            {"scene_index": 4, "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
            {"scene_index": 5, "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today, details on screen."},
        ]
    })
    images = [Path(f"img_{i}.jpg") for i in range(5)]

    # When LLM_API_KEY is in env, build_scene_plan uses LLM automatically (use_llm=None)
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(valid_content)):
            plan = build_scene_plan(sample_prop, images)  # default use_llm=None
            assert plan.source == "llm"


def test_no_llm_forces_template_even_when_key_set(sample_prop):
    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        # Explicit use_llm=False (--no-llm flag behavior)
        plan = build_scene_plan(sample_prop, images, use_llm=False)
        assert plan.source == "template"


# ── 10. Custom Script Takes Priority ────────────────────────────────────────

def test_custom_script_takes_priority_over_llm(sample_prop):
    sample_prop.custom_script = {
        "title": {
            "caption": "Custom Green Valley\nApartment",
            "voiceover": "Welcome to our custom property title.",
        }
    }
    llm_content = json.dumps({
        "scenes": [
            {"scene_index": 1, "caption": "LLM Title\nApartment", "voiceover": "LLM title voiceover."},
            {"scene_index": 2, "caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
            {"scene_index": 3, "caption": "Modular Kitchen", "voiceover": "Features a modern modular kitchen."},
            {"scene_index": 4, "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
            {"scene_index": 5, "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today, details on screen."},
        ]
    })
    images = [Path(f"img_{i}.jpg") for i in range(5)]

    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(llm_content)):
            plan = build_scene_plan(sample_prop, images, use_llm=True)
            # Custom script took priority for scene 0 (title)
            assert plan.scenes[0].caption == "Custom Green Valley\nApartment"
            assert plan.scenes[0].voiceover == "Welcome to our custom property title."
            # Scenes 1-4 came from LLM
            assert plan.scenes[1].caption == "Covered Parking"
            # 4 of 5 scenes from LLM -> mixed
            assert plan.source == "mixed (4 of 5 scenes from LLM)"
