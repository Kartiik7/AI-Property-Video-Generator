"""
test_llm_client.py — Tests for the LLM client with mocked HTTP layer.

Verifies:
  1. no key (LLM_API_KEY not set, clean fallback)
  2. all-valid LLM response (all scenes used, source="llm (primary)")
  3. one bad scene (others kept, source="mixed (N of M scenes from llm (primary))")
  4. retry fixing a bad scene (2 calls, problem description sent, source="llm (primary)")
  5. retry still failing (2 calls, bad scene replaced with template)
  6. 429 rate limit error (clear warning, source="template")
  7. timeout error (20s timeout caught cleanly, source="template")
  8. malformed JSON (clear warning, source="template")
  9. default env behavior & --no-llm precedence
  10. custom_script takes priority for scenes it covers
  11. GEMINI_API_KEY support
  12. primary 503 then success on retry
  13. primary fails twice then fallback succeeds
  14. both providers fail -> template
  15. 401 does not retry and warns "check LLM_API_KEY"
  16. JSON wrapped in fences or extra text parses
  17. keys never appear in logs or warnings
"""

import json
import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import urllib.error

from src.models import PropertyData
from src.llm_client import generate_scene_copy
from src.script_generator import build_scene_plan


@pytest.fixture(autouse=True)
def fast_retry_sleep():
    """Bypass real 2s sleep backoff during unit tests so test suite runs instantaneously."""
    with patch("time.sleep", return_value=None):
        yield


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


def _mock_http_error(code: int, msg: str = "Error"):
    """Create a mock HTTPError with the given status code."""
    return urllib.error.HTTPError(
        url="https://api.openai.com/v1/chat/completions",
        code=code,
        msg=msg,
        hdrs={},
        fp=None,
    )


# ── 1. No API Key ───────────────────────────────────────────────────────────

def test_no_key_falls_back_to_template(sample_prop, skeleton):
    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with patch.dict("os.environ", {"LLM_API_KEY": "", "GEMINI_API_KEY": "", "LLM_FALLBACK_API_KEY": ""}):
        scenes, warnings, *rest = generate_scene_copy(sample_prop, skeleton)
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
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key", "LLM_FALLBACK_API_KEY": ""}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(valid_content)):
            scenes, warnings, provider = generate_scene_copy(sample_prop, skeleton)
            assert scenes is not None
            assert len(scenes) == 5
            assert all(s is not None for s in scenes)
            assert scenes[0]["caption"] == "Green Valley 2BHK\nApartment"
            assert provider == "primary"

            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.source == "llm (primary)"
            assert plan.scenes[0].caption == "Green Valley 2BHK\nApartment"


# ── 3. One Bad Scene (Others Kept) ──────────────────────────────────────────

def test_one_bad_scene_others_kept(sample_prop, skeleton):
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

    with patch.dict("os.environ", {"LLM_API_KEY": "test_key", "LLM_FALLBACK_API_KEY": ""}):
        with patch("urllib.request.urlopen", side_effect=mock_responses):
            scenes, warnings, provider = generate_scene_copy(sample_prop, skeleton)
            assert scenes is not None
            assert scenes[0] is not None
            assert scenes[1] is not None
            assert scenes[2] is None      # Bad scene replaced
            assert scenes[3] is not None
            assert scenes[4] is not None
            assert any("Scene 3 LLM copy failed" in w for w in warnings)

    with patch.dict("os.environ", {"LLM_API_KEY": "test_key", "LLM_FALLBACK_API_KEY": ""}):
        with patch("urllib.request.urlopen", side_effect=[_mock_http_response(bad_initial), _mock_http_response(bad_retry)]):
            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.source == "mixed (4 of 5 scenes from llm (primary))"
            assert plan.scenes[2].caption == "Modular Kitchen"
            assert plan.scenes[0].caption == "Green Valley 2BHK\nApartment"


# ── 4. Retry Fixing a Bad Scene ─────────────────────────────────────────────

def test_retry_fixing_a_bad_scene(sample_prop, skeleton):
    bad_initial = json.dumps({
        "scenes": [
            {"scene_index": 1, "caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
            {"scene_index": 2, "caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
            {"scene_index": 3, "caption": "Kitchen 500 sq ft", "voiceover": "Features a 500 sq ft modular kitchen."},
            {"scene_index": 4, "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
            {"scene_index": 5, "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today, details on screen."},
        ]
    })
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
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key", "LLM_FALLBACK_API_KEY": ""}):
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            scenes, warnings, provider = generate_scene_copy(sample_prop, skeleton)
            assert len(recorded_requests) == 2
            assert scenes is not None
            assert all(s is not None for s in scenes)
            assert scenes[2]["caption"] == "Modular Kitchen"
            assert provider == "primary"

    recorded_requests.clear()
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key", "LLM_FALLBACK_API_KEY": ""}):
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.source == "llm (primary)"
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
    still_bad_retry = json.dumps({
        "scenes": [
            {"scene_index": 3, "caption": "Luxury Kitchen", "voiceover": "Features a luxury kitchen."},
        ]
    })

    with patch.dict("os.environ", {"LLM_API_KEY": "test_key", "LLM_FALLBACK_API_KEY": ""}):
        with patch("urllib.request.urlopen", side_effect=[_mock_http_response(bad_initial), _mock_http_response(still_bad_retry)]):
            scenes, warnings, provider = generate_scene_copy(sample_prop, skeleton)
            assert scenes is not None
            assert scenes[2] is None
            assert any("after retry" in w for w in warnings)


# ── 6. 429 Rate Limit Error ─────────────────────────────────────────────────

def test_rate_limit_429_error(sample_prop, skeleton):
    mock_err = _mock_http_error(429, "Too Many Requests")
    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key", "LLM_FALLBACK_API_KEY": ""}):
        with patch("urllib.request.urlopen", side_effect=mock_err):
            scenes, warnings, provider = generate_scene_copy(sample_prop, skeleton)
            assert scenes is None
            assert any("429" in w or "rate limit" in w.lower() for w in warnings)

            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.source == "template"


# ── 7. Timeout Error ────────────────────────────────────────────────────────

def test_timeout_error(sample_prop, skeleton):
    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key", "LLM_FALLBACK_API_KEY": ""}):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")):
            scenes, warnings, provider = generate_scene_copy(sample_prop, skeleton)
            assert scenes is None
            assert any("timed out" in w.lower() for w in warnings)

            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.source == "template"


# ── 8. Malformed JSON ───────────────────────────────────────────────────────

def test_malformed_json_falls_back(sample_prop, skeleton):
    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key", "LLM_FALLBACK_API_KEY": ""}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response("Not JSON at all! {invalid")):
            scenes, warnings, provider = generate_scene_copy(sample_prop, skeleton)
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

    with patch.dict("os.environ", {"LLM_API_KEY": "test_key", "LLM_FALLBACK_API_KEY": ""}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(valid_content)):
            plan = build_scene_plan(sample_prop, images)
            assert plan.source == "llm (primary)"


def test_no_llm_forces_template_even_when_key_set(sample_prop):
    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key", "LLM_FALLBACK_API_KEY": ""}):
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

    with patch.dict("os.environ", {"LLM_API_KEY": "test_key", "LLM_FALLBACK_API_KEY": ""}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(llm_content)):
            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.scenes[0].caption == "Custom Green Valley\nApartment"
            assert plan.scenes[0].voiceover == "Welcome to our custom property title."
            assert plan.scenes[1].caption == "Covered Parking"
            assert plan.source == "mixed (4 of 5 scenes from llm (primary))"


# ── 11. GEMINI_API_KEY Support ──────────────────────────────────────────────

def test_gemini_api_key_automatically_used(sample_prop, skeleton):
    valid_content = json.dumps({
        "scenes": [
            {"scene_index": 1, "caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
            {"scene_index": 2, "caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
            {"scene_index": 3, "caption": "Modular Kitchen", "voiceover": "Features a modern modular kitchen."},
            {"scene_index": 4, "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
            {"scene_index": 5, "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today, details on screen."},
        ]
    })
    recorded_req = []

    def mock_open(req, timeout=20):
        recorded_req.append(req)
        return _mock_http_response(valid_content)

    with patch.dict("os.environ", {"GEMINI_API_KEY": "AIzaSyFakeGeminiKey", "LLM_API_KEY": "", "LLM_FALLBACK_API_KEY": "", "LLM_MODEL": ""}):
        with patch("urllib.request.urlopen", side_effect=mock_open):
            scenes, warnings, provider = generate_scene_copy(sample_prop, skeleton)
            assert scenes is not None
            assert len(recorded_req) == 1
            assert "generativelanguage.googleapis.com" in recorded_req[0].full_url
            body = json.loads(recorded_req[0].data.decode("utf-8"))
            assert body["model"] == "gemini-2.5-flash"
            assert provider == "primary"


# ── 12. Primary 503 then Success on Retry ───────────────────────────────────

def test_primary_503_then_success_on_retry(sample_prop, skeleton):
    """Primary provider returns 503 on first call, retries once with backoff, and succeeds."""
    valid_content = json.dumps({
        "scenes": [
            {"scene_index": 1, "caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
            {"scene_index": 2, "caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
            {"scene_index": 3, "caption": "Modular Kitchen", "voiceover": "Features a modern modular kitchen."},
            {"scene_index": 4, "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
            {"scene_index": 5, "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today, contact details on screen."},
        ]
    })

    req_count = 0

    def mock_urlopen(req, timeout=20):
        nonlocal req_count
        req_count += 1
        if req_count == 1:
            raise _mock_http_error(503, "Service Unavailable")
        return _mock_http_response(valid_content)

    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with patch.dict("os.environ", {"LLM_API_KEY": "primary_key", "LLM_FALLBACK_API_KEY": ""}):
        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            scenes, warnings, provider = generate_scene_copy(sample_prop, skeleton)
            assert req_count == 2
            assert scenes is not None
            assert provider == "primary"

    req_count = 0
    with patch.dict("os.environ", {"LLM_API_KEY": "primary_key", "LLM_FALLBACK_API_KEY": ""}):
        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert req_count == 2
            assert plan.source == "llm (primary)"
            assert plan.scenes[0].caption == "Green Valley 2BHK\nApartment"


# ── 13. Primary Fails Twice then Fallback Succeeds ──────────────────────────

def test_primary_fails_twice_then_fallback_succeeds(sample_prop, skeleton):
    """Primary fails twice (500), then fallback provider (OpenRouter) is called and succeeds."""
    valid_content = json.dumps({
        "scenes": [
            {"scene_index": 1, "caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
            {"scene_index": 2, "caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
            {"scene_index": 3, "caption": "Modular Kitchen", "voiceover": "Features a modern modular kitchen."},
            {"scene_index": 4, "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
            {"scene_index": 5, "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today, contact details on screen."},
        ]
    })

    recorded_calls = []

    def mock_urlopen(req, timeout=20):
        url = req.full_url
        recorded_calls.append(url)
        if "api.openai.com" in url:
            # Primary fails on attempt 1 and attempt 2 (retry)
            raise _mock_http_error(500, "Internal Server Error")
        elif "openrouter.ai" in url:
            # Fallback succeeds
            return _mock_http_response(valid_content)
        raise RuntimeError("Unexpected URL: " + url)

    images = [Path(f"img_{i}.jpg") for i in range(5)]
    env_vars = {
        "LLM_API_KEY": "primary_secret",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_FALLBACK_API_KEY": "fallback_secret",
        "LLM_FALLBACK_BASE_URL": "https://openrouter.ai/api/v1",
        "LLM_FALLBACK_MODEL": "openrouter/auto",
    }

    with patch.dict("os.environ", env_vars):
        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            scenes, warnings, provider = generate_scene_copy(sample_prop, skeleton)
            assert len(recorded_calls) == 3
            assert recorded_calls[0] == "https://api.openai.com/v1/chat/completions"
            assert recorded_calls[1] == "https://api.openai.com/v1/chat/completions"
            assert recorded_calls[2] == "https://openrouter.ai/api/v1/chat/completions"
            assert scenes is not None
            assert provider == "fallback: openrouter"

    recorded_calls.clear()
    with patch.dict("os.environ", env_vars):
        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert len(recorded_calls) == 3
            assert plan.source == "llm (fallback: openrouter)"
            # Verify script.json dictionary output contains exact source
            plan_dict = plan.to_dict()
            assert plan_dict["source"] == "llm (fallback: openrouter)"


# ── 14. Both Fail -> Template ───────────────────────────────────────────────

def test_both_providers_fail_falls_back_to_template(sample_prop, skeleton):
    """When both primary and fallback fail (each after 1 retry), system gracefully falls back to template."""
    recorded_calls = []

    def mock_urlopen(req, timeout=20):
        recorded_calls.append(req.full_url)
        if "api.openai.com" in req.full_url:
            raise _mock_http_error(502, "Bad Gateway")
        raise _mock_http_error(504, "Gateway Timeout")

    images = [Path(f"img_{i}.jpg") for i in range(5)]
    env_vars = {
        "LLM_API_KEY": "primary_secret",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_FALLBACK_API_KEY": "fallback_secret",
        "LLM_FALLBACK_BASE_URL": "https://api.groq.com/openai/v1",
    }

    with patch.dict("os.environ", env_vars):
        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            scenes, warnings, provider = generate_scene_copy(sample_prop, skeleton)
            # Primary tried twice, fallback tried twice = 4 calls
            assert len(recorded_calls) == 4
            assert scenes is None
            assert provider is None
            assert any("All LLM providers failed" in w for w in warnings)

            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.source == "template"
            assert len(plan.scenes) == 5


# ── 15. 401 Does Not Retry ──────────────────────────────────────────────────

def test_401_does_not_retry_and_warns(sample_prop, skeleton):
    """HTTP 401 (bad key) does not retry; warns 'check LLM_API_KEY' and moves to fallback immediately."""
    valid_content = json.dumps({
        "scenes": [
            {"scene_index": 1, "caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
            {"scene_index": 2, "caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
            {"scene_index": 3, "caption": "Modular Kitchen", "voiceover": "Features a modern modular kitchen."},
            {"scene_index": 4, "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
            {"scene_index": 5, "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today, contact details on screen."},
        ]
    })

    recorded_calls = []

    def mock_urlopen(req, timeout=20):
        recorded_calls.append(req.full_url)
        if "api.openai.com" in req.full_url:
            raise _mock_http_error(401, "Unauthorized")
        return _mock_http_response(valid_content)

    images = [Path(f"img_{i}.jpg") for i in range(5)]
    env_vars = {
        "LLM_API_KEY": "bad_primary_key",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_FALLBACK_API_KEY": "valid_fallback_key",
        "LLM_FALLBACK_BASE_URL": "https://openrouter.ai/api/v1",
    }

    with patch.dict("os.environ", env_vars):
        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            scenes, warnings, provider = generate_scene_copy(sample_prop, skeleton)
            # Primary was called ONCE (no retry on 401), then fallback was called ONCE
            assert len(recorded_calls) == 2
            assert recorded_calls[0] == "https://api.openai.com/v1/chat/completions"
            assert recorded_calls[1] == "https://openrouter.ai/api/v1/chat/completions"
            assert any("check LLM_API_KEY" in w for w in warnings)
            assert scenes is not None
            assert provider == "fallback: openrouter"

            plan = build_scene_plan(sample_prop, images, use_llm=True)
            assert plan.source == "llm (fallback: openrouter)"


# ── 16. JSON Wrapped in Fences or Extra Text Parses ─────────────────────────

@pytest.mark.parametrize(
    "raw_text",
    [
        # Markdown fences with json tag
        "```json\n" + json.dumps({
            "scenes": [
                {"scene_index": 1, "caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
                {"scene_index": 2, "caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
                {"scene_index": 3, "caption": "Modular Kitchen", "voiceover": "Features a modern modular kitchen."},
                {"scene_index": 4, "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
                {"scene_index": 5, "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today, contact details on screen."},
            ]
        }) + "\n```",
        # Thinking tags (<think>...</think>) before JSON
        "<think>\nAnalyzing facts for Green Valley 2BHK in Baner...\n</think>\n" + json.dumps({
            "scenes": [
                {"scene_index": 1, "caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
                {"scene_index": 2, "caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
                {"scene_index": 3, "caption": "Modular Kitchen", "voiceover": "Features a modern modular kitchen."},
                {"scene_index": 4, "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
                {"scene_index": 5, "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today, contact details on screen."},
            ]
        }),
        # Thinking process and conversational text surrounding JSON block
        "Thought: Checking amenities and constraints.\nHere is the requested output:\n" + json.dumps({
            "scenes": [
                {"scene_index": 1, "caption": "Green Valley 2BHK\nApartment", "voiceover": "Welcome to Green Valley 2BHK in Baner."},
                {"scene_index": 2, "caption": "Covered Parking", "voiceover": "Includes covered parking for your vehicle."},
                {"scene_index": 3, "caption": "Modular Kitchen", "voiceover": "Features a modern modular kitchen."},
                {"scene_index": 4, "caption": "₹45,00,000\nBaner, Pune", "voiceover": "Priced at 45 lakh rupees in Baner."},
                {"scene_index": 5, "caption": "Book a site visit today\n+91-00000-00000", "voiceover": "Book a site visit today, contact details on screen."},
            ]
        }) + "\nI hope this meets the requirements!",
    ],
)
def test_json_wrapped_in_fences_and_thinking_text_parses(sample_prop, skeleton, raw_text):
    """Strip markdown code fences and leading thinking text before parsing JSON."""
    with patch.dict("os.environ", {"LLM_API_KEY": "test_key", "LLM_FALLBACK_API_KEY": ""}):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(raw_text)):
            scenes, warnings, provider = generate_scene_copy(sample_prop, skeleton)
            assert scenes is not None
            assert len(scenes) == 5
            assert scenes[0]["caption"] == "Green Valley 2BHK\nApartment"
            assert scenes[2]["caption"] == "Modular Kitchen"


# ── 17. Keys Never Appear in Logs or Warnings ───────────────────────────────

def test_keys_never_appear_in_logs_or_warnings(sample_prop, skeleton, caplog):
    """Ensure primary and fallback API keys never appear in logs, warnings, or script structures."""
    primary_secret = "sk-PRIMARY_SUPER_SECRET_KEY_12345"
    fallback_secret = "sk-FALLBACK_SUPER_SECRET_KEY_67890"

    env_vars = {
        "LLM_API_KEY": primary_secret,
        "LLM_FALLBACK_API_KEY": fallback_secret,
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_FALLBACK_BASE_URL": "https://openrouter.ai/api/v1",
    }

    # Simulate an error that might contain the auth header or exception message
    def mock_urlopen_leaky_error(req, timeout=20):
        auth_hdr = req.headers.get("Authorization", "")
        # Emulate an error whose string contains auth header / keys
        raise urllib.error.HTTPError(
            url=req.full_url,
            code=401,
            msg=f"Unauthorized: request with {auth_hdr} failed",
            hdrs={},
            fp=None,
        )

    images = [Path(f"img_{i}.jpg") for i in range(5)]
    with caplog.at_level(logging.DEBUG):
        with patch.dict("os.environ", env_vars):
            with patch("urllib.request.urlopen", side_effect=mock_urlopen_leaky_error):
                plan = build_scene_plan(sample_prop, images, use_llm=True)
                assert plan.source == "template"

                # Check all plan warnings
                for warn in plan.warnings:
                    assert primary_secret not in warn
                    assert fallback_secret not in warn

                # Check plan dictionary
                plan_str = json.dumps(plan.to_dict())
                assert primary_secret not in plan_str
                assert fallback_secret not in plan_str

                # Check all logged records
                for record in caplog.records:
                    log_text = record.getMessage()
                    assert primary_secret not in log_text
                    assert fallback_secret not in log_text
