"""
test_tts_gemini.py — Unit tests for the Gemini TTS provider in audio_builder.py.

Verifies:
  - Happy path: valid JSON response is decoded and WAV is written
  - Cache reuse: second call with same text hits cache (no HTTP call)
  - Retry on 429 / 5xx: transient error retries once with backoff
  - No-retry on 401 / 403: auth failure fails immediately without retry
  - Fallback to edge on Gemini failure: regenerates all scenes with edge-tts and records warning
  - Provider resolution in auto mode: gemini if key present, else edge
  - Key-redaction safety: raw API keys are never leaked in error messages or logs
"""

import os
import json
import base64
import wave
import io
import urllib.error
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from src.models import Scene, ScenePlan
from src.audio_builder import (
    get_gemini_api_key,
    resolve_tts_provider,
    synthesize_gemini_voiceover,
    build_audio_tracks,
    _safe_error_msg,
    CACHE_DIR,
)


def _make_dummy_pcm(duration_samples: int = 4800, sample_rate: int = 24000) -> bytes:
    """Generate dummy mono 16-bit PCM bytes (silence or small values)."""
    return b"\x00\x00" * duration_samples


def _make_gemini_response_json(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Build a mock Gemini generateContent JSON response with inlineData audio."""
    b64_audio = base64.b64encode(pcm_bytes).decode("utf-8")
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": f"audio/pcm;rate={sample_rate}",
                                "data": b64_audio,
                            }
                        }
                    ]
                }
            }
        ]
    }
    return json.dumps(payload).encode("utf-8")


# ── 1. Provider Resolution & Key Discovery ───────────────────────────────────

def test_resolve_tts_provider_explicit(monkeypatch):
    assert resolve_tts_provider("gemini") == "gemini"
    assert resolve_tts_provider("edge") == "edge"

    monkeypatch.setenv("TTS_PROVIDER", "gemini")
    assert resolve_tts_provider(None) == "gemini"

    monkeypatch.setenv("TTS_PROVIDER", "edge")
    assert resolve_tts_provider(None) == "edge"


def test_resolve_tts_provider_auto(monkeypatch):
    # Case A: No keys set -> edge
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("TTS_PROVIDER", "auto")
    assert resolve_tts_provider() == "edge"

    # Case B: GEMINI_API_KEY set -> gemini
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-key")
    assert resolve_tts_provider() == "gemini"

    # Case C: Only LLM_API_KEY set, but LLM_BASE_URL contains generativelanguage -> gemini
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "dummy-llm-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    assert resolve_tts_provider() == "gemini"

    # Case D: LLM_API_KEY set but for OpenAI -> edge
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    assert resolve_tts_provider() == "edge"


# ── 2. Happy Path Synthesis ──────────────────────────────────────────────────

def test_gemini_tts_happy_path(tmp_path, monkeypatch):
    fake_pcm = _make_dummy_pcm(4800, sample_rate=24000)
    fake_resp = _make_gemini_response_json(fake_pcm, sample_rate=24000)

    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_resp
    mock_resp.__enter__.return_value = mock_resp

    out_file = tmp_path / "test_gemini.wav"

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        result_path = synthesize_gemini_voiceover(
            text="Welcome to this beautiful 3 BHK apartment.",
            out_path=out_file,
            voice="Kore",
            model="gemini-3.8-flash-lite-tts",
            api_key="test-secret-key-12345",
            use_cache=False,
        )

        assert result_path == out_file
        assert out_file.exists()
        assert out_file.stat().st_size > 0

        # Validate it's a valid WAV file
        with wave.open(str(out_file), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 24000
            assert wf.getnframes() == 4800

        # Verify request structure
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_header("X-goog-api-key") == "test-secret-key-12345"
        req_body = json.loads(req.data.decode("utf-8"))
        assert "speech_metadata" in req_body["contents"][0]["parts"][0]
        assert req_body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"


# ── 3. Disk Cache Reuse ──────────────────────────────────────────────────────

def test_gemini_tts_cache_reuse(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr("src.audio_builder.CACHE_DIR", cache_dir)

    fake_pcm = _make_dummy_pcm(2400, sample_rate=24000)
    fake_resp = _make_gemini_response_json(fake_pcm, sample_rate=24000)

    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_resp
    mock_resp.__enter__.return_value = mock_resp

    out_file1 = tmp_path / "scene1.wav"
    out_file2 = tmp_path / "scene1_cached.wav"

    text = "Unique cached narration line for testing cache hit"
    key = "test-cache-key-xyz"

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        # First call: makes network request and stores in cache
        res1 = synthesize_gemini_voiceover(
            text=text,
            out_path=out_file1,
            voice="Kore",
            api_key=key,
            use_cache=True,
        )
        assert mock_urlopen.call_count == 1
        assert res1.exists()

        # Second call: hits disk cache, 0 new network calls
        res2 = synthesize_gemini_voiceover(
            text=text,
            out_path=out_file2,
            voice="Kore",
            api_key=key,
            use_cache=True,
        )
        assert mock_urlopen.call_count == 1  # count did not increase
        assert res2.exists()
        assert res2.stat().st_size == res1.stat().st_size


# ── 4. Transient Error Retry (429 / 5xx) ─────────────────────────────────────

def test_gemini_tts_retry_on_429(tmp_path):
    fake_pcm = _make_dummy_pcm(2400)
    fake_resp = _make_gemini_response_json(fake_pcm)

    mock_resp_success = MagicMock()
    mock_resp_success.read.return_value = fake_resp
    mock_resp_success.__enter__.return_value = mock_resp_success

    http_429 = urllib.error.HTTPError(
        url="https://generativelanguage.googleapis.com",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=io.BytesIO(b'{"error": "Resource has been exhausted"}'),
    )

    out_file = tmp_path / "retry_test.wav"

    with patch("time.sleep", return_value=None), \
         patch("urllib.request.urlopen", side_effect=[http_429, mock_resp_success]) as mock_urlopen:
        res = synthesize_gemini_voiceover(
            text="Retry on 429 line",
            out_path=out_file,
            api_key="key-retry-429",
            use_cache=False,
        )
        assert res.exists()
        assert mock_urlopen.call_count == 2


# ── 5. No Retry on 401 / 403 ─────────────────────────────────────────────────

def test_gemini_tts_no_retry_on_401(tmp_path):
    http_401 = urllib.error.HTTPError(
        url="https://generativelanguage.googleapis.com",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=io.BytesIO(b'{"error": "API key not valid"}'),
    )

    out_file = tmp_path / "auth_fail.wav"

    with patch("time.sleep", return_value=None) as mock_sleep, \
         patch("urllib.request.urlopen", side_effect=http_401) as mock_urlopen:
        with pytest.raises(RuntimeError, match="HTTP 401"):
            synthesize_gemini_voiceover(
                text="Auth failure line",
                out_path=out_file,
                api_key="invalid-key-401",
                use_cache=False,
            )
        # Must fail immediately without sleeping or retrying
        assert mock_urlopen.call_count == 1
        assert mock_sleep.call_count == 0


# ── 6. Fallback to Edge-TTS on Gemini Failure in Pipeline ────────────────────

def test_build_audio_tracks_gemini_fallback_to_edge(tmp_path, monkeypatch):
    plan = ScenePlan(
        scenes=[
            Scene(
                scene_type="title",
                image_path="img1.jpg",
                caption="Title",
                voiceover="Welcome to the property.",
                duration=4.0,
            ),
            Scene(
                scene_type="feature",
                image_path="img2.jpg",
                caption="Feature 1",
                voiceover="Spacious living room.",
                duration=4.0,
            ),
        ],
        source="llm",
    )

    # Force provider to gemini
    monkeypatch.setenv("TTS_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")

    gemini_calls = 0
    edge_calls = 0

    def mock_gemini_synth(*args, **kwargs):
        nonlocal gemini_calls
        gemini_calls += 1
        raise RuntimeError("Simulated Gemini quota exceeded (HTTP 429)")

    def mock_edge_synth(text, out_path, voice=None, rate=0, use_cache=True):
        nonlocal edge_calls
        edge_calls += 1
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Create small valid wave
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(_make_dummy_pcm(4800))
        return out

    with patch("src.audio_builder.synthesize_gemini_voiceover", side_effect=mock_gemini_synth), \
         patch("src.audio_builder.synthesize_edge_voiceover", side_effect=mock_edge_synth):
        warnings = []
        clip, durs = build_audio_tracks(
            plan=plan,
            temp_dir=tmp_path / "audio_test",
            tts_provider="gemini",
            warnings=warnings,
            no_music=True,
        )

        # Gemini failed on scene 1, aborted remaining scenes, and edge-tts synthesized all 2 scenes
        assert gemini_calls == 1
        assert edge_calls == 2
        # Warning recorded
        assert any("Gemini TTS failed" in w and "regenerated all narration with edge-tts" in w for w in warnings)
        # Metadata updated to edge
        assert plan.metadata.get("tts_provider") == "edge"
        assert plan.metadata.get("tts_model") == "edge-tts"


# ── 7. Key Redaction Safety ──────────────────────────────────────────────────

def test_key_redaction_safety(monkeypatch):
    secret = "AIzaSySecretApiKey998877"
    monkeypatch.setenv("GEMINI_API_KEY", secret)

    # Exception containing raw key
    exc = Exception(f"Failed to connect using key {secret} to server")
    safe_msg = _safe_error_msg(exc)

    assert secret not in safe_msg
    assert "[REDACTED]" in safe_msg
