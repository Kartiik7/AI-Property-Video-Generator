"""
test_pipeline.py — Tests for the unified pipeline and Streamlit app.

Verifies:
- Pipeline and CLI behavioral parity (same validation, plan structure, and results).
- run_pipeline handles dicts, file paths, and PropertyData.
- Error handling without uncaught exceptions.
- app.py smoke test and Streamlit AppTest simulation.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from src.pipeline import run_pipeline, PipelineOptions, PipelineResult
from src.models import PropertyData, ScenePlan
import main


SAMPLE_PROP_JSON = Path("input/samples/house_1/property.json")
SAMPLE_IMG_DIR = Path("input/samples/house_1/images")


def test_pipeline_parity_with_cli(tmp_path):
    """Test that run_pipeline produces the exact same plan as main CLI in --no-video mode."""
    script_cli = tmp_path / "script_cli.json"
    script_pipe = tmp_path / "script_pipe.json"

    # 1. Run via CLI main()
    cli_args = [
        "main.py",
        "--data", str(SAMPLE_PROP_JSON),
        "--images", str(SAMPLE_IMG_DIR),
        "--no-video",
        "--no-llm",
        "--script-out", str(script_cli),
    ]
    with patch("sys.argv", cli_args):
        ret = main.main()
        assert ret == 0

    # 2. Run via run_pipeline directly
    opts = PipelineOptions(
        use_llm=False,
        no_video=True,
        script_out=script_pipe,
    )
    result = run_pipeline(
        property_input=SAMPLE_PROP_JSON,
        images_input=SAMPLE_IMG_DIR,
        options=opts,
    )

    assert result.ok is True
    assert result.plan is not None
    assert script_cli.exists()
    assert script_pipe.exists()

    data_cli = json.loads(script_cli.read_text(encoding="utf-8"))
    data_pipe = json.loads(script_pipe.read_text(encoding="utf-8"))

    # Both plans must have identical scenes and texts
    assert len(data_cli["scenes"]) == len(data_pipe["scenes"])
    for sc_cli, sc_pipe in zip(data_cli["scenes"], data_pipe["scenes"]):
        assert sc_cli["scene_type"] == sc_pipe["scene_type"]
        assert sc_cli["caption"] == sc_pipe["caption"]
        assert sc_cli["voiceover"] == sc_pipe["voiceover"]
        assert sc_cli["duration"] == sc_pipe["duration"]


def test_pipeline_with_dict_and_explicit_images_list():
    """Test run_pipeline with a raw dict and explicit list of image Paths."""
    prop_dict = {
        "name": "Emerald Heights",
        "location": "Bandra, Mumbai",
        "price": "₹2,50,00,000",
        "property_type": "Apartment",
        "features": ["Sea facing balcony", "Modern kitchen", "Modular wardrobe"],
        "cta_text": "Schedule a private tour",
    }
    imgs = sorted(list(SAMPLE_IMG_DIR.glob("*.jpg")))
    assert len(imgs) >= 4

    result = run_pipeline(
        property_input=prop_dict,
        images_input=imgs,
        options=PipelineOptions(no_video=True, use_llm=False),
    )

    assert result.ok is True
    assert result.property is not None
    assert result.property.name == "Emerald Heights"
    assert result.plan is not None
    assert len(result.plan.scenes) >= 5
    assert len(result.errors) == 0


def test_pipeline_error_handling_graceful():
    """Test run_pipeline when given missing file or invalid property data."""
    # Non-existent file
    res = run_pipeline(
        property_input=Path("non_existent_prop.json"),
        images_input=SAMPLE_IMG_DIR,
    )
    assert res.ok is False
    assert len(res.errors) > 0
    assert "does not exist" in res.errors[0]

    # Empty property dict
    res2 = run_pipeline(
        property_input={},
        images_input=SAMPLE_IMG_DIR,
    )
    assert res2.ok is False
    assert any("Missing required field" in e for e in res2.errors)


def test_streamlit_app_smoke():
    """Smoke test: verify app.py imports cleanly and runs under AppTest."""
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path.resolve()), default_timeout=15)
    at.run()
    assert not at.exception

    # Check title and buttons exist
    assert len(at.title) > 0
    assert "AI Property Video Generator" in at.title[0].value

    # Simulate clicking "Load Sample" button
    sample_btn = [b for b in at.button if "Load Sample" in b.label]
    assert len(sample_btn) > 0
    sample_btn[0].click().run()
    assert not at.exception
