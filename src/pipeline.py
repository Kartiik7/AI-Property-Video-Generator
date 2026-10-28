"""
pipeline.py — Unified orchestration pipeline for video generation.

Used by both CLI (main.py) and Streamlit UI (app.py) to guarantee identical
validation, script generation, audio mixing, and video rendering behavior.
"""

import os
import time
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Any

from src.models import PropertyData, ScenePlan
from src.validators import validate_property, validate_images
from src.script_generator import build_scene_plan
from src.audio_builder import build_audio_tracks
from src.video_builder import build_video, slugify
from src.logger import setup_logger

logger = setup_logger()


@dataclass
class PipelineOptions:
    """Options controlling pipeline execution."""
    use_llm: bool | None = None
    no_video: bool = False
    size: str = "720p"          # "720p" or "1080p"
    fps: int = 24
    out_path: Path | str | None = None
    script_out: Path | str | None = None
    no_audio: bool = False
    no_music: bool = False
    no_narration: bool = False
    music_path: Path | str | None = None
    voice: str = "en-IN-NeerjaNeural"
    tts_rate: int = 0


_py_property = property

@dataclass
class PipelineResult:
    """Standardized output returned by run_pipeline."""
    video_path: Path | None = None
    plan: ScenePlan | None = None
    property: PropertyData | None = None
    images: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    file_size: int = 0

    @_py_property
    def ok(self) -> bool:
        """True if pipeline produced no blocking errors and succeeded."""
        return len(self.errors) == 0 and (self.video_path is not None or self.plan is not None)


def run_pipeline(
    property_input: dict | Path | str | PropertyData,
    images_input: Path | str | list[Path],
    options: PipelineOptions | dict | None = None,
    progress_cb: Callable[[str, float, str], None] | None = None,
) -> PipelineResult:
    """Execute the full video generator pipeline.

    Args:
        property_input: Dict of property fields, path to property.json, or PropertyData.
        images_input: Directory path containing images or explicit list of image Paths.
        options: Configuration options (PipelineOptions or dict).
        progress_cb: Optional callback func(stage, percent_float, message).

    Returns:
        PipelineResult with video_path, plan, warnings, errors, timings, and file_size.
    """
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    warnings: list[str] = []
    errors: list[str] = []

    # Normalize options
    if options is None:
        opts = PipelineOptions()
    elif isinstance(options, dict):
        opts = PipelineOptions(**options)
    else:
        opts = options

    def _notify(stage: str, frac: float, msg: str):
        if progress_cb:
            try:
                progress_cb(stage, frac, msg)
            except Exception as e:
                logger.debug("Progress callback error: %s", e)

    # ── Stage 1: Input Loading & Validation ─────────────────────────────────
    _notify("validation", 0.05, "Validating property details and images...")
    t0 = time.perf_counter()

    prop: PropertyData | None = None

    if isinstance(property_input, PropertyData):
        prop = property_input
    elif isinstance(property_input, dict):
        prop, prop_warns, prop_errs = validate_property(property_input)
        warnings.extend(prop_warns)
        errors.extend(prop_errs)
    else:
        # File path
        data_path = Path(property_input)
        if not data_path.exists():
            errors.append(f"Property file '{data_path}' does not exist.")
        else:
            try:
                content = json.loads(data_path.read_text(encoding="utf-8"))
                prop, prop_warns, prop_errs = validate_property(content)
                warnings.extend(prop_warns)
                errors.extend(prop_errs)
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid JSON in '{data_path}': {exc}")
            except Exception as exc:
                errors.append(f"Could not read property file '{data_path}': {exc}")

    # Validate images
    accepted_images: list[Path] = []
    if isinstance(images_input, list):
        # Explicit list of Path objects
        if not images_input:
            errors.append("No images provided; at least 1 image is required.")
        else:
            accepted_images = [Path(p) for p in images_input if Path(p).exists()]
            if len(accepted_images) < 4:
                warnings.append(
                    f"Only {len(accepted_images)} image(s) provided. Recommended: 4–8. "
                    "Images will be reused across scenes."
                )
            if len(accepted_images) > 8:
                warnings.append(f"{len(accepted_images)} images provided; using the first 8.")
                accepted_images = accepted_images[:8]
    else:
        # Directory path
        img_dir = Path(images_input)
        accepted_images, img_warns, img_errs = validate_images(img_dir)
        warnings.extend(img_warns)
        errors.extend(img_errs)

    timings["validation"] = round(time.perf_counter() - t0, 3)

    if errors or prop is None:
        logger.warning("Pipeline validation failed with %d error(s).", len(errors))
        return PipelineResult(
            warnings=warnings,
            errors=errors,
            timings=timings,
        )

    # ── Stage 2: Storyboard & Script Generation ─────────────────────────────
    _notify("script", 0.20, "Building scene plan and narration script...")
    t0 = time.perf_counter()

    try:
        plan = build_scene_plan(prop, accepted_images, use_llm=opts.use_llm)
        warnings.extend(plan.warnings)
    except Exception as exc:
        logger.exception("Error in build_scene_plan: %s", exc)
        errors.append(f"Script generation failed: {exc}")
        return PipelineResult(
            property=prop,
            images=accepted_images,
            warnings=warnings,
            errors=errors,
            timings=timings,
        )

    timings["script"] = round(time.perf_counter() - t0, 3)

    # Save script JSON if requested
    if opts.script_out:
        script_path = Path(opts.script_out)
        try:
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(
                json.dumps(plan.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            warnings.append(f"Could not save scene plan to '{script_path}': {exc}")

    # If --no-video requested, exit early with plan
    if opts.no_video:
        timings["total"] = round(time.perf_counter() - t_start, 3)
        return PipelineResult(
            plan=plan,
            property=prop,
            images=accepted_images,
            warnings=warnings,
            errors=errors,
            timings=timings,
        )

    # ── Stage 3: Audio Synthesis & Mixing ───────────────────────────────────
    _notify("audio", 0.40, "Synthesizing voiceover narration and mixing audio...")
    t0 = time.perf_counter()

    res_tuple = (1920, 1080) if opts.size == "1080p" else (1280, 720)

    # Destination video path
    if opts.out_path:
        out_video = Path(opts.out_path).resolve()
    else:
        slug = slugify(prop.name)
        out_video = Path("output") / f"{slug}.mp4"

    out_video.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as audio_temp_dir:
        audio_clip = None
        try:
            audio_clip, _ = build_audio_tracks(
                plan=plan,
                temp_dir=Path(audio_temp_dir),
                voice=opts.voice,
                base_rate=opts.tts_rate,
                music_path=opts.music_path,
                no_audio=opts.no_audio,
                no_music=opts.no_music,
                no_narration=opts.no_narration,
                warnings=warnings,
            )
            timings["audio"] = round(time.perf_counter() - t0, 3)

            # ── Stage 4: Video Rendering ────────────────────────────────────
            _notify("render", 0.60, f"Rendering video scenes ({opts.size} @ {opts.fps} fps)...")
            t0 = time.perf_counter()

            total_scenes = len(plan.scenes)

            def render_progress(cur_scene: int, total: int, msg: str):
                frac = 0.60 + (cur_scene / total) * 0.35
                _notify("render", frac, f"Rendering scene {cur_scene}/{total}...")

            exported_path = build_video(
                plan=plan,
                prop=prop,
                out_path=out_video,
                size=res_tuple,
                fps=opts.fps,
                progress_cb=render_progress,
                audio=audio_clip,
            )
            timings["render"] = round(time.perf_counter() - t0, 3)

        except Exception as exc:
            logger.exception("Video rendering error: %s", exc)
            errors.append(f"Video rendering failed: {exc}")
            return PipelineResult(
                plan=plan,
                property=prop,
                images=accepted_images,
                warnings=warnings,
                errors=errors,
                timings=timings,
            )
        finally:
            if audio_clip is not None:
                try:
                    audio_clip.close()
                except Exception:
                    pass

    file_size = exported_path.stat().st_size if exported_path.exists() else 0
    timings["total"] = round(time.perf_counter() - t_start, 3)
    _notify("complete", 1.0, "Video generation complete!")

    return PipelineResult(
        video_path=exported_path,
        plan=plan,
        property=prop,
        images=accepted_images,
        warnings=warnings,
        errors=errors,
        timings=timings,
        file_size=file_size,
    )
