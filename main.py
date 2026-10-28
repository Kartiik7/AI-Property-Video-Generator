"""
main.py — CLI entry point for the AI Property Video Generator.

Phase 1: input loading and validation only.
Video generation will be added in Phase 3.

Exit codes:
  0 — validation passed
  1 — validation errors (bad data / missing images)
  2 — bad CLI usage (handled automatically by argparse)
"""

import os
import sys
import argparse
from pathlib import Path

# ── Windows console fix ────────────────────────────────────────────────────
# Must happen before any print so rupee signs (₹) and symbols don't crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.logger import setup_logger

logger = setup_logger()

BASE_DIR = Path(__file__).parent


# ── CLI parser ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="property-video",
        description=(
            "AI Property Video Generator — "
            "generates scripts, captions, and storyboards from property data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --data input/property.json\n"
            "  python main.py --data input/property.json --use-llm\n"
            "  python main.py --data input/property.json --script-out output/my_script.json\n"
            "  python main.py --interactive\n"
        ),
    )

    # --data and --interactive are mutually exclusive; one is required
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data",
        metavar="PATH",
        help="Path to a JSON file containing property details.",
    )
    source.add_argument(
        "--interactive",
        action="store_true",
        help="Enter property details interactively via prompts.",
    )

    p.add_argument(
        "--images",
        metavar="PATH",
        default=str(BASE_DIR / "input" / "images"),
        help="Directory containing property images (default: input/images).",
    )
    p.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable optional LLM mode for generating captions and voiceover (default if LLM_API_KEY is set).",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Force template generation and disable LLM mode even if LLM_API_KEY is set.",
    )
    p.add_argument(
        "--script-out",
        metavar="PATH",
        default=str(BASE_DIR / "output" / "script.json"),
        help="Path to write generated scene plan JSON (default: output/script.json).",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Generate and print scene plan only, without rendering the video.",
    )
    p.add_argument(
        "--size",
        choices=["720p", "1080p"],
        default="720p",
        help="Video resolution: 720p (1280x720) or 1080p (1920x1080) (default: 720p).",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Video frame rate in frames per second (default: 24).",
    )
    p.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Output video path (default: output/<property-name-slug>.mp4).",
    )
    p.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable all audio (both narration and background music).",
    )
    p.add_argument(
        "--no-music",
        action="store_true",
        help="Disable background music (narration will still play).",
    )
    p.add_argument(
        "--music",
        metavar="PATH",
        default=None,
        help="Path to custom background music audio file.",
    )
    p.add_argument(
        "--voice",
        metavar="NAME",
        default="en-IN-NeerjaNeural",
        help="edge-tts voice name (default: en-IN-NeerjaNeural).",
    )
    p.add_argument(
        "--tts-rate",
        type=int,
        default=0,
        help="TTS speed percentage adjustment (e.g. 10 for +10%%, default: 0).",
    )
    p.add_argument(
        "--open",
        action="store_true",
        help="Open the finished video in the default media player upon completion.",
    )
    return p


# ── Summary printer ────────────────────────────────────────────────────────

def print_summary(result) -> None:
    """Print a human-readable summary of the LoadResult."""
    SEP = "=" * 60

    print(f"\n{SEP}")
    print("  PROPERTY VIDEO GENERATOR — INPUT SUMMARY")
    print(SEP)

    # Property fields
    if result.property:
        p = result.property
        print("\nProperty Details:")
        print(f"  Name          : {p.name}")
        print(f"  Location      : {p.location}")
        print(f"  Price         : {p.price}")
        print(f"  Type          : {p.property_type}")
        print(f"  Features      : {', '.join(p.features)}")
        print(f"  CTA           : {p.cta_text}")
        if p.contact:
            print(f"  Contact       : {p.contact}")
        if p.tagline:
            print(f"  Tagline       : {p.tagline}")
    else:
        print("\nProperty Details: [could not be loaded — see ERRORS below]")

    # Accepted images
    print(f"\nImages ({len(result.images)} accepted):")
    if result.images:
        for img in result.images:
            print(f"  - {img.name}")
    else:
        print("  [none]")

    # Warnings
    if result.warnings:
        print(f"\nWARNINGS ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"  ⚠  {w}")

    # Errors
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for e in result.errors:
            print(f"  ✗  {e}")


def print_scene_plan(plan) -> None:
    """Print the generated storyboard and script in a clean table format."""
    SEP = "=" * 70
    LINE = "-" * 70

    print(f"\n{SEP}")
    print(f"  SCENE PLAN (Source: {plan.source} | Total duration: {plan.total_duration}s)")
    print(SEP)

    header = f"{'#':<3} {'Type':<16} {'Image':<18} {'Duration':<10} {'Caption / Voiceover'}"
    print(header)
    print(LINE)

    for i, s in enumerate(plan.scenes, 1):
        cap_lines = s.caption.split("\n")
        first_cap = cap_lines[0] if cap_lines else ""
        dur_str = f"{s.duration:.1f}s"
        img_name = s.image_path.name

        print(f"{i:<3} {s.scene_type:<16} {img_name:<18} {dur_str:<10} Caption: {first_cap}")
        for extra_line in cap_lines[1:]:
            print(f"{'':<49}          {extra_line}")
        print(f"{'':<49} VO     : \"{s.voiceover}\"")
        if i < len(plan.scenes):
            print()

    if plan.warnings:
        print(f"\nPLAN WARNINGS ({len(plan.warnings)}):")
        for w in plan.warnings:
            print(f"  ⚠  {w}")

    print(SEP)


# ── Main ───────────────────────────────────────────────────────────────────

def open_video_in_player(path: Path) -> None:
    """Open video file in the OS default video player."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(path)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        logger.warning("Could not automatically open video: %s", exc)
        print(f"\n⚠ Could not open media player: {exc}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    images_dir = Path(args.images)

    from src.input_loader import load_interactive
    from src.pipeline import run_pipeline, PipelineOptions

    # Handle interactive input
    if args.interactive:
        load_res = load_interactive(images_dir)
        if any("cancelled by user" in e.lower() or "ctrl+c" in e.lower() for e in load_res.errors):
            return 130
        prop_input = load_res.property
        if not load_res.ok:
            print_summary(load_res)
            return 1
    else:
        prop_input = Path(args.data)

    # Determine LLM preference
    has_api_key = bool(os.getenv("LLM_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip())
    if args.no_llm:
        use_llm = False
    elif args.use_llm:
        use_llm = True
    else:
        use_llm = has_api_key

    options = PipelineOptions(
        use_llm=use_llm,
        no_video=args.no_video,
        size=args.size,
        fps=args.fps,
        out_path=args.out,
        script_out=args.script_out,
        no_audio=args.no_audio,
        no_music=args.no_music,
        music_path=args.music,
        voice=args.voice,
        tts_rate=args.tts_rate,
    )

    header_printed = False

    def progress_cb(stage: str, frac: float, msg: str) -> None:
        nonlocal header_printed
        if stage == "render":
            if not header_printed:
                res = "1920x1080" if args.size == "1080p" else "1280x720"
                SEP = "=" * 70
                print(f"\n{SEP}")
                print(f"  RENDERING VIDEO ({args.size} [{res}] @ {args.fps} fps)")
                print(SEP)
                header_printed = True
            if "scene" in msg.lower():
                print(f"  {msg}")

    result = run_pipeline(
        property_input=prop_input,
        images_input=images_dir,
        options=options,
        progress_cb=progress_cb,
    )

    print_summary(result)

    if not result.ok:
        print("=" * 60)
        logger.warning("Pipeline execution failed with %d error(s).", len(result.errors))
        return 1

    if result.plan:
        print_scene_plan(result.plan)
        if args.script_out:
            print(f"\n✓ Scene plan saved to: {args.script_out}")

    if args.no_video:
        print("\n✓ Skipping video rendering (--no-video flag set).")
        return 0

    if result.video_path:
        file_size_bytes = result.file_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        render_time = result.timings.get("render", 0.0)
        res_str = "1920x1080 (1080p)" if args.size == "1080p" else "1280x720 (720p)"

        SEP = "=" * 70
        print(f"\n{SEP}")
        print("  VIDEO GENERATION COMPLETE")
        print(SEP)
        print(f"  Output file : {result.video_path}")
        print(f"  Duration    : {result.plan.total_duration}s")
        print(f"  Resolution  : {res_str}")
        print(f"  File size   : {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
        print(f"  Render time : {render_time:.1f}s")
        print("\n  To view the video:")
        print(f'    Windows (PowerShell): Start-Process "{result.video_path}"')
        print(f'    macOS               : open "{result.video_path}"')
        print(f'    Linux               : xdg-open "{result.video_path}"')
        print(SEP)

        if args.open:
            open_video_in_player(result.video_path)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise   # let argparse / explicit sys.exit() propagate normally
    except Exception as exc:
        import traceback

        logger.exception("Unexpected error in main")
        print(f"\nUnexpected error: {exc}")
        print("Details have been written to logs/app.log")
        sys.exit(1)
