"""
main.py — CLI entry point for the AI Property Video Generator.

Phase 1: input loading and validation only.
Video generation will be added in Phase 3.

Exit codes:
  0 — validation passed
  1 — validation errors (bad data / missing images)
  2 — bad CLI usage (handled automatically by argparse)
"""

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
            "validates property data and images (Phase 1)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --data input/property.json\n"
            "  python main.py --data input/property.json --images input/images\n"
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
        "--out",
        metavar="PATH",
        default=str(BASE_DIR / "output" / "property_video.mp4"),
        help="Output video path (default: output/property_video.mp4). "
             "Unused in Phase 1.",
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

    # Phase stub
    print("\nVideo generation: not implemented yet (Phase 3)")
    print(SEP)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    images_dir = Path(args.images)

    from src.input_loader import load_from_json, load_interactive

    if args.interactive:
        result = load_interactive(images_dir)
    else:
        result = load_from_json(Path(args.data), images_dir)

    print_summary(result)

    if result.ok:
        logger.info("Input validation passed — %d image(s) ready.", len(result.images))
        return 0
    else:
        logger.warning(
            "Input validation failed with %d error(s).", len(result.errors)
        )
        return 1


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
