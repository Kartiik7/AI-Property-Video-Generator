"""
input_loader.py — Load property data from JSON file or interactive prompts.

Both public functions return a LoadResult and NEVER raise for
user-input problems; all issues go into warnings/errors.
"""

import json
from pathlib import Path

from src.models import LoadResult
from src.validators import validate_property, validate_images


# ── JSON loader ────────────────────────────────────────────────────────────

def load_from_json(
    json_path: Path | str,
    images_dir: Path | str,
) -> LoadResult:
    """Load and validate property data from *json_path* + images in *images_dir*.

    Handles: file-not-found, bad encoding, invalid JSON (with line/col),
    JSON that is not an object, and all validation rules.
    """
    json_path = Path(json_path)
    images_dir = Path(images_dir)
    warnings: list[str] = []
    errors: list[str] = []

    # ── Read file ────────────────────────────────────────────────────────
    try:
        # utf-8-sig strips the BOM that some Windows editors add
        text = json_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        errors.append(
            f"Property data file not found: '{json_path}'. "
            "Create the file or use --interactive to enter data manually."
        )
        return LoadResult(property=None, images=[], warnings=warnings, errors=errors)
    except Exception as exc:
        errors.append(f"Could not read '{json_path}': {exc}")
        return LoadResult(property=None, images=[], warnings=warnings, errors=errors)

    # ── Parse JSON ───────────────────────────────────────────────────────
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        errors.append(
            f"Invalid JSON in '{json_path}' "
            f"at line {exc.lineno}, column {exc.colno}: {exc.msg}. "
            "Fix the syntax and try again."
        )
        return LoadResult(property=None, images=[], warnings=warnings, errors=errors)

    # ── Must be a dict ────────────────────────────────────────────────────
    if not isinstance(raw, dict):
        errors.append(
            f"'{json_path}' must contain a JSON object {{…}}, "
            f"but got a {type(raw).__name__}. "
            "Wrap your data in curly braces {{}}."
        )
        return LoadResult(property=None, images=[], warnings=warnings, errors=errors)

    # ── Validate property fields ─────────────────────────────────────────
    prop, pw, pe = validate_property(raw)
    warnings.extend(pw)
    errors.extend(pe)

    # ── Validate images ──────────────────────────────────────────────────
    images, iw, ie = validate_images(images_dir)
    warnings.extend(iw)
    errors.extend(ie)

    return LoadResult(property=prop, images=images, warnings=warnings, errors=errors)


# ── Interactive loader ─────────────────────────────────────────────────────

def load_interactive(images_dir: Path | str) -> LoadResult:
    """Prompt the user for each property field via the terminal.

    Features are entered one per line; a blank line finishes input.
    Ctrl+C exits cleanly with a friendly message (no traceback).
    """
    images_dir = Path(images_dir)
    warnings: list[str] = []
    errors: list[str] = []
    data: dict = {}

    print("\n" + "=" * 60)
    print("  Property Video Generator — Interactive Input")
    print("  (Press Ctrl+C at any time to cancel)")
    print("=" * 60 + "\n")

    try:
        # ── Required text fields ─────────────────────────────────────────
        _PROMPTS = [
            ("name",          "Property name"),
            ("location",      "Location (city / area)"),
            ("price",         "Price  (number or string, e.g. 4500000)"),
            ("property_type", "Property type  (e.g. Apartment, Villa)"),
            ("cta_text",      "Call-to-action  (e.g. Book a site visit today)"),
        ]
        for field_name, prompt in _PROMPTS:
            while True:
                value = input(f"  {prompt}: ").strip()
                if value:
                    data[field_name] = value
                    break
                print(f"  [!] '{field_name}' is required — please enter a value.\n")

        # ── Features ─────────────────────────────────────────────────────
        print("\n  Features — enter one per line, then press Enter on a blank line:")
        features: list[str] = []
        while True:
            line = input("    > ").strip()
            if not line:
                if not features:
                    print("  [!] Please enter at least one feature.\n")
                    continue
                break
            features.append(line)
        data["features"] = features

        # ── Optional fields ───────────────────────────────────────────────
        print()
        contact = input("  Contact info  (optional — press Enter to skip): ").strip()
        if contact:
            data["contact"] = contact

        tagline = input("  Tagline       (optional — press Enter to skip): ").strip()
        if tagline:
            data["tagline"] = tagline

    except KeyboardInterrupt:
        print("\n\n  [!] Input cancelled (Ctrl+C). No video will be generated.")
        errors.append("Input cancelled by user (Ctrl+C).")
        return LoadResult(property=None, images=[], warnings=warnings, errors=errors)

    # ── Validate collected data ──────────────────────────────────────────
    prop, pw, pe = validate_property(data)
    warnings.extend(pw)
    errors.extend(pe)

    images, iw, ie = validate_images(images_dir)
    warnings.extend(iw)
    errors.extend(ie)

    return LoadResult(property=prop, images=images, warnings=warnings, errors=errors)
