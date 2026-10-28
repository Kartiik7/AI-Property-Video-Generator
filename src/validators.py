"""
validators.py — Pure validation functions (no I/O, no printing).

Returns (result, warnings, errors) tuples so callers can decide
how to display issues.
"""

import re
from pathlib import Path
from PIL import Image, UnidentifiedImageError

from src.models import PropertyData

# ── Constants ──────────────────────────────────────────────────────────────

REQUIRED_TEXT_FIELDS = ["name", "location", "property_type", "cta_text"]
ALL_KNOWN_FIELDS = {
    "name", "location", "price", "property_type",
    "features", "cta_text", "contact", "tagline", "custom_script",
}

MAX_TEXT_LEN = 120    # chars; must fit on screen
MAX_FEATURE_LEN = 80  # chars per feature item
MAX_FEATURES = 6      # keep only the first N

ACCEPTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGES = 8
MIN_WIDTH, MIN_HEIGHT = 640, 360   # below this -> low-res warning


# ── Helpers ────────────────────────────────────────────────────────────────

def _format_indian_price(amount: int) -> str:
    """Format a positive integer as Indian-style rupee string.

    Examples:
        8500000 -> "₹85,00,000"
        100000  -> "₹1,00,000"
    """
    s = str(amount)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    groups: list[str] = []
    while len(rest) > 2:
        groups.append(rest[-2:])
        rest = rest[:-2]
    groups.append(rest)
    groups.reverse()
    return "₹" + ",".join(groups) + "," + last3


def _natural_key(path: Path) -> list:
    """Key for natural (human) sort: 'img2' < 'img10'."""
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# Matches a price string that is purely numeric after stripping a leading
# rupee symbol or 'Rs' prefix plus commas/spaces — e.g. "2000000",
# "₹20,00,000", "Rs 45,00,000". Strings like "45 Lakhs" or "On request"
# do NOT match and are kept exactly as entered.
_NUMERIC_STR_RE = re.compile(
    r'^(?:₹|[Rr][Ss]\.?\s*)?[\d,\s]+$'
)


def _parse_price_string(s: str) -> int | None:
    """Return the integer value of a numeric price string, or None.

    Strips a leading ₹/Rs prefix and commas/spaces, then checks whether
    what remains is a plain integer.  Returns None for textual prices.
    """
    if not _NUMERIC_STR_RE.match(s):
        return None
    # Remove prefix (₹ or Rs/rs with optional dot and spaces)
    cleaned = re.sub(r'^(?:₹|[Rr][Ss]\.?\s*)', '', s)
    # Remove formatting characters
    cleaned = cleaned.replace(',', '').replace(' ', '')
    return int(cleaned) if cleaned.isdigit() and cleaned else None


# ── Property validator ─────────────────────────────────────────────────────

def validate_property(
    data: dict,
) -> tuple[PropertyData | None, list[str], list[str]]:
    """Validate raw property data (from JSON or interactive input).

    Returns:
        (PropertyData | None, warnings, errors)
        PropertyData is None when any error is present.
    """
    warnings: list[str] = []
    errors: list[str] = []

    if not isinstance(data, dict):
        errors.append("Property data must be a JSON object {…}.")
        return None, warnings, errors

    # Warn on unknown / extra fields (never crash on them)
    for extra in sorted(set(data.keys()) - ALL_KNOWN_FIELDS):
        warnings.append(f"Unknown field '{extra}' will be ignored.")

    validated: dict = {}

    # ── Required plain-text fields ──────────────────────────────────────
    for field_name in REQUIRED_TEXT_FIELDS:
        raw = data.get(field_name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            errors.append(
                f"Missing required field '{field_name}'. "
                "Please provide a non-empty value."
            )
        else:
            value = str(raw).strip()
            if len(value) > MAX_TEXT_LEN:
                errors.append(
                    f"Field '{field_name}' is too long ({len(value)} chars). "
                    f"Maximum is {MAX_TEXT_LEN} characters (must fit on screen)."
                )
            else:
                validated[field_name] = value

    # ── Price ───────────────────────────────────────────────────────────
    raw_price = data.get("price")
    if raw_price is None or raw_price == "":
        errors.append(
            "Missing required field 'price'. "
            "Provide a number (e.g. 4500000) or a string (e.g. '₹45,00,000')."
        )
    elif isinstance(raw_price, str):
        price_str = raw_price.strip()
        if not price_str:
            errors.append(
                "Missing required field 'price'. Please provide a non-empty value."
            )
        elif len(price_str) > MAX_TEXT_LEN:
            errors.append(
                f"Field 'price' is too long ({len(price_str)} chars). "
                f"Maximum is {MAX_TEXT_LEN} characters."
            )
        else:
            # If the string is purely numeric (with optional ₹/Rs prefix and
            # commas), treat it like a JSON number and apply Indian formatting.
            # Genuinely textual prices ("₹45 Lakhs", "On request") are kept.
            parsed = _parse_price_string(price_str)
            if parsed is not None:
                if parsed <= 0:
                    errors.append(
                        f"Field 'price' must be a positive number, got '{price_str}'. "
                        "Please provide a value greater than zero."
                    )
                else:
                    validated["price"] = _format_indian_price(parsed)
            else:
                validated["price"] = price_str      # keep textual price as-is
    elif isinstance(raw_price, (int, float)):
        num = round(raw_price)                       # int-ify floats
        if num <= 0:
            errors.append(
                f"Field 'price' must be a positive number, got {raw_price}. "
                "Please provide a value greater than zero."
            )
        else:
            validated["price"] = _format_indian_price(num)
    else:
        errors.append(
            f"Field 'price' must be a string or number, "
            f"got {type(raw_price).__name__}."
        )

    # ── Features ────────────────────────────────────────────────────────
    raw_features = data.get("features")
    if raw_features is None:
        errors.append(
            "Missing required field 'features'. "
            "Provide a list of property highlights, e.g. [\"Garden\", \"Parking\"]."
        )
    elif not isinstance(raw_features, list):
        errors.append(
            "Field 'features' must be a list of strings, "
            "e.g. [\"Garden\", \"Parking\"]."
        )
    else:
        seen: set[str] = set()
        clean: list[str] = []

        for item in raw_features:
            item = str(item).strip()
            if not item:
                continue                             # drop blanks silently
            if len(item) > MAX_FEATURE_LEN:
                errors.append(
                    f"Feature '{item[:40]}…' is too long ({len(item)} chars). "
                    f"Maximum is {MAX_FEATURE_LEN} characters (must fit on screen)."
                )
                continue
            if item.lower() in seen:
                continue                             # drop duplicates silently
            seen.add(item.lower())
            clean.append(item)

        if not clean:
            errors.append(
                "Field 'features' has no valid entries. "
                "Provide at least one non-empty feature string shorter than "
                f"{MAX_FEATURE_LEN} characters."
            )
        else:
            n = len(clean)
            if n <= 2:
                warnings.append(
                    f"Only {n} feature(s) provided; the video will be short. "
                    "Recommended: 3–6 features."
                )
            if n > MAX_FEATURES:
                warnings.append(
                    f"{n} features provided; only the first {MAX_FEATURES} "
                    "will be used in the video."
                )
                clean = clean[:MAX_FEATURES]
            validated["features"] = clean

    # ── Contact (optional) ──────────────────────────────────────────────
    raw_contact = data.get("contact")
    if not raw_contact or (isinstance(raw_contact, str) and not raw_contact.strip()):
        warnings.append(
            "Optional field 'contact' is missing; "
            "no contact information will appear in the video."
        )
        validated["contact"] = None
    else:
        contact = str(raw_contact).strip()
        if len(contact) > MAX_TEXT_LEN:
            errors.append(
                f"Field 'contact' is too long ({len(contact)} chars). "
                f"Maximum is {MAX_TEXT_LEN} characters."
            )
        else:
            validated["contact"] = contact

    # ── Tagline (optional) ──────────────────────────────────────────────
    raw_tagline = data.get("tagline")
    if raw_tagline and isinstance(raw_tagline, str) and raw_tagline.strip():
        tagline = raw_tagline.strip()
        if len(tagline) > MAX_TEXT_LEN:
            errors.append(
                f"Field 'tagline' is too long ({len(tagline)} chars). "
                f"Maximum is {MAX_TEXT_LEN} characters."
            )
        else:
            validated["tagline"] = tagline
    else:
        validated["tagline"] = None

    # ── Custom script (optional) ────────────────────────────────────────
    raw_custom_script = data.get("custom_script")
    if raw_custom_script is not None:
        if isinstance(raw_custom_script, dict):
            validated["custom_script"] = raw_custom_script
        else:
            warnings.append(
                "Optional field 'custom_script' must be a JSON object; ignoring."
            )
            validated["custom_script"] = None
    else:
        validated["custom_script"] = None

    # ── Build result ────────────────────────────────────────────────────
    if errors:
        return None, warnings, errors

    prop = PropertyData(
        name=validated["name"],
        location=validated["location"],
        price=validated["price"],
        property_type=validated["property_type"],
        features=validated["features"],
        cta_text=validated["cta_text"],
        contact=validated.get("contact"),
        tagline=validated.get("tagline"),
        custom_script=validated.get("custom_script"),
    )
    return prop, warnings, errors


# ── Image validator ────────────────────────────────────────────────────────

def validate_images(
    folder: Path,
) -> tuple[list[Path], list[str], list[str]]:
    """Validate and filter images in *folder*.

    Returns:
        (accepted_paths, warnings, errors)
        accepted_paths is empty on error.
    """
    warnings: list[str] = []
    errors: list[str] = []

    # ── Folder checks ───────────────────────────────────────────────────
    if not folder.exists():
        errors.append(
            f"Images folder not found: '{folder}'. "
            "Create the directory and add property images."
        )
        return [], warnings, errors
    if not folder.is_dir():
        errors.append(
            f"'{folder}' is not a directory. "
            "Provide a path to a directory containing images."
        )
        return [], warnings, errors

    # ── Scan files ──────────────────────────────────────────────────────
    all_files = sorted(folder.iterdir(), key=_natural_key)
    accepted: list[Path] = []

    for f in all_files:
        if not f.is_file():
            continue
        # Skip hidden files (e.g. .gitkeep, .DS_Store) silently — no warning.
        if f.name.startswith('.'):
            continue
        ext = f.suffix.lower()
        if ext not in ACCEPTED_EXTENSIONS:
            warnings.append(
                f"'{f.name}' is not a supported image type "
                f"({ext or 'no extension'}) — skipped. "
                f"Accepted: {', '.join(sorted(ACCEPTED_EXTENSIONS))}."
            )
            continue

        # Verify the image can be opened and decoded
        try:
            with Image.open(f) as img:
                img.load()          # force full decode — detects corruption
                w, h = img.size
        except (UnidentifiedImageError, Exception) as exc:
            warnings.append(
                f"'{f.name}' could not be read (possibly corrupt): {exc}. "
                "Skipping this file."
            )
            continue

        # Low-resolution warning (image is still accepted)
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            warnings.append(
                f"'{f.name}' is low-resolution ({w}×{h}). "
                f"Recommended minimum: {MIN_WIDTH}×{MIN_HEIGHT}. "
                "The image will still be used but may appear blurry."
            )

        accepted.append(f)

    # ── Count checks ────────────────────────────────────────────────────
    if not accepted:
        errors.append(
            f"No usable images found in '{folder}'. "
            f"Add .jpg, .jpeg, .png, or .webp files and try again."
        )
        return [], warnings, errors

    if len(accepted) <= 3:
        warnings.append(
            f"Only {len(accepted)} image(s) found. "
            "Recommended: 4–8 images. The video will be shorter."
        )

    if len(accepted) > MAX_IMAGES:
        warnings.append(
            f"{len(accepted)} images found; only the first {MAX_IMAGES} "
            "will be used in the video."
        )
        accepted = accepted[:MAX_IMAGES]

    return accepted, warnings, errors
