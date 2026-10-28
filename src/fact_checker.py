"""
fact_checker.py — Verify that generated scripts and captions use ONLY supplied facts.

Rules:
1. No numbers or currency amounts not present in the input.
2. No amenity/feature keywords not present in the input.
3. Banned promotional claims (e.g. luxury, guaranteed, RERA, near metro)
   are disallowed unless the word explicitly appears in the input.
"""

import re
from src.models import PropertyData

# ── Banned claims list ──────────────────────────────────────────────────────
# Words or phrases that imply unverified claims or standard marketing fluff
BANNED_CLAIMS = [
    "luxury",
    "luxurious",
    "premium",
    "best",
    "best in class",
    "guaranteed",
    "no.1",
    "no 1",
    "number 1",
    "rera",
    "rera approved",
    "near metro",
    "close to metro",
    "investment return",
    "high return",
    "roi",
    "discount",
    "offer",
    "limited time",
    "exclusive",
    "world-class",
    "world class",
    "unmatched",
    "zero brokerage",
]

# ── Common real-estate amenities ───────────────────────────────────────────
# If any of these are mentioned in copy, they MUST be in the input facts
KNOWN_AMENITIES = [
    "gym",
    "gymnasium",
    "swimming pool",
    "pool",
    "clubhouse",
    "club house",
    "spa",
    "squash court",
    "squash",
    "tennis court",
    "tennis",
    "badminton",
    "badminton court",
    "yoga",
    "yoga deck",
    "elevator",
    "lift",
    "parking",
    "covered parking",
    "open parking",
    "garden",
    "private garden",
    "terrace",
    "rooftop terrace",
    "rooftop pool",
    "rooftop swimming pool",
    "balcony",
    "power backup",
    "generator",
    "security",
    "24/7 security",
    "cctv",
    "intercom",
    "gas pipeline",
    "piped gas",
    "modular kitchen",
    "play area",
    "children play area",
    "kids play area",
    "jogging track",
    "community hall",
    "amphitheatre",
    "amphitheater",
    "jacuzzi",
    "sauna",
    "wifi",
    "ev charging",
    "solar",
]


class AllowedFacts(set):
    """Set of allowed tokens, numbers, and amenities derived from PropertyData."""
    def __init__(
        self,
        elements=(),
        raw_text: str = "",
        numbers: set[str] | None = None,
        amenities: set[str] | None = None,
        input_lower: str = "",
    ):
        super().__init__(elements)
        self.raw_text = raw_text
        self.numbers = numbers or set()
        self.amenities = amenities or set()
        self.input_lower = input_lower


def _extract_numbers(text: str) -> set[str]:
    """Extract numeric sequences from text (e.g. '4500000', '2', '3BHK' -> '3')."""
    nums = set(re.findall(r"\b\d+\b", text))
    # Also extract digits embedded in words like 2BHK, 3BHK
    for embedded in re.findall(r"(\d+)[A-Za-z]+", text):
        nums.add(embedded)
    return nums


def extract_allowed_facts(prop: PropertyData) -> AllowedFacts:
    """Extract allowed tokens, numbers, amenities, and phrases from PropertyData."""
    parts: list[str] = [
        prop.name,
        prop.location,
        str(prop.price),
        prop.property_type,
        " ".join(prop.features),
        prop.cta_text,
    ]
    if prop.contact:
        parts.append(prop.contact)
    if prop.tagline:
        parts.append(prop.tagline)

    full_text = " ".join(parts)
    full_lower = full_text.lower()

    # Extract all numbers from input
    numbers = _extract_numbers(full_text)

    # If price has numeric amount, also allow Indian currency word parts (lakh, crore, etc.)
    # e.g. "4500000" implies 45 and lakh
    cleaned_price = re.sub(r"[^\d]", "", str(prop.price))
    if cleaned_price:
        numbers.add(cleaned_price)
        val = int(cleaned_price)
        if val >= 10_000_000:
            cr = val // 10_000_000
            rem_lakh = (val % 10_000_000) // 100_000
            numbers.add(str(cr))
            if rem_lakh > 0:
                numbers.add(str(rem_lakh))
        elif val >= 100_000:
            lk = val // 100_000
            numbers.add(str(lk))

        th = (val % 100_000) // 1_000
        if th > 0:
            numbers.add(str(th))

        # Add Indian-formatted digit groups (e.g. '85', '00', '000', '1', '25')
        s = str(val)
        if len(s) > 3:
            numbers.add(s[-3:])
            rest = s[:-3]
            while len(rest) > 2:
                numbers.add(rest[-2:])
                rest = rest[:-2]
            numbers.add(rest)

    # Identify which known amenities are present in the input
    amenities: set[str] = set()
    for amenity in KNOWN_AMENITIES:
        pattern = r"\b" + re.escape(amenity) + r"\b"
        if re.search(pattern, full_lower):
            amenities.add(amenity)

    # Word tokens
    tokens = set(re.findall(r"[a-zA-Z0-9₹]+", full_lower))
    # Add common currency / spoken price words if price is provided
    tokens.update({"rupee", "rupees", "inr", "rs", "lakh", "lakhs", "crore", "crores", "thousand", "thousands"})

    return AllowedFacts(
        elements=tokens,
        raw_text=full_text,
        numbers=numbers,
        amenities=amenities,
        input_lower=full_lower,
    )


def check_text(text: str, allowed: AllowedFacts | set[str]) -> tuple[bool, list[str]]:
    """Check text against allowed facts.
    
    Returns:
        (ok, problems)
        ok is True when problems is empty.
    """
    problems: list[str] = []
    text_lower = text.lower()

    if isinstance(allowed, AllowedFacts):
        input_lower = allowed.input_lower
        allowed_nums = allowed.numbers
        allowed_amenities = allowed.amenities
    else:
        input_lower = " ".join(allowed).lower()
        allowed_nums = _extract_numbers(input_lower)
        allowed_amenities = {
            a for a in KNOWN_AMENITIES if re.search(r"\b" + re.escape(a) + r"\b", input_lower)
        }

    # ── 1. Check for banned claims ─────────────────────────────────────────
    for claim in BANNED_CLAIMS:
        pattern = r"\b" + re.escape(claim) + r"\b"
        if re.search(pattern, text_lower):
            if not re.search(pattern, input_lower):
                problems.append(
                    f"Banned promotional claim '{claim}' found in text but was not in input facts."
                )

    # ── 2. Check for unmentioned amenities ─────────────────────────────────
    for amenity in KNOWN_AMENITIES:
        pattern = r"\b" + re.escape(amenity) + r"\b"
        if re.search(pattern, text_lower):
            if amenity not in allowed_amenities:
                if not re.search(pattern, input_lower):
                    problems.append(
                        f"Amenity '{amenity}' mentioned in text was not present in property features/facts."
                    )

    # ── 3. Check for foreign numbers ───────────────────────────────────────
    text_nums = _extract_numbers(text)
    for num in text_nums:
        if num not in allowed_nums:
            problems.append(
                f"Number '{num}' found in text does not match any number in property facts."
            )

    return (len(problems) == 0, problems)
