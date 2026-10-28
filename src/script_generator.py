"""
script_generator.py — Template-based (and optional LLM) storyboard and script generation.

Generates ScenePlan:
  - Title scene (property name + type + tagline)
  - Feature scenes (up to 4 scenes, merging remaining features)
  - Price & Location scene
  - CTA scene (cta_text + contact)

Enforces:
  - Strict facts-only rule (verified via fact_checker)
  - Caption: max 2 lines, max 38 chars/line, wrapped at word boundaries
  - Voiceover: max 18 words, short natural sentences
  - Scene duration >= 3.0s (title/CTA min 4.0s), total between 20s and 40s
  - Deterministic variant selection (same input -> same output)
"""

import os
import re
import hashlib
import textwrap
from pathlib import Path
from typing import TypeVar

from src.models import PropertyData, Scene, ScenePlan
from src.fact_checker import extract_allowed_facts, check_text

T = TypeVar("T")


# ── Spoken price formatter ──────────────────────────────────────────────────

def format_spoken_price(price_str: str) -> str:
    """Convert formatted rupee string to natural spoken English.

    Examples:
        "₹45,00,000"   -> "45 lakh rupees"
        "₹1,20,00,000" -> "1 crore 20 lakh rupees"
        "4500000"      -> "45 lakh rupees"
        "On request"   -> "On request"
    """
    if not price_str or not isinstance(price_str, str):
        return str(price_str)

    # Strip symbols and delimiters to check if purely numeric
    cleaned = re.sub(r"^(?:₹|[Rr][Ss]\.?\s*)", "", price_str.strip())
    digits_only = cleaned.replace(",", "").replace(" ", "")

    if not digits_only.isdigit() or not digits_only:
        # Non-numeric text like "On request", "Call for price"
        return price_str.strip()

    num = int(digits_only)
    if num <= 0:
        return price_str.strip()

    parts: list[str] = []
    crores = num // 10_000_000
    remainder = num % 10_000_000

    lakhs = remainder // 100_000
    remainder = remainder % 100_000

    thousands = remainder // 1_000
    remainder = remainder % 1_000

    if crores > 0:
        parts.append(f"{crores} crore")
    if lakhs > 0:
        parts.append(f"{lakhs} lakh")
    if thousands > 0:
        parts.append(f"{thousands} thousand")
    if remainder > 0 and not parts:
        parts.append(str(remainder))

    if not parts:
        return f"{num} rupees"

    return " ".join(parts) + " rupees"


# ── Text formatting helpers ────────────────────────────────────────────────

def wrap_caption(text: str, max_lines: int = 2, max_line_len: int = 38) -> str:
    """Wrap caption to at most *max_lines* of *max_line_len* characters.
    
    Splits at word boundaries and never cuts mid-word.
    """
    lines: list[str] = []
    # If the text already has newlines, process each line individually
    for segment in text.split("\n"):
        wrapped = textwrap.wrap(segment.strip(), width=max_line_len)
        if wrapped:
            lines.extend(wrapped)
        else:
            lines.append("")

    # Enforce max lines
    if len(lines) > max_lines:
        lines = lines[:max_lines]

    return "\n".join(lines).strip()


def limit_words(text: str, max_words: int = 18) -> str:
    """Ensure voiceover has at most *max_words*, preserving sentence ending."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    truncated = " ".join(words[:max_words]).rstrip(",;")
    if not truncated.endswith((".", "!", "?")):
        truncated += "."
    return truncated


def _pick_variant(variants: list[T], seed_key: str, salt: str = "") -> T:
    """Deterministically pick a variant using MD5 hash."""
    h = hashlib.md5((seed_key + salt).encode("utf-8")).hexdigest()
    idx = int(h, 16) % len(variants)
    return variants[idx]


# ── Feature grouping ───────────────────────────────────────────────────────

def group_features(features: list[str], max_scenes: int = 4) -> list[str]:
    """Group features into at most *max_scenes*.

    If there are more than *max_scenes*, merge the extras into the final scene.
    Example: 5 features -> 3 single features + 1 merged pair ("Feat 4 & Feat 5").
    """
    clean = [f.strip() for f in features if f.strip()]
    if not clean:
        return ["Key Amenities"]

    if len(clean) <= max_scenes:
        return clean

    # First (max_scenes - 1) features stay individual
    grouped = clean[: max_scenes - 1]
    # Merge all remaining features into the last item
    remaining = clean[max_scenes - 1 :]
    merged = " & ".join(remaining)
    grouped.append(merged)
    return grouped


# ── Duration calculation ───────────────────────────────────────────────────

def calculate_durations(scenes_data: list[dict]) -> list[float]:
    """Estimate and balance durations so total is between 20.0s and 40.0s.

    Rules:
      - Raw estimate = max(3.5, words / 2.5 + 0.8)
      - Title and CTA minimum 4.0s
      - Scene minimum >= 3.0s
      - If total < 20s, expand scenes evenly
      - If total > 40s, trim pause times evenly down to 3.0s minimum
    """
    n = len(scenes_data)
    durations: list[float] = []

    for item in scenes_data:
        stype = item["scene_type"]
        vo = item["voiceover"]
        words = len(vo.split())
        est = max(3.5, (words / 2.5) + 0.8)
        if stype in ("title", "cta"):
            est = max(4.0, est)
        durations.append(est)

    total = sum(durations)

    if total < 20.0 and n > 0:
        add_per_scene = (20.0 - total) / n
        durations = [d + add_per_scene for d in durations]
    elif total > 40.0 and n > 0:
        excess = total - 40.0
        trim_per_scene = excess / n
        new_durations = []
        for d in durations:
            new_durations.append(max(3.0, d - trim_per_scene))
        durations = new_durations

    return [round(d, 1) for d in durations]


# ── Template Wording Helpers ───────────────────────────────────────────────

def format_property_type(prop_type: str) -> str:
    """Format property type with correct indefinite article (a/an) in lowercase.

    Examples:
        "Independent House" -> "an independent house"
        "Apartment"         -> "an apartment"
        "Villa"             -> "a villa"
    """
    clean = prop_type.strip().lower()
    if not clean:
        return ""
    article = "an" if clean[0] in "aeiou" else "a"
    return f"{article} {clean}"


def format_feature_for_voiceover(feat: str) -> str:
    """Format feature text for voiceover mid-sentence.

    Lowercases the first letter unless it starts with a digit or an all-caps word.
    Examples:
        "Spacious kitchen" -> "spacious kitchen"
        "24/7 Security"    -> "24/7 Security"
        "CCTV cover"       -> "CCTV cover"
    """
    s = feat.strip()
    if not s:
        return s
    first_word = s.split()[0]
    if s[0].isdigit():
        return s
    if first_word.isupper() and len(first_word) >= 2:
        return s
    return s[0].lower() + s[1:]


# ── Template Scene Builders ────────────────────────────────────────────────

def _build_title_scene(prop: PropertyData) -> dict:
    caption_text = f"{prop.name}\n{prop.property_type}"
    if prop.tagline and len(prop.tagline) <= 38:
        caption_text = f"{prop.name}\n{prop.tagline}"

    ptype = format_property_type(prop.property_type)
    base_variants = [
        f"Welcome to {prop.name}, {ptype}.",
        f"Presenting {prop.name}, offering {ptype}.",
        f"Discover {prop.name}, {ptype}.",
    ]
    base_vo = _pick_variant(base_variants, prop.name, "title")

    if prop.tagline and prop.tagline.strip():
        tagline_clean = prop.tagline.strip().rstrip(".!?,;:") + "."
        candidate_vo = f"{base_vo} {tagline_clean}"
        # Keep title voiceover to max 18 words; drop tagline from VO if too long
        if len(candidate_vo.split()) <= 18:
            vo = candidate_vo
        else:
            vo = base_vo
    else:
        vo = base_vo

    return {
        "scene_type": "title",
        "caption": wrap_caption(caption_text),
        "voiceover": limit_words(vo),
    }


def _build_feature_scene(prop: PropertyData, feat: str, idx: int) -> dict:
    feat_text = format_feature_for_voiceover(feat).rstrip(".!?,;:")
    vo_variants = [
        f"Highlights include {feat_text}.",
        f"Next up: {feat_text}.",
        f"Also featured: {feat_text}.",
    ]
    vo = _pick_variant(vo_variants, prop.name, f"feat_{idx}")

    return {
        "scene_type": "feature",
        "caption": wrap_caption(feat),
        "voiceover": limit_words(vo),
    }


def _build_price_location_scene(prop: PropertyData) -> dict:
    caption_text = f"{prop.price}\n{prop.location}"
    spoken_price = format_spoken_price(prop.price)

    vo_variants = [
        f"Priced at {spoken_price}, located at {prop.location}.",
        f"Located at {prop.location}, priced at {spoken_price}.",
        f"Available at {spoken_price} in {prop.location}.",
    ]
    vo = _pick_variant(vo_variants, prop.name, "price_loc")

    return {
        "scene_type": "price_location",
        "caption": wrap_caption(caption_text),
        "voiceover": limit_words(vo),
    }


def _build_cta_scene(prop: PropertyData) -> dict:
    caption_text = f"{prop.cta_text}\n{prop.contact}" if prop.contact else prop.cta_text
    cta_clean = prop.cta_text.strip().rstrip(".!?,;:")
    contact_part = " Contact details are on screen." if prop.contact else ""

    vo = f"{cta_clean}.{contact_part}".strip()

    return {
        "scene_type": "cta",
        "caption": wrap_caption(caption_text),
        "voiceover": limit_words(vo),
    }


# ── Custom Script Validation ────────────────────────────────────────────────

def _validate_custom_scene(
    scene_label: str,
    custom_entry: dict,
    allowed_facts,
) -> tuple[dict | None, list[str]]:
    """Validate a custom caption and voiceover against formatting limits and facts."""
    if not isinstance(custom_entry, dict):
        return None, [f"Custom script for {scene_label} scene must be a JSON object with 'caption' and 'voiceover'."]

    cap = custom_entry.get("caption")
    vo = custom_entry.get("voiceover")
    if not isinstance(cap, str) or not isinstance(vo, str):
        return None, [f"Custom script for {scene_label} scene must have string 'caption' and 'voiceover'."]

    problems: list[str] = []

    # Check caption limits: max 2 lines, max 38 chars/line
    cap_lines = cap.split("\n")
    if len(cap_lines) > 2:
        problems.append(f"caption has {len(cap_lines)} lines (max 2)")
    for line_idx, line in enumerate(cap_lines, 1):
        if len(line) > 38:
            problems.append(f"caption line {line_idx} has {len(line)} chars (max 38)")
    if not cap.strip():
        problems.append("caption is empty")

    # Check voiceover limits: max 18 words
    vo_words = vo.split()
    if len(vo_words) > 18:
        problems.append(f"voiceover has {len(vo_words)} words (max 18)")
    if len(vo_words) == 0:
        problems.append("voiceover is empty")

    # Fact check
    ok_c, prob_c = check_text(cap, allowed_facts)
    ok_v, prob_v = check_text(vo, allowed_facts)
    for p in prob_c:
        problems.append(f"caption fact check failed: {p}")
    for p in prob_v:
        problems.append(f"voiceover fact check failed: {p}")

    if problems:
        return None, problems

    return {"caption": cap.strip(), "voiceover": vo.strip()}, []


def _assign_scene_images(
    scenes_data: list[dict],
    images: list[Path],
    warnings: list[str],
) -> list[Path]:
    """Assign images to scenes, supporting room-based ordering when available."""
    total_scenes = len(scenes_data)
    num_images = len(images)

    if num_images == 0:
        warnings.append("No images provided; placeholder paths used.")
        return [Path(f"image_{i+1}.jpg") for i in range(total_scenes)]

    if num_images < total_scenes:
        warnings.append(
            f"Only {num_images} image(s) available for {total_scenes} scenes; "
            "images were reused."
        )

    # Check if images have recognizable room names (e.g. 1_kitchen.jpg, 1_bedroom.jpg, 1_frontal.jpg)
    room_keys = ("kitchen", "bedroom", "bathroom", "bath", "bed", "frontal", "front", "exterior")
    has_room_tags = any(
        any(rk in img.stem.lower() for rk in room_keys)
        for img in images
    )

    if not has_room_tags:
        # Standard sequential assignment (Phase 2 default)
        return [images[i % num_images] for i in range(total_scenes)]

    # Helper to find matching image
    def find_img(*keywords: str) -> Path | None:
        for kw in keywords:
            for img in images:
                if kw in img.stem.lower():
                    return img
        return None

    frontal_img = find_img("frontal", "front", "exterior", "elevation") or images[0]
    kitchen_img = find_img("kitchen")
    bedroom_img = find_img("bedroom", "bed")
    bathroom_img = find_img("bathroom", "bath")

    assigned: list[Path] = []
    for i, s in enumerate(scenes_data):
        stype = s["scene_type"]
        if stype == "title":
            assigned.append(frontal_img)
        elif stype == "price_location" or stype == "cta":
            assigned.append(frontal_img)
        elif stype == "feature":
            text = (s.get("caption", "") + " " + s.get("voiceover", "")).lower()
            if "kitchen" in text and kitchen_img:
                assigned.append(kitchen_img)
            elif ("bedroom" in text or "bed" in text) and bedroom_img:
                assigned.append(bedroom_img)
            elif ("bathroom" in text or "bath" in text) and bathroom_img:
                assigned.append(bathroom_img)
            else:
                assigned.append(images[i % num_images])
        else:
            assigned.append(images[i % num_images])

    return assigned


# ── Main Entry Point ───────────────────────────────────────────────────────

def build_scene_plan(
    prop: PropertyData,
    images: list[Path],
    use_llm: bool | None = None,
) -> ScenePlan:
    """Build a complete ScenePlan from validated PropertyData and image paths.

    If custom_script is present on PropertyData, each valid custom line replaces the
    template text for that scene and takes priority.

    If use_llm is True (or None with LLM_API_KEY set in .env), attempts to enhance
    captions/voiceovers per-scene via OpenAI-compatible endpoint with per-scene fact checks
    and targeted retry. Failing scenes fall back to deterministic template text.
    """
    if use_llm is None:
        use_llm = bool(os.getenv("LLM_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip())

    warnings: list[str] = []

    # ── 1. Create skeleton scenes ─────────────────────────────────────────
    scenes_data: list[dict] = []

    # Title scene
    scenes_data.append(_build_title_scene(prop))

    # Feature scenes (grouped to max 4)
    grouped_feats = group_features(prop.features, max_scenes=4)
    for idx, feat in enumerate(grouped_feats):
        scenes_data.append(_build_feature_scene(prop, feat, idx))

    # Price & Location scene
    scenes_data.append(_build_price_location_scene(prop))

    # CTA scene
    scenes_data.append(_build_cta_scene(prop))

    allowed_facts = extract_allowed_facts(prop)
    template_scenes = [dict(s) for s in scenes_data]

    # ── 2. Apply optional custom_script ───────────────────────────────────
    custom_covered_indices: set[int] = set()
    custom_scenes_count = 0
    if prop.custom_script and isinstance(prop.custom_script, dict):
        cs = prop.custom_script

        # Title
        if "title" in cs:
            valid, problems = _validate_custom_scene("title", cs["title"], allowed_facts)
            if valid:
                scenes_data[0]["caption"] = valid["caption"]
                scenes_data[0]["voiceover"] = valid["voiceover"]
                custom_covered_indices.add(0)
                custom_scenes_count += 1
            else:
                warnings.append(
                    f"Custom script for title scene rejected: {'; '.join(problems)}. Using template text."
                )

        # Features
        if "features" in cs and isinstance(cs["features"], list):
            for f_idx, custom_feat in enumerate(cs["features"]):
                scene_idx = 1 + f_idx
                if scene_idx < 1 + len(grouped_feats):
                    valid, problems = _validate_custom_scene(
                        f"feature {f_idx + 1}", custom_feat, allowed_facts
                    )
                    if valid:
                        scenes_data[scene_idx]["caption"] = valid["caption"]
                        scenes_data[scene_idx]["voiceover"] = valid["voiceover"]
                        custom_covered_indices.add(scene_idx)
                        custom_scenes_count += 1
                    else:
                        warnings.append(
                            f"Custom script for feature {f_idx + 1} scene rejected: {'; '.join(problems)}. Using template text."
                        )

        # Price & Location
        pl_idx = 1 + len(grouped_feats)
        if "price_location" in cs:
            valid, problems = _validate_custom_scene(
                "price_location", cs["price_location"], allowed_facts
            )
            if valid:
                scenes_data[pl_idx]["caption"] = valid["caption"]
                scenes_data[pl_idx]["voiceover"] = valid["voiceover"]
                custom_covered_indices.add(pl_idx)
                custom_scenes_count += 1
            else:
                warnings.append(
                    f"Custom script for price_location scene rejected: {'; '.join(problems)}. Using template text."
                )

        # CTA
        cta_idx = pl_idx + 1
        if "cta" in cs:
            valid, problems = _validate_custom_scene("cta", cs["cta"], allowed_facts)
            if valid:
                scenes_data[cta_idx]["caption"] = valid["caption"]
                scenes_data[cta_idx]["voiceover"] = valid["voiceover"]
                custom_covered_indices.add(cta_idx)
                custom_scenes_count += 1
            else:
                warnings.append(
                    f"Custom script for cta scene rejected: {'; '.join(problems)}. Using template text."
                )

    if custom_scenes_count > 0:
        warnings.append(f"Custom script used for {custom_scenes_count} scenes.")

    # ── 3. Optional LLM mode ──────────────────────────────────────────────
    llm_scenes_used = 0
    if use_llm:
        from src.llm_client import generate_scene_copy
        scene_skeleton = []
        for idx, s in enumerate(template_scenes):
            sk = {"scene_index": idx + 1, "scene_type": s["scene_type"]}
            if s["scene_type"] == "feature":
                sk["feature"] = s["caption"]
            scene_skeleton.append(sk)

        llm_scenes, llm_warnings = generate_scene_copy(prop, scene_skeleton)
        warnings.extend(llm_warnings)

        if llm_scenes is not None:
            for idx, s in enumerate(llm_scenes):
                # custom_script in property.json still takes priority for the scenes it covers
                if idx in custom_covered_indices:
                    continue
                if s is not None:
                    scenes_data[idx]["caption"] = s["caption"]
                    scenes_data[idx]["voiceover"] = s["voiceover"]
                    llm_scenes_used += 1

    total_scenes = len(scenes_data)
    if llm_scenes_used == total_scenes:
        source = "llm"
    elif llm_scenes_used == 0:
        source = "template"
    else:
        source = f"mixed ({llm_scenes_used} of {total_scenes} scenes from LLM)"

    # ── 4. Calculate durations ────────────────────────────────────────────
    durations = calculate_durations(scenes_data)

    # ── 5. Assign images ──────────────────────────────────────────────────
    assigned_images = _assign_scene_images(scenes_data, images, warnings)

    # ── 6. Assemble Scene objects ─────────────────────────────────────────
    final_scenes: list[Scene] = []

    for i, data in enumerate(scenes_data):
        cap = wrap_caption(data["caption"])
        vo = limit_words(data["voiceover"])
        dur = max(3.0, durations[i])

        # Self-check facts on generated copy
        ok_c, prob_c = check_text(cap, allowed_facts)
        ok_v, prob_v = check_text(vo, allowed_facts)
        for p in prob_c + prob_v:
            warnings.append(f"Fact check notice (Scene {i+1}): {p}")

        final_scenes.append(
            Scene(
                scene_type=data["scene_type"],
                image_path=assigned_images[i],
                caption=cap,
                voiceover=vo,
                duration=dur,
            )
        )

    return ScenePlan(scenes=final_scenes, source=source, warnings=warnings)
