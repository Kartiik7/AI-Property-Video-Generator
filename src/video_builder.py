"""
video_builder.py — Renders high-quality promotional MP4 videos from a ScenePlan.

Key features:
1. Smart background: Cover-crop + Gaussian blur + darkening behind sharp contain-fit foreground.
2. Motion: Smooth Ken Burns slow zoom (1.00 to ~1.08) with alternating direction and reuse-aware flipping.
3. Typography: Pillow-rendered custom text overlays with auto-fitted fonts, drop shadows, safe margins,
   scene-specific color palettes (title hero, amber price/CTA, white contact), and a 0.4s fade-in.
4. Transitions: 0.5s scene crossfades (total duration strictly matches plan), 0.5s fade from black, 0.7s fade to black.
5. Reliability: Fault-tolerant image loading with solid dark brand fallback; atomic output file writes.
"""

import os
import re
import time
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from moviepy import VideoClip, CompositeVideoClip, vfx

from src.models import PropertyData, ScenePlan, Scene

logger = logging.getLogger("property_video")

# ── Font resolution & glyph checks ──────────────────────────────────────────

WINDOWS_FONTS = [
    Path("C:/Windows/Fonts/segoeuib.ttf"),
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
]

UNIX_FONTS = [
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
    Path("/Library/Fonts/Arial Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
]


def font_has_glyph(font_path: Path, glyph: str = "₹") -> bool:
    """Check if font file can render the specified glyph."""
    try:
        f = ImageFont.truetype(str(font_path), 32)
        m = f.getmask(glyph)
        return m.size[0] > 0 and m.size[1] > 0
    except Exception:
        return False


def get_best_font_path(warnings: list[str] | None = None) -> tuple[Path | None, bool]:
    """Resolve the best available system font and check for rupee glyph support.

    Returns:
        (font_path, supports_rupee)
    """
    candidates = WINDOWS_FONTS + UNIX_FONTS
    available = [p for p in candidates if p.exists() and p.is_file()]

    # First attempt: find a font that natively supports the rupee sign (₹)
    for p in available:
        if font_has_glyph(p, "₹"):
            return p, True

    # Fallback: font found, but lacks rupee glyph
    if available:
        if warnings is not None:
            warnings.append(
                "Selected font lacks the rupee (₹) glyph; displaying 'Rs.' on screen."
            )
        return available[0], False

    # No system font found at all
    if warnings is not None:
        warnings.append("No system font found; using default font ('Rs.' on screen).")
    return None, False


def fit_font(
    font_path: Path | None,
    lines: list[str],
    max_width: int,
    base_size: int,
    min_size: int = 12,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Find the largest font size so all lines fit within max_width."""
    if not font_path or not font_path.exists():
        try:
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    size = base_size
    while size >= min_size:
        try:
            font = ImageFont.truetype(str(font_path), size)
            all_fit = True
            for line in lines:
                if not line:
                    continue
                bbox = font.getbbox(line)
                w = bbox[2] - bbox[0]
                if w > max_width:
                    all_fit = False
                    break
            if all_fit:
                return font
        except Exception:
            break
        size -= 2

    try:
        return ImageFont.truetype(str(font_path), min_size)
    except Exception:
        return ImageFont.load_default()


# ── String and slug helpers ─────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert text to a safe filename slug: 'Maple Residency' -> 'maple_residency'."""
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s or "property_video"


def prepare_caption_text(caption: str, supports_rupee: bool) -> list[str]:
    """Clean caption lines (max 2 lines) and replace rupee sign if unsupported."""
    lines = [l.strip() for l in caption.split("\n") if l.strip()][:2]
    if not supports_rupee:
        lines = [l.replace("₹", "Rs. ") for l in lines]
    return lines


# ── Image Preparation & Background Composition ──────────────────────────────

def create_base_composed_frame(
    image_path: Path,
    canvas_size: tuple[int, int],
    warnings: list[str] | None = None,
) -> Image.Image:
    """Create a high-resolution base frame with blurred cover background and sharp centered contain-fit foreground.

    If the image fails to load, generates a solid dark brand-colored placeholder canvas and records a warning.
    """
    W, H = canvas_size

    try:
        if not image_path.exists():
            raise FileNotFoundError(f"File not found: {image_path}")
        with Image.open(image_path) as src:
            src.load()
            img = src.convert("RGB")
    except Exception as exc:
        msg = f"Could not load image '{image_path}': {exc}. Using placeholder background."
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        # Solid dark brand-colored placeholder
        return Image.new("RGB", (W, H), (20, 28, 45))

    # 1. Background: cover-crop + blur + darken
    scale_bg = max(W / img.width, H / img.height)
    bg_w = max(1, int(round(img.width * scale_bg)))
    bg_h = max(1, int(round(img.height * scale_bg)))

    bg_resized = img.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
    left = (bg_w - W) // 2
    top = (bg_h - H) // 2
    bg_cropped = bg_resized.crop((left, top, left + W, top + H))

    # Gaussian blur & darken
    bg_blurred = bg_cropped.filter(ImageFilter.GaussianBlur(radius=28))
    bg_dark = ImageEnhance.Brightness(bg_blurred).enhance(0.40)

    # 2. Foreground: contain-fit without distortion
    scale_fg = min(W / img.width, H / img.height)
    fg_w = max(1, int(round(img.width * scale_fg)))
    fg_h = max(1, int(round(img.height * scale_fg)))

    fg_resized = img.resize((fg_w, fg_h), Image.Resampling.LANCZOS)
    paste_x = (W - fg_w) // 2
    paste_y = (H - fg_h) // 2

    canvas = bg_dark.copy()
    canvas.paste(fg_resized, (paste_x, paste_y))
    return canvas


# ── Text Overlay Rendering ─────────────────────────────────────────────────

def render_text_overlay(
    scene_type: str,
    lines: list[str],
    size: tuple[int, int],
    font_path: Path | None,
) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """Render semi-transparent dark rounded banner overlay for a scene.

    Returns:
        (overlay_image_rgba, bounding_box)
    """
    W, H = size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if not lines:
        return overlay, None

    draw = ImageDraw.Draw(overlay)
    safe_margin = int(W * 0.05)
    max_text_width = W - 2 * safe_margin

    # Determine layout and styling by scene type
    if scene_type == "title":
        # Hero title card: larger text, centered vertically in middle/lower half
        font_l1 = fit_font(font_path, [lines[0]], max_text_width - 80, base_size=int(H * 0.065), min_size=18)
        font_l2 = fit_font(font_path, lines[1:], max_text_width - 80, base_size=int(H * 0.040), min_size=14) if len(lines) > 1 else font_l1

        # Measure text
        bbox1 = font_l1.getbbox(lines[0])
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]

        w2, h2 = 0, 0
        if len(lines) > 1:
            bbox2 = font_l2.getbbox(lines[1])
            w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]

        content_w = max(w1, w2)
        line_gap = int(H * 0.015)
        content_h = h1 + (h2 + line_gap if len(lines) > 1 else 0)

        pad_x = int(W * 0.035)
        pad_y = int(H * 0.028)
        card_w = min(W - 2 * safe_margin, content_w + 2 * pad_x)
        card_h = content_h + 2 * pad_y

        card_x = (W - card_w) // 2
        card_y = int(H * 0.38)

        # Draw rounded card with dark backdrop and subtle border
        draw.rounded_rectangle(
            (card_x, card_y, card_x + card_w, card_y + card_h),
            radius=18,
            fill=(15, 23, 42, 220),
            outline=(255, 255, 255, 30),
            width=1,
        )

        # Line 1: White bold
        text1_x = (W - w1) // 2
        text1_y = card_y + pad_y - bbox1[1]
        draw.text((text1_x + 2, text1_y + 2), lines[0], font=font_l1, fill=(0, 0, 0, 160))
        draw.text((text1_x, text1_y), lines[0], font=font_l1, fill=(255, 255, 255, 255))

        # Line 2: Soft slate
        if len(lines) > 1:
            text2_x = (W - w2) // 2
            text2_y = text1_y + h1 + line_gap - bbox2[1] + bbox1[1]
            draw.text((text2_x + 2, text2_y + 2), lines[1], font=font_l2, fill=(0, 0, 0, 160))
            draw.text((text2_x, text2_y), lines[1], font=font_l2, fill=(226, 232, 240, 240))

    elif scene_type == "price_location":
        # Price (amber/gold accent) + Location (white) in lower third
        font_price = fit_font(font_path, [lines[0]], max_text_width - 60, base_size=int(H * 0.058), min_size=18)
        font_loc = fit_font(font_path, lines[1:], max_text_width - 60, base_size=int(H * 0.040), min_size=14) if len(lines) > 1 else font_price

        bbox1 = font_price.getbbox(lines[0])
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = 0, 0
        if len(lines) > 1:
            bbox2 = font_loc.getbbox(lines[1])
            w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]

        content_w = max(w1, w2)
        line_gap = int(H * 0.012)
        content_h = h1 + (h2 + line_gap if len(lines) > 1 else 0)

        pad_x = int(W * 0.028)
        pad_y = int(H * 0.022)
        card_w = min(W - 2 * safe_margin, content_w + 2 * pad_x)
        card_h = content_h + 2 * pad_y

        card_x = (W - card_w) // 2
        card_y = int(H * 0.70)

        draw.rounded_rectangle(
            (card_x, card_y, card_x + card_w, card_y + card_h),
            radius=16,
            fill=(15, 23, 42, 205),
            outline=(255, 255, 255, 25),
            width=1,
        )

        # Line 1: Accent amber price
        text1_x = (W - w1) // 2
        text1_y = card_y + pad_y - bbox1[1]
        draw.text((text1_x + 2, text1_y + 2), lines[0], font=font_price, fill=(0, 0, 0, 160))
        draw.text((text1_x, text1_y), lines[0], font=font_price, fill=(251, 191, 36, 255))

        # Line 2: White location
        if len(lines) > 1:
            text2_x = (W - w2) // 2
            text2_y = text1_y + h1 + line_gap - bbox2[1] + bbox1[1]
            draw.text((text2_x + 2, text2_y + 2), lines[1], font=font_loc, fill=(0, 0, 0, 160))
            draw.text((text2_x, text2_y), lines[1], font=font_loc, fill=(255, 255, 255, 255))

    elif scene_type == "cta":
        # CTA line highlighted (amber) + contact line larger (white)
        font_cta = fit_font(font_path, [lines[0]], max_text_width - 60, base_size=int(H * 0.046), min_size=16)
        font_contact = fit_font(font_path, lines[1:], max_text_width - 60, base_size=int(H * 0.054), min_size=18) if len(lines) > 1 else font_cta

        bbox1 = font_cta.getbbox(lines[0])
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = 0, 0
        if len(lines) > 1:
            bbox2 = font_contact.getbbox(lines[1])
            w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]

        content_w = max(w1, w2)
        line_gap = int(H * 0.012)
        content_h = h1 + (h2 + line_gap if len(lines) > 1 else 0)

        pad_x = int(W * 0.028)
        pad_y = int(H * 0.022)
        card_w = min(W - 2 * safe_margin, content_w + 2 * pad_x)
        card_h = content_h + 2 * pad_y

        card_x = (W - card_w) // 2
        card_y = int(H * 0.70)

        draw.rounded_rectangle(
            (card_x, card_y, card_x + card_w, card_y + card_h),
            radius=16,
            fill=(15, 23, 42, 210),
            outline=(255, 255, 255, 25),
            width=1,
        )

        # Line 1: CTA highlighted
        text1_x = (W - w1) // 2
        text1_y = card_y + pad_y - bbox1[1]
        draw.text((text1_x + 2, text1_y + 2), lines[0], font=font_cta, fill=(0, 0, 0, 160))
        draw.text((text1_x, text1_y), lines[0], font=font_cta, fill=(251, 191, 36, 255))

        # Line 2: Contact larger in crisp white
        if len(lines) > 1:
            text2_x = (W - w2) // 2
            text2_y = text1_y + h1 + line_gap - bbox2[1] + bbox1[1]
            draw.text((text2_x + 2, text2_y + 2), lines[1], font=font_contact, fill=(0, 0, 0, 160))
            draw.text((text2_x, text2_y), lines[1], font=font_contact, fill=(255, 255, 255, 255))

    else:
        # Standard feature scene: lower third banner with white text
        font = fit_font(font_path, lines, max_text_width - 60, base_size=int(H * 0.048), min_size=16)
        widths, heights = [], []
        for line in lines:
            b = font.getbbox(line)
            widths.append(b[2] - b[0])
            heights.append(b[3] - b[1])

        content_w = max(widths) if widths else 100
        line_gap = int(H * 0.012)
        content_h = sum(heights) + line_gap * max(0, len(lines) - 1)

        pad_x = int(W * 0.028)
        pad_y = int(H * 0.020)
        card_w = min(W - 2 * safe_margin, content_w + 2 * pad_x)
        card_h = content_h + 2 * pad_y

        card_x = (W - card_w) // 2
        card_y = int(H * 0.73)

        draw.rounded_rectangle(
            (card_x, card_y, card_x + card_w, card_y + card_h),
            radius=14,
            fill=(15, 23, 42, 200),
            outline=(255, 255, 255, 20),
            width=1,
        )

        curr_y = card_y + pad_y
        for i, line in enumerate(lines):
            lw, lh = widths[i], heights[i]
            lx = (W - lw) // 2
            b = font.getbbox(line)
            draw.text((lx + 2, curr_y - b[1] + 2), line, font=font, fill=(0, 0, 0, 160))
            draw.text((lx, curr_y - b[1]), line, font=font, fill=(255, 255, 255, 255))
            curr_y += lh + line_gap

    bbox = overlay.getbbox()
    return overlay, bbox


# ── Scene Clip Generator (Ken Burns Motion + Fade-in Text) ──────────────────

def make_scene_clip(
    scene: Scene,
    clip_duration: float,
    size: tuple[int, int],
    zoom_in: bool,
    font_path: Path | None,
    supports_rupee: bool,
    warnings: list[str],
) -> VideoClip:
    """Build an animated MoviePy VideoClip for one scene with slow Ken Burns zoom and text overlay."""
    W, H = size
    # Base frame is generated at 1.10x size to maintain high sharpness under 1.08 zoom
    W_base, H_base = int(round(W * 1.10)), int(round(H * 1.10))

    base_img = create_base_composed_frame(scene.image_path, (W_base, H_base), warnings)

    # Render static overlay and pre-extract alpha crop
    caption_lines = prepare_caption_text(scene.caption, supports_rupee)
    overlay_img, bbox = render_text_overlay(scene.scene_type, caption_lines, size, font_path)

    has_overlay = bbox is not None
    if has_overlay:
        ox1, oy1, ox2, oy2 = bbox
        sub_ov = overlay_img.crop(bbox)
        sub_ov_rgb = np.array(sub_ov.convert("RGB"), dtype=np.float32)
        sub_ov_alpha = np.array(sub_ov)[:, :, 3:4].astype(np.float32) / 255.0
    else:
        ox1 = oy1 = ox2 = oy2 = 0
        sub_ov_rgb = None
        sub_ov_alpha = None

    def make_frame(t: float) -> np.ndarray:
        p = min(1.0, max(0.0, t / clip_duration)) if clip_duration > 0 else 1.0
        # Smoothstep easing: 3p^2 - 2p^3
        ease = p * p * (3.0 - 2.0 * p)

        if zoom_in:
            s = 1.00 + 0.08 * ease
        else:
            s = 1.08 - 0.08 * ease

        crop_w = W_base / s
        crop_h = H_base / s
        x0 = (W_base - crop_w) / 2.0
        y0 = (H_base - crop_h) / 2.0
        x1 = x0 + crop_w
        y1 = y0 + crop_h

        frame_pil = base_img.crop((x0, y0, x1, y1)).resize((W, H), Image.Resampling.BILINEAR)
        frame_rgb = np.array(frame_pil)

        # Fade in text over 0.4s
        if has_overlay:
            t_fade = min(1.0, max(0.0, t / 0.4))
            if t_fade > 0.0:
                cur_a = sub_ov_alpha * t_fade
                sub_base = frame_rgb[oy1:oy2, ox1:ox2].astype(np.float32)
                blended = (sub_ov_rgb * cur_a + sub_base * (1.0 - cur_a)).astype(np.uint8)
                frame_rgb[oy1:oy2, ox1:ox2] = blended

        return frame_rgb

    return VideoClip(make_frame, duration=clip_duration)


# ── Main Video Builder ─────────────────────────────────────────────────────

def build_video(
    plan: ScenePlan,
    prop: PropertyData,
    out_path: Path | str,
    size: tuple[int, int] = (1280, 720),
    fps: int = 24,
    progress_cb: Callable[[int, int, str], None] | None = None,
    audio: Any = None,
) -> Path:
    """Build and export a complete MP4 video from a ScenePlan.

    Args:
        plan: Storyboard ScenePlan with validated scenes and durations.
        prop: Source property data for styling and context.
        out_path: Destination path for the generated MP4.
        size: Target (width, height) tuple, defaults to 720p (1280, 720).
        fps: Frames per second, defaults to 24.
        progress_cb: Optional callback func(scene_idx, total_scenes, msg).
        audio: Optional audio clip ready for Phase 4 (default: None).

    Returns:
        Path to the successfully exported MP4 file.
    """
    out_path = Path(out_path).resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guarantee even dimensions for libx264 yuv420p
    W = size[0] - size[0] % 2
    H = size[1] - size[1] % 2
    clean_size = (W, H)

    font_path, supports_rupee = get_best_font_path(plan.warnings)

    total_scenes = len(plan.scenes)
    if total_scenes == 0:
        raise ValueError("Cannot build video from an empty ScenePlan.")

    TRANSITION = 0.5
    clips: list[VideoClip] = []
    current_start = 0.0

    image_usage: dict[str, int] = {}

    try:
        for i, scene in enumerate(plan.scenes):
            if progress_cb:
                progress_cb(i + 1, total_scenes, f"Rendering scene {i+1}/{total_scenes}...")

            # Clip length adds overlap duration for crossfade (except final scene)
            clip_dur = scene.duration + (TRANSITION if i < total_scenes - 1 else 0.0)

            # Determine zoom direction: alternate by index, flip if image was reused
            img_key = str(scene.image_path)
            usage_count = image_usage.get(img_key, 0)
            image_usage[img_key] = usage_count + 1

            base_zoom_in = (i % 2 == 0)
            zoom_in = (not base_zoom_in) if (usage_count % 2 == 1) else base_zoom_in

            clip = make_scene_clip(
                scene=scene,
                clip_duration=clip_dur,
                size=clean_size,
                zoom_in=zoom_in,
                font_path=font_path,
                supports_rupee=supports_rupee,
                warnings=plan.warnings,
            )

            if i > 0:
                clip = clip.with_effects([vfx.CrossFadeIn(TRANSITION)])
            clip = clip.with_start(current_start)

            clips.append(clip)
            current_start += scene.duration

        # Assemble final composite video clip
        final_clip = CompositeVideoClip(clips, size=clean_size)
        # Fade-in from black at start (0.5s), fade-out to black at end (0.7s)
        final_clip = final_clip.with_effects([vfx.FadeIn(0.5), vfx.FadeOut(0.7)])

        # Optional audio slot for Phase 4
        if audio is not None:
            final_clip = final_clip.with_audio(audio)

        # Write to temporary file first for atomic export
        temp_out = out_dir / f".tmp_{os.getpid()}_{out_path.name}"
        if temp_out.exists():
            temp_out.unlink()

        try:
            final_clip.write_videofile(
                str(temp_out),
                fps=fps,
                codec="libx264",
                preset="medium",
                ffmpeg_params=["-movflags", "+faststart", "-crf", "23", "-pix_fmt", "yuv420p"],
                logger=None,
            )

            # Atomically replace destination
            if out_path.exists():
                out_path.unlink()
            temp_out.replace(out_path)

        except (PermissionError, OSError) as exc:
            msg = (
                f"Cannot write video to '{out_path}'. "
                f"The file may be open in a media player or locked: {exc}"
            )
            logger.error(msg)
            raise OSError(msg) from exc
        finally:
            if temp_out.exists():
                try:
                    temp_out.unlink()
                except Exception:
                    pass

    finally:
        # Clean up MoviePy clip resources
        try:
            if "final_clip" in locals():
                final_clip.close()
            for c in clips:
                c.close()
        except Exception:
            pass

    return out_path
