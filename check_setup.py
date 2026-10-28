"""
check_setup.py — Validates the development environment for the
AI Property Video Generator.

Run with:  python check_setup.py
"""

import sys
from pathlib import Path

# ── helpers ────────────────────────────────────────────────────────────────

PASS  = "  [OK]"
FAIL  = "  [FAIL]"
SEP   = "-" * 60

failures: list[str] = []   # collect every failure; report all at the end


def ok(msg: str) -> None:
    print(f"{PASS} {msg}")


def fail(msg: str) -> None:
    print(f"{FAIL} {msg}")
    failures.append(msg)


# ── 1. Python version ───────────────────────────────────────────────────────

print(SEP)
print("Checking Python version …")
ver = sys.version_info
print(f"       Python {sys.version}")

if (ver.major, ver.minor) >= (3, 10):
    ok(f"Python {ver.major}.{ver.minor} meets the minimum requirement (3.10+).")
else:
    fail(
        f"Python {ver.major}.{ver.minor} is too old. "
        "Please upgrade to Python 3.10 or newer."
    )

# ── 2. Required packages ────────────────────────────────────────────────────

print(SEP)
print("Checking required packages …")

MISSING_HINT = "Run:  pip install -r requirements.txt"

# moviepy
try:
    import moviepy
    ok(f"moviepy         {moviepy.__version__}")
except ImportError:
    fail(f"moviepy         MISSING — {MISSING_HINT}")

# Pillow / PIL
try:
    import PIL
    ok(f"Pillow          {PIL.__version__}")
except ImportError:
    fail(f"Pillow          MISSING — {MISSING_HINT}")

# edge_tts
try:
    import edge_tts
    ok(f"edge-tts        {edge_tts.__version__}")
except ImportError:
    fail(f"edge-tts        MISSING — {MISSING_HINT}")

# python-dotenv
try:
    import dotenv
    ok(f"python-dotenv   {dotenv.__version__}")
except ImportError:
    fail(f"python-dotenv   MISSING — {MISSING_HINT}")

# ── 3. FFmpeg via imageio-ffmpeg ─────────────────────────────────────────────

print(SEP)
print("Checking FFmpeg availability (via imageio-ffmpeg) …")

ffmpeg_path: str | None = None

try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    ok(f"FFmpeg found at: {ffmpeg_path}")
except ImportError:
    fail(f"imageio-ffmpeg  MISSING — {MISSING_HINT}")
except Exception as exc:
    fail(f"Could not locate FFmpeg binary: {exc}")

# ── 4. Create a short test video ─────────────────────────────────────────────

print(SEP)
print("Creating 2-second test video at output/_setup_test.mp4 …")

OUTPUT_DIR  = Path(__file__).parent / "output"
TEST_VIDEO  = OUTPUT_DIR / "_setup_test.mp4"

try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from moviepy import ColorClip, TextClip, CompositeVideoClip

    # Solid blue background, 2 seconds, 24 fps, 640×360
    bg = ColorClip(size=(640, 360), color=(30, 100, 200), duration=2)

    # "Setup OK" text centred on the frame
    txt = (
        TextClip(
            text="Setup OK",
            font_size=60,
            color="white",
            font="Arial",
            duration=2,
        )
        .with_position("center")
    )

    composite = CompositeVideoClip([bg, txt])
    composite.write_videofile(
        str(TEST_VIDEO),
        fps=24,
        logger=None,   # suppress moviepy progress bars
        codec="libx264",
        audio=False,
    )

    if TEST_VIDEO.exists() and TEST_VIDEO.stat().st_size > 0:
        size_kb = TEST_VIDEO.stat().st_size / 1024
        ok(f"Test video created successfully ({size_kb:.1f} KB).")
    else:
        fail("Test video file is empty or was not created.")

except ImportError as exc:
    fail(f"moviepy import error while building test video: {exc}")
except Exception as exc:
    fail(f"Test video creation failed: {exc}")

# ── 5. Final report ──────────────────────────────────────────────────────────

print(SEP)
if not failures:
    print("ALL CHECKS PASSED ✓")
else:
    print(f"CHECKS FAILED ({len(failures)} issue(s)):")
    for i, f_msg in enumerate(failures, 1):
        print(f"  {i}. {f_msg}")
    sys.exit(1)
