"""
setup_checker.py — Environment and capability verification.

Reports core requirements and optional capabilities without crashing.
Used by check_setup.py and 'python main.py --check'.
"""

import os
import sys
import socket
from pathlib import Path
from importlib.metadata import version as pkg_version, PackageNotFoundError
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent.resolve()


def check_capabilities() -> dict[str, tuple[bool, str]]:
    """Inspect environment capabilities and return status dict."""
    load_dotenv()
    results: dict[str, tuple[bool, str]] = {}

    # 1. FFmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        results["FFmpeg"] = (True, f"Found at {ffmpeg_exe}")
    except Exception as exc:
        results["FFmpeg"] = (False, f"Not available: {exc}")

    # 2. edge-tts reachability (optional network check)
    try:
        # Fast DNS / socket check to speech.platform.bing.com:443
        s = socket.create_connection(("speech.platform.bing.com", 443), timeout=3.0)
        s.close()
        results["edge-tts"] = (True, "Online service reachable (narration enabled)")
    except Exception:
        results["edge-tts"] = (False, "Service unreachable or offline (graceful silent export active)")

    # 3. LLM API Key configured (Never print the key!)
    has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip())
    has_llm = bool(os.getenv("LLM_API_KEY", "").strip())
    if has_gemini or has_llm:
        provider = "Gemini" if has_gemini else "OpenAI-compatible"
        results["LLM API"] = (True, f"Key configured ({provider} AI script generation ready)")
    else:
        results["LLM API"] = (False, "No key configured in .env (deterministic template mode active)")

    # 4. Background music
    music_dir = BASE_DIR / "assets" / "music"
    music_files = []
    if music_dir.exists():
        music_files = [
            f for f in music_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".mp3", ".wav", ".m4a"}
        ]
    if music_files:
        results["Music"] = (True, f"Found '{music_files[0].name}' in assets/music/")
    else:
        results["Music"] = (False, "No audio file in assets/music/ (export without music)")

    # 5. Streamlit installed
    try:
        st_ver = pkg_version("streamlit")
        results["Streamlit"] = (True, f"Version {st_ver} installed")
    except PackageNotFoundError:
        results["Streamlit"] = (False, "Not installed (pip install streamlit)")

    return results


def print_capabilities_report() -> None:
    """Print formatted capability report to stdout."""
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  AI PROPERTY VIDEO GENERATOR — CAPABILITIES REPORT")
    print(SEP)

    results = check_capabilities()
    for cap, (is_ok, details) in results.items():
        status = "[OK]      " if is_ok else "[OPTIONAL]"
        print(f"  {status} {cap:<12} : {details}")

    print(SEP)
    print("  All core features operate offline with graceful degradation.\n")
