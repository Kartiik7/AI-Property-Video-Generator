"""
verify_video.py — Media verification tool using FFmpeg probe.

Probes an MP4 and prints:
- Duration (seconds)
- Resolution (width x height)
- Video codec (e.g. h264)
- Pixel format (e.g. yuv420p)
- Audio stream presence
- File size (MB & bytes)

Exit code 1 if:
- Duration is outside 20–40s (or custom range)
- No audio stream exists (unless allowed)
- Video codec is not h264
- Pixel format is not yuv420p
"""

import sys
import re
import argparse
import subprocess
from pathlib import Path

# Force UTF-8 stdout for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def probe_video(video_path: Path | str) -> dict:
    """Probe video file using ffmpeg -i."""
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"File not found: {video_path}")

    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    proc = subprocess.run(
        [ffmpeg_exe, "-i", str(video_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # ffmpeg -i outputs metadata to stderr
    output = proc.stderr

    info: dict = {
        "path": str(video_path),
        "file_size_bytes": video_path.stat().st_size,
        "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
        "duration": None,
        "resolution": None,
        "video_codec": None,
        "pix_fmt": None,
        "has_audio": False,
        "audio_codec": None,
    }

    # 1. Parse Duration: 00:00:26.34
    dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
    if dur_match:
        hours = int(dur_match.group(1))
        mins = int(dur_match.group(2))
        secs = float(dur_match.group(3))
        info["duration"] = round(hours * 3600 + mins * 60 + secs, 2)

    # 2. Parse Video Stream: Stream #0:0... Video: h264 (High)... yuv420p... 1280x720
    vid_match = re.search(r"Stream #\d+:\d+.*?: Video: ([^,\s]+).*?,\s*([^,\s]+(?:\(.*?\))?).*?,\s*(\d{3,4}x\d{3,4})", output)
    if vid_match:
        info["video_codec"] = vid_match.group(1).lower()
        info["pix_fmt"] = vid_match.group(2).lower().split("(")[0].strip()
        info["resolution"] = vid_match.group(3)
    else:
        # Fallback individual regexes
        c_m = re.search(r"Video:\s*([a-zA-Z0-9_-]+)", output)
        if c_m:
            info["video_codec"] = c_m.group(1).lower()
        res_m = re.search(r",\s*(\d{3,4}x\d{3,4})", output)
        if res_m:
            info["resolution"] = res_m.group(1)
        if "yuv420p" in output.lower():
            info["pix_fmt"] = "yuv420p"

    # 3. Parse Audio Stream: Stream #0:1... Audio: aac...
    aud_match = re.search(r"Stream #\d+:\d+.*?: Audio: ([^,\s]+)", output)
    if aud_match:
        info["has_audio"] = True
        info["audio_codec"] = aud_match.group(1).lower()

    return info


def verify_video(
    video_path: Path | str,
    min_duration: float = 20.0,
    max_duration: float = 40.0,
    require_audio: bool = True,
    require_h264: bool = True,
    require_yuv420p: bool = True,
) -> tuple[bool, list[str], dict]:
    """Verify video satisfies all deliverable standards."""
    errors = []
    info = probe_video(video_path)

    # Duration check
    dur = info["duration"]
    if dur is None:
        errors.append("Could not determine video duration.")
    elif dur < min_duration or dur > max_duration:
        errors.append(f"Duration {dur:.1f}s is outside required {min_duration}–{max_duration}s range.")

    # Audio check
    if require_audio and not info["has_audio"]:
        errors.append("Final MP4 has no audio stream (narration or music required).")

    # Codec checks
    if require_h264 and (not info["video_codec"] or "h264" not in info["video_codec"]):
        errors.append(f"Video codec '{info['video_codec']}' is not H.264.")

    if require_yuv420p and (not info["pix_fmt"] or "yuv420p" not in info["pix_fmt"]):
        errors.append(f"Pixel format '{info['pix_fmt']}' is not yuv420p.")

    ok = len(errors) == 0
    return ok, errors, info


def main():
    parser = argparse.ArgumentParser(description="Verify MP4 video compliance.")
    parser.add_argument("video_path", help="Path to MP4 file to verify")
    parser.add_argument("--min-dur", type=float, default=20.0, help="Min allowed duration (default: 20.0)")
    parser.add_argument("--max-dur", type=float, default=40.0, help="Max allowed duration (default: 40.0)")
    parser.add_argument("--allow-no-audio", action="store_true", help="Allow video without audio track")

    args = parser.parse_args()
    vpath = Path(args.video_path)

    try:
        ok, errors, info = verify_video(
            video_path=vpath,
            min_duration=args.min_dur,
            max_duration=args.max_dur,
            require_audio=not args.allow_no_audio,
        )
    except Exception as exc:
        print(f"ERROR: Could not probe video '{vpath}': {exc}")
        sys.exit(1)

    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"  VIDEO VERIFICATION: {vpath.name}")
    print(SEP)
    print(f"  File Size    : {info['file_size_mb']} MB ({info['file_size_bytes']:,} bytes)")
    print(f"  Duration     : {info['duration']}s")
    print(f"  Resolution   : {info['resolution']}")
    print(f"  Video Codec  : {info['video_codec']}")
    print(f"  Pixel Format : {info['pix_fmt']}")
    print(f"  Audio Stream : {'Yes (' + str(info['audio_codec']) + ')' if info['has_audio'] else 'NO AUDIO'}")
    print(SEP)

    if ok:
        print("  ✓ ALL VERIFICATION CHECKS PASSED\n")
        sys.exit(0)
    else:
        print("  ❌ VERIFICATION FAILED:")
        for err in errors:
            print(f"    - {err}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
