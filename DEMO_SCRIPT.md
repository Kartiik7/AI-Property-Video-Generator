# Two-minute demo script

1. **Open the project and run the check.** Show `python main.py --check` and
   point out that optional LLM/TTS capabilities are reported without exposing keys.
2. **Run a sample.** Use the House 1 CLI command and show validation, scene-plan
   generation, audio mixing, and the final MP4 path.
3. **Show the scene plan.** Open the generated JSON and highlight captions,
   voiceovers, durations, image mapping, and the recorded source.
4. **Show the UI.** Launch `streamlit run app.py`, load a sample, and show the
   storyboard, narration/music toggles, and video preview.
5. **Demonstrate fact safety.** Enable LLM mode or show a deliberately invalid
   generated line; explain the fact checker, one targeted retry, and per-scene
   template fallback.
6. **Demonstrate graceful degradation.** Run `python make_samples.py` or open
   `samples_output/graceful_degradation_demo.txt` to show missing-image,
   corrupt-image, missing-field, and unavailable-audio messages without tracebacks.
7. **Finish with the deliverable.** Run `tools/verify_video.py` and play the
   MP4, pointing out duration, H.264/YUV420p output, narration, and music.

## Likely questions

- **Why this stack?** Python, MoviePy, Pillow, and bundled FFmpeg provide a
  portable, inspectable media pipeline with a small operational footprint.
- **How do you stop hallucinations?** The model sees only supplied facts;
  numbers, amenities, claims, phone narration, and lengths are checked before
  each scene is accepted.
- **Why not require an LLM?** Template-first behavior works offline, is
  reproducible, and avoids making a paid service a prerequisite.
- **What would you improve?** Add local TTS, subtitle export, more aspect ratios,
  stronger visual tests, and provider health/cost observability.
