# AI, libraries, and external services

Only synthetic/demo property data is sent to external providers. Do not place
secrets in source files, scene JSON, logs, or deliverables. Free-tier providers
may use prompts and outputs to improve their products; review each provider's
current terms before using real client data.

| Service/library | Purpose | Key/internet required | License/terms note | Fallback |
|---|---|---|---|---|
| Gemini LLM (`LLM_BASE_URL`, `LLM_MODEL`, `GEMINI_API_KEY`) | Optional copy generation | Key and internet | Google API terms; model availability varies | OpenAI-compatible fallback, then templates |
| OpenRouter or another OpenAI-compatible provider (`LLM_FALLBACK_*`) | Optional LLM fallback | Key and internet | Provider/model terms apply | Deterministic templates |
| Gemini TTS (`TTS_GEMINI_MODEL`, `TTS_GEMINI_VOICE`) | Optional narration | Gemini key and internet | Google API terms; voice/model availability varies | edge-tts |
| edge-tts | Narration fallback | Internet generally required; no API key | Microsoft service terms apply | Continue without narration |
| MoviePy | Video/audio composition | No key | MIT | None; pipeline reports errors |
| Pillow | Image transforms and text overlays | No key | HPND-style Pillow license | None |
| imageio-ffmpeg | Bundled FFmpeg executable | No key; local binary | FFmpeg is LGPL/GPL depending on build options | System FFmpeg if configured |
| Streamlit | Optional web UI | No key; local server | Apache-2.0 | CLI |
| pytest | Test runner | No key | MIT | None |
| python-dotenv | Loads local `.env` configuration | No key | BSD-3-Clause | Environment variables |

## Credits

The bundled images are from the [Houses Dataset](https://github.com/emanhamed/houses-dataset).
Its README cites Ahmed and Moustafa, *House Price Estimation from Visual and
Textual Features*, IJCCI 2016, DOI
[10.5220/0006040700620068](https://doi.org/10.5220/0006040700620068).

Background music attribution is recorded in
[`assets/music/CREDITS.txt`](./assets/music/CREDITS.txt): **Music Promotion -
Miromaxmusic - Pixabay - Pixabay Content License**.
