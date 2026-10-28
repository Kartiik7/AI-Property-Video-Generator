# AI Property Video Generator

A Python tool that takes property details and 4–8 images and exports a polished
20–40 second promotional MP4 with text overlays, transitions, and optional
narration/background music.

> **Phase 3 complete** — video rendering with text overlays, Ken Burns motion, and transitions.
> Audio/TTS narration will be attached in Phase 4.

---

## Generating the Video

Build a silent promotional MP4 directly from property data and images:

```bash
# House 1 (Maple Residency)
python main.py --data input/samples/house_1/property.json --images input/samples/house_1/images

# House 2 (Cedar Park Villa)
python main.py --data input/samples/house_2/property.json --images input/samples/house_2/images
```

### Video Options & Flags

- `--out <path>`: Destination MP4 file (defaults to `output/<slug>.mp4`, e.g. `output/maple_residency.mp4`).
- `--size 720p|1080p`: Video resolution — `720p` (1280×720, default) or `1080p` (1920×1080).
- `--fps <int>`: Render frame rate (default: `24`).
- `--no-video`: Generate and save the scene plan only without rendering MP4.

```bash
# Render in 1080p at 30 fps to custom output
python main.py --data input/samples/house_1/property.json --images input/samples/house_1/images --size 1080p --fps 30 --out output/high_res.mp4

# Skip video generation, inspect plan only
python main.py --data input/samples/house_1/property.json --images input/samples/house_1/images --no-video
```

## Usage

### 1 — Activate the virtual environment

```bash
# Windows (PowerShell)
cd property-video-generator
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 2 — Generate Script from a JSON File (Template Mode)

```bash
python main.py --data input/property.json
```

Custom script output path:

```bash
python main.py --data input/property.json --script-out output/my_script.json
```

Point to a different images directory:

```bash
python main.py --data input/property.json --images path/to/images
```

### 3 — LLM Mode (Auto-Enabled when API Key is Present)

Copy `.env.example` to `.env` and set your API key:

```bash
cp .env.example .env
```

```env
LLM_API_KEY=your_key_here
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
```

- **Automatic LLM Mode**: If `LLM_API_KEY` is set in `.env`, LLM mode is used automatically.
- **Force Template Mode**: Use `--no-llm` to force template generation even when an API key is configured.
- **Explicit Flag**: `--use-llm` is supported as an explicit alias.
- **Custom Script Priority**: Any `custom_script` provided in `property.json` always takes precedence for the scenes it covers.

```bash
# Automatic LLM mode (when LLM_API_KEY is in .env)
python main.py --data input/property.json

# Force template mode even if LLM_API_KEY is configured
python main.py --data input/property.json --no-llm
```

### 4 — Enter details interactively

```bash
python main.py --interactive
```

Features are entered one per line; press **Enter** on a blank line to finish.
Press **Ctrl+C** at any time to cancel cleanly (exits with code 130).

### 5 — Run tests

```bash
pytest tests/ -v
```

---

## How the Script is Generated

The generator converts property facts and images into a structured **ScenePlan**:

1. **Title Scene**: Introduces property name, type, and optional tagline on the first image.
2. **Feature Scenes**: Features are grouped into at most 4 scenes (merging extra features into the last feature scene). Each feature scene pairs with an image.
3. **Price & Location Scene**: Displays formatted price (`₹45,00,000`) and location; spoken voiceover converts prices to natural Indian words (e.g. `"45 lakh rupees"`, `"1 crore 20 lakh rupees"`).
4. **Call to Action (CTA) Scene**: Displays call to action text and contact info. Reuses the first image if images run out.

### AI / LLM Script Generation

When LLM mode is active (default when `LLM_API_KEY` is present):

- **What the AI does**: The AI acts as a professional real estate copywriter, authoring natural, conversational spoken voiceovers and punchy on-screen captions tailored to each scene type. It uses a low temperature (0.3) for high consistency.
- **What information it uses**: The prompt receives **strictly** the declared property facts (`name`, `location`, `price`, `property_type`, `features`, `cta_text`, optional `contact`, optional `tagline`) and the scene skeleton (`title`, `feature`, `price_location`, `cta`). The LLM is never given open-ended creative license to assume or extrapolate facts.
- **How its output is checked**:
  - **Per-Scene Constraints**: Each scene is vetted individually: captions must stay within 2 lines of max 38 characters each, voiceovers must not exceed 18 words, and the phone number is never spoken.
  - **Fact Checking**: Every generated line is verified by `src/fact_checker.py`. No foreign numbers, unmentioned amenities (e.g. unlisted gym, swimming pool, clubhouse), or prohibited promotional claims (`luxury`, `premium`, `best in class`, `guaranteed`, `RERA approved`, `near metro`) are permitted.
  - **Single Targeted Retry**: If any scenes fail validation or fact checks, the system issues **one** focused retry call providing the LLM with the exact error details (e.g., `"scene 3 contains the number 500 which is not in the input"`), requesting corrections for only the failing scenes.
- **Fallback Behavior**:
  - **Per-Scene Fallback**: Only scenes that still fail after retry are replaced with deterministic template text; all valid LLM scenes are preserved.
  - **Graceful Network & Format Recovery**: If the API call times out (20s limit), encounters HTTP errors or rate limits (`429`), or returns malformed JSON, the generator logs a clear warning and seamlessly falls back to template generation without crashing. The API key is never printed or logged.
  - **Transparent Source Tracking**: The plan records its exact origin — `"llm"` (all scenes from AI), `"template"` (deterministic fallback), or `"mixed (N of M scenes from LLM)"` — and saves this in `output/script.json`.

### Duration Budgeting
- Per-scene duration estimated from voiceover length: $\max(3.5, \text{words} / 2.5 + 0.8)$ seconds (title/CTA minimum 4.0s).
- Total duration is balanced evenly to guarantee the video stays strictly between **20 and 40 seconds**.
- No scene is shorter than 3.0 seconds.

### Strict Fact Checking
Every caption and voiceover (whether generated by template or LLM) must pass pure fact checks in `src/fact_checker.py`:
- **No foreign numbers**: Flags any numeric value or currency amount not present in the input.
- **No invented amenities**: Flags any unmentioned amenity keywords (e.g., gym, swimming pool, clubhouse, lift).
- **No banned claims**: Blocks promotional buzzwords (`luxury`, `premium`, `best in class`, `RERA approved`, `near metro`, `guaranteed`, `ROI`) unless they were explicitly present in the input property data.

---

## Property JSON format

```json
{
  "name": "Green Valley 2BHK",
  "location": "Baner, Pune",
  "price": 4500000,
  "property_type": "Apartment",
  "features": ["Pool", "Parking", "24/7 Security", "Modular Kitchen"],
  "cta_text": "Book a site visit today",
  "contact": "+91-00000-00000",
  "tagline": "Your dream home awaits"
}
```

| Field           | Required | Notes                                           |
|-----------------|----------|-------------------------------------------------|
| `name`          | ✓        | Max 120 chars                                   |
| `location`      | ✓        | Max 120 chars                                   |
| `price`         | ✓        | Number (auto-formatted Indian style) or string  |
| `property_type` | ✓        | e.g. Apartment, Villa, Plot                     |
| `features`      | ✓        | List; 3–6 recommended; max 80 chars each        |
| `cta_text`      | ✓        | Max 120 chars                                   |
| `contact`       |          | Optional; warning if absent                     |
| `tagline`       |          | Optional; max 120 chars                         |
| `custom_script` |          | Optional; object with handwritten scene scripts |

Place property images (`.jpg`, `.jpeg`, `.png`, `.webp`) in `input/images/`.
Recommended: 4–8 images at 640×360 or higher.

---

## Sample Data & Hand-Written Custom Scripts

Two sample properties with realistic photos and hand-written scripts are included in `input/samples/`:

- **House 1 (`input/samples/house_1/`)**: *Maple Residency* (Independent House in Sector 150, Noida)
- **House 2 (`input/samples/house_2/`)**: *Cedar Park Villa* (Villa in Sector 62, Noida)

### Running Sample Houses

```bash
# House 1
python main.py --data input/samples/house_1/property.json --images input/samples/house_1/images --script-out output/house_1_script.json

# House 2
python main.py --data input/samples/house_2/property.json --images input/samples/house_2/images --script-out output/house_2_script.json
```

### Optional `custom_script` Field

Properties can optionally specify exact scene copy via `custom_script`:

```json
{
  "custom_script": {
    "title": {
      "caption": "Maple Residency\nIndependent House",
      "voiceover": "Welcome to Maple Residency, an independent house in Sector 150, Noida."
    },
    "features": [
      {"caption": "Spacious kitchen", "voiceover": "Start your day in a spacious kitchen."},
      {"caption": "Comfortable bedroom", "voiceover": "Unwind in a comfortable bedroom."},
      {"caption": "Clean bathroom", "voiceover": "A clean bathroom completes the home."}
    ],
    "price_location": {
      "caption": "₹85,00,000\nSector 150, Noida",
      "voiceover": "Priced at 85 lakh rupees, in Sector 150, Noida."
    },
    "cta": {
      "caption": "Book a site visit today\n+91-00000-00000",
      "voiceover": "Book a site visit today. Contact details are on screen."
    }
  }
}
```

- **Fact-Checked**: Every custom caption and voiceover is strictly vetted against the property's declared facts. Formatted and spoken price forms (e.g. `₹85,00,000`, `85 lakh rupees`) are recognized.
- **Graceful Scene-by-Scene Fallback**: If any custom line fails the fact check or exceeds layout constraints (captions: max 2 lines × 38 characters; voiceovers: max 18 words), the system automatically falls back to template generation for that scene only and logs an informative warning.
- **Room-Aware Image Mapping**: Features automatically map to matching room images (e.g. kitchen feature to kitchen photo) when tagged photos (`_kitchen.jpg`, `_bedroom.jpg`, `_bathroom.jpg`, `_frontal.jpg`) are present.

