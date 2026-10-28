# AI Property Video Generator

A Python tool that takes property details and 4–8 images and exports a polished
20–40 second promotional MP4 with text overlays, transitions, and optional
narration/background music.

> **Phase 1 complete** — input handling and validation.
> Video generation will be added in Phase 3.

---

## Usage (Phase 1)

### 1 — Activate the virtual environment

```bash
# Windows (PowerShell)
cd property-video-generator
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 2 — Validate from a JSON file

```bash
python main.py --data input/property.json
```

Point to a different images directory:

```bash
python main.py --data input/property.json --images path/to/images
```

### 3 — Enter details interactively

```bash
python main.py --interactive
```

Features are entered one per line; press **Enter** on a blank line to finish.
Press **Ctrl+C** at any time to cancel cleanly.

### 4 — Run the environment check

```bash
python check_setup.py
```

### 5 — Run tests

```bash
pytest tests/ -v
```

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

Place property images (`.jpg`, `.jpeg`, `.png`, `.webp`) in `input/images/`.
Recommended: 4–8 images at 640×360 or higher.

---

_Phases will be documented here as they are completed._
