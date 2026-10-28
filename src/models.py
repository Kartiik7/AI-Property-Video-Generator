"""
models.py — Core data structures for the AI Property Video Generator.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PropertyData:
    """All property details needed to generate the video."""
    name: str
    location: str
    price: str           # always stored as a display string
    property_type: str
    features: list[str]
    cta_text: str
    contact: str | None = None   # optional; warning if absent
    tagline: str | None = None   # optional; short headline


@dataclass
class LoadResult:
    """Outcome of a load attempt: property data, images, and any issues."""
    property: PropertyData | None
    images: list[Path]
    warnings: list[str]
    errors: list[str]

    @property
    def ok(self) -> bool:
        """True when there are no errors (warnings are non-blocking)."""
        return len(self.errors) == 0


@dataclass
class Scene:
    """A single video scene: visual image, text caption, voiceover, and duration."""
    scene_type: str       # "title" | "feature" | "price_location" | "cta"
    image_path: Path
    caption: str
    voiceover: str
    duration: float       # seconds (>= 3.0)


@dataclass
class ScenePlan:
    """Full storyboard plan composed of multiple sequential scenes."""
    scenes: list[Scene]
    source: str           # "template" | "llm"
    warnings: list[str] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        """Total duration of all scenes in seconds."""
        return round(sum(s.duration for s in self.scenes), 2)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "source": self.source,
            "total_duration": self.total_duration,
            "warnings": self.warnings,
            "scenes": [
                {
                    "scene_type": s.scene_type,
                    "image_path": str(s.image_path),
                    "caption": s.caption,
                    "voiceover": s.voiceover,
                    "duration": round(s.duration, 2),
                }
                for s in self.scenes
            ],
        }
