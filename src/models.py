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
