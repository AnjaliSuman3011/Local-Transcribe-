"""Plain data types shared across the package."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Segment:
    start: float  # seconds
    end: float    # seconds
    text: str


@dataclass
class Transcript:
    segments: List[Segment] = field(default_factory=list)
    language: Optional[str] = None

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.segments if s.text)


@dataclass
class Job:
    """One thing to transcribe."""
    source: str          # local path or http(s) URL
    name: str            # output file stem (unique within a run)
    is_url: bool = False
