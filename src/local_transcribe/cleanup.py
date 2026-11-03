"""Remove Whisper's known failure mode: runaway repetition loops.

On silence or music, Whisper can emit the same phrase hundreds of times
("you can do it with a hands and feet and you can do it with ..."). We
collapse those loops to a single occurrence. Normal speech is untouched,
but genuine 4x+ back-to-back repeats ("no no no no") are also shortened.
"""
import re
from typing import List

from .models import Segment

# A phrase of 1-15 words repeated back-to-back 4+ times
_REPEAT_RE = re.compile(r"((?:\S+\s+){1,15}?)\1{3,}", re.IGNORECASE)
_MAX_SAME_SEGMENTS = 2


def collapse_repetitions(text: str) -> str:
    if not text:
        return text
    # Trailing space lets the final repetition match too
    return _REPEAT_RE.sub(r"\1", text + " ").strip()


def clean_segments(segments: List[Segment]) -> List[Segment]:
    """Collapse loops inside segments and drop runs of identical segments."""
    out: List[Segment] = []
    run = 0
    for seg in segments:
        text = collapse_repetitions(seg.text.strip())
        if not text:
            continue
        if out and text.lower() == out[-1].text.lower():
            run += 1
            if run >= _MAX_SAME_SEGMENTS:
                out[-1].end = seg.end  # extend the kept segment instead
                continue
        else:
            run = 0
        out.append(Segment(seg.start, seg.end, text))
    return out
