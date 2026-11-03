"""Output formats. Add one by decorating a function with @writer("ext")."""
import json
from pathlib import Path
from typing import Callable, Dict

from .models import Transcript

Writer = Callable[[Transcript, Path], None]
WRITERS: Dict[str, Writer] = {}


def writer(ext: str) -> Callable[[Writer], Writer]:
    def deco(fn: Writer) -> Writer:
        WRITERS[ext] = fn
        return fn
    return deco


@writer("txt")
def write_txt(t: Transcript, path: Path) -> None:
    path.write_text(t.text + "\n", encoding="utf-8")


def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


@writer("srt")
def write_srt(t: Transcript, path: Path) -> None:
    blocks = [f"{i}\n{_ts(s.start)} --> {_ts(s.end)}\n{s.text}\n"
              for i, s in enumerate(t.segments, 1)]
    path.write_text("\n".join(blocks), encoding="utf-8")


@writer("json")
def write_json(t: Transcript, path: Path) -> None:
    data = {
        "language": t.language,
        "text": t.text,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in t.segments],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
