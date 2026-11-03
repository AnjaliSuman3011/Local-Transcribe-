"""Turn CLI inputs (files, folders, URLs, list files) into Jobs."""
import csv
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .audio import INPUT_FORMATS, extension_of, is_supported
from .models import Job

log = logging.getLogger(__name__)


def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _safe_name(s: str) -> str:
    s = re.sub(r"[^\w.-]+", "_", s.strip()).strip("._")
    return s[:150] or "untitled"


def _stem(source: str) -> str:
    return Path(source.split("?", 1)[0]).stem


def _read_list(path: Path, url_column: str, id_column: Optional[str]) -> List[Tuple[str, Optional[str]]]:
    """Return (source, id) pairs from a .csv (with header) or a plain one-per-line file."""
    rows: List[Tuple[str, Optional[str]]] = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            if url_column not in cols:
                raise ValueError(f"Column '{url_column}' not in {path.name}. Found: {cols}")
            if id_column and id_column not in cols:
                raise ValueError(f"Column '{id_column}' not in {path.name}. Found: {cols}")
            for r in reader:
                src = (r.get(url_column) or "").strip()
                ident = (r.get(id_column) or "").strip() if id_column else ""
                if src:
                    rows.append((src, ident or None))
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append((line, None))
    return rows


def collect_jobs(inputs: Iterable[str], list_file: Optional[str] = None,
                 url_column: str = "url", id_column: Optional[str] = None) -> List[Job]:
    pairs: List[Tuple[str, Optional[str]]] = []
    for item in inputs:
        if _is_url(item):
            pairs.append((item, None))
            continue
        p = Path(item).expanduser()
        if p.is_dir():
            pairs += [(str(f), None) for f in sorted(p.rglob("*"))
                      if f.is_file() and f.suffix.lower() in INPUT_FORMATS]
        elif p.is_file():
            pairs.append((str(p), None))
        else:
            log.warning("Not found, skipping: %s", item)
    if list_file:
        pairs += _read_list(Path(list_file).expanduser(), url_column, id_column)

    jobs: List[Job] = []
    used = set()
    for src, ident in pairs:
        if not is_supported(src):
            log.warning("Unsupported format '%s', skipping: %s (supported: %s)",
                        extension_of(src) or "none", src, ", ".join(sorted(INPUT_FORMATS)))
            continue
        base = _safe_name(ident or _stem(src))
        name, n = base, 2
        while name in used:  # keep output names unique
            name, n = f"{base}-{n}", n + 1
        used.add(name)
        jobs.append(Job(source=src, name=name, is_url=_is_url(src)))
    return jobs
