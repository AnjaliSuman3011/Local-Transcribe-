"""Run jobs: extract audio -> transcribe -> clean -> write outputs + manifest."""
import csv
import logging
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .audio import duration_seconds, extract_audio
from .cleanup import clean_segments
from .engines import Engine
from .models import Job, Transcript
from .writers import WRITERS

log = logging.getLogger(__name__)

MANIFEST_FIELDS = ["name", "source", "status", "language", "audio_minutes",
                   "seconds_taken", "speed_x_realtime", "error", "finished_at"]


class Manifest:
    """manifest.csv in the output folder: one row per input, rewritten after each job."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: Dict[str, Dict[str, str]] = {}
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    self.rows[r["name"]] = r

    def update(self, row: Dict[str, str]) -> None:
        self.rows[row["name"]] = row
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.rows.values())
        tmp.replace(self.path)


def run(jobs: List[Job], engine: Engine, out_dir: Path, formats: List[str],
        language: Optional[str] = None, overwrite: bool = False,
        keep_audio: bool = False) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(out_dir / "manifest.csv")
    counts = {"done": 0, "skipped": 0, "error": 0}
    total = len(jobs)

    for i, job in enumerate(jobs, 1):
        targets = {fmt: out_dir / f"{job.name}.{fmt}" for fmt in formats}
        if not overwrite and all(p.exists() for p in targets.values()):
            log.info("[%d/%d] %s: already done, skipping", i, total, job.name)
            counts["skipped"] += 1
            continue

        log.info("[%d/%d] %s: starting", i, total, job.name)
        row = {"name": job.name, "source": job.source}
        started = time.time()
        tmp = Path(tempfile.mkdtemp(prefix="transcribe_"))
        try:
            wav = extract_audio(job.source, tmp / f"{job.name}.wav", job.is_url)
            minutes = duration_seconds(wav) / 60
            log.info("[%d/%d] %s: audio ready (%.1f min), transcribing...", i, total, job.name, minutes)

            transcript = engine.transcribe(wav, language)
            transcript = Transcript(clean_segments(transcript.segments), transcript.language)
            if not transcript.text.strip():
                raise RuntimeError("no speech detected")

            for fmt, path in targets.items():
                WRITERS[fmt](transcript, path)
            if keep_audio:
                shutil.copy2(wav, out_dir / wav.name)

            took = time.time() - started
            row.update(status="done", language=transcript.language or "",
                       audio_minutes=f"{minutes:.1f}", seconds_taken=f"{took:.0f}",
                       speed_x_realtime=f"{(minutes * 60 / took):.1f}" if took else "")
            counts["done"] += 1
            log.info("[%d/%d] %s: done in %.0fs (%.1fx real-time)", i, total, job.name,
                     took, minutes * 60 / took if took else 0)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # one bad file must not stop the batch
            row.update(status="error", error=str(e)[:500],
                       seconds_taken=f"{time.time() - started:.0f}")
            counts["error"] += 1
            log.error("[%d/%d] %s: FAILED - %s", i, total, job.name, e)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            row["finished_at"] = datetime.now().isoformat(timespec="seconds")
            manifest.update(row)

    return counts
