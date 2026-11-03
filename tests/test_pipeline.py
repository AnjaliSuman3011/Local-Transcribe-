"""Tests run without downloading any Whisper model: a stub engine stands in."""
import csv
import subprocess
from pathlib import Path

import pytest

from local_transcribe import cli
from local_transcribe.audio import duration_seconds, extract_audio
from local_transcribe.cleanup import clean_segments, collapse_repetitions
from local_transcribe.engines import Engine, register
from local_transcribe.models import Segment, Transcript
from local_transcribe.sources import collect_jobs


@register("stub")
class StubEngine(Engine):
    calls = 0

    @classmethod
    def is_installed(cls):
        return True

    def transcribe(self, wav, language):
        StubEngine.calls += 1
        assert wav.suffix == ".wav" and wav.stat().st_size > 0
        loop = [Segment(5 + i, 6 + i, "thank you") for i in range(10)]
        return Transcript([Segment(0, 2, "hello there"), *loop,
                           Segment(20, 21, "so so so so so we start")], "en")


def _make_media(path: Path, seconds: int = 3) -> Path:
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    if path.suffix in (".mp4", ".mov", ".mkv", ".webm"):
        cmd += ["-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=160x120:rate=10", "-shortest"]
    subprocess.run(cmd + [str(path)], check=True)
    return path


@pytest.mark.parametrize("ext", [".mp4", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".m4a"])
def test_extract_audio_all_formats(tmp_path, ext):
    src = _make_media(tmp_path / f"in{ext}")
    wav = extract_audio(str(src), tmp_path / "out.wav")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels",
                            "-of", "csv=p=0", str(wav)], capture_output=True, text=True).stdout.strip()
    assert probe == "16000,1"
    assert 2.5 < duration_seconds(wav) < 3.5


def test_collapse_repetitions():
    loop = "intro. " + "you can do it with a hands and feet and " * 300 + "bye"
    assert collapse_repetitions(loop) == "intro. you can do it with a hands and feet and bye"
    normal = "We do it well and it is fine. Yes yes it is."
    assert collapse_repetitions(normal) == normal


def test_clean_segments_drops_repeated_segments():
    segs = [Segment(i, i + 1, "thank you") for i in range(10)]
    out = clean_segments([Segment(0, 1, "hi")] + segs)
    assert [s.text for s in out] == ["hi", "thank you", "thank you"]
    assert out[-1].end == 10


def test_collect_jobs_csv_ids_and_filtering(tmp_path):
    lst = tmp_path / "sheet.csv"
    lst.write_text("recording_link,prospect_id\n"
                   "https://x.com/a/Rec-1.mp4,abc-1\n"
                   "https://x.com/a/Rec-2.mp4?X-Amz-Signature=z,abc-1\n"
                   "https://x.com/a/notes.pdf,abc-3\n"
                   ",abc-4\n")
    jobs = collect_jobs([], str(lst), "recording_link", "prospect_id")
    assert [j.name for j in jobs] == ["abc-1", "abc-1-2"]  # pdf + blank skipped, dup id made unique
    assert all(j.is_url for j in jobs)


def test_cli_end_to_end_and_resume(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    _make_media(media / "call one.mp4")
    _make_media(media / "voice.mp3")
    out = tmp_path / "out"
    StubEngine.calls = 0

    rc = cli.main([str(media), "-o", str(out), "-e", "stub", "-f", "txt,srt,json"])
    assert rc == 0 and StubEngine.calls == 2
    txt = (out / "call_one.txt").read_text()
    assert txt.splitlines() == ["hello there", "thank you", "thank you", "so we start"]
    assert (out / "call_one.srt").read_text().startswith("1\n00:00:00,000 --> 00:00:02,000\nhello there")
    rows = list(csv.DictReader((out / "manifest.csv").open()))
    assert {r["status"] for r in rows} == {"done"} and len(rows) == 2

    rc = cli.main([str(media), "-o", str(out), "-e", "stub", "-f", "txt,srt,json"])
    assert rc == 0 and StubEngine.calls == 2  # resumed: nothing re-transcribed


def test_cli_bad_file_does_not_stop_batch(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "broken.mp4").write_bytes(b"not a video")
    _make_media(media / "good.mp4")
    out = tmp_path / "out"
    rc = cli.main([str(media), "-o", str(out), "-e", "stub"])
    assert rc == 1
    rows = {r["name"]: r for r in csv.DictReader((out / "manifest.csv").open())}
    assert rows["good"]["status"] == "done"
    assert rows["broken"]["status"] == "error" and "ffmpeg failed" in rows["broken"]["error"]
