"""Input formats and audio extraction.

Every input goes through ffmpeg, which converts it to the 16 kHz mono WAV that
Whisper expects. To support a new input format, add its extension to
INPUT_FORMATS. If a format needs special handling (e.g. a container ffmpeg
can't read directly), register a custom extractor with `register_extractor`.
"""
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict

# extension -> kind. mp4 is the primary, production-tested format.
INPUT_FORMATS: Dict[str, str] = {
    ".mp4": "video",
    ".mov": "video",
    ".mkv": "video",
    ".webm": "video",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
}

Extractor = Callable[[str, Path, bool], None]
_EXTRACTORS: Dict[str, Extractor] = {}


def register_extractor(extension: str, fn: Extractor, kind: str = "custom") -> None:
    """Plug in custom extraction for an extension: fn(source, out_wav, is_url)."""
    ext = extension.lower() if extension.startswith(".") else "." + extension.lower()
    INPUT_FORMATS[ext] = kind
    _EXTRACTORS[ext] = fn


def is_supported(path_or_url: str) -> bool:
    return extension_of(path_or_url) in INPUT_FORMATS


def extension_of(path_or_url: str) -> str:
    # Strip URL query strings (e.g. signed S3 links) before reading the extension
    return Path(path_or_url.split("?", 1)[0]).suffix.lower()


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(
                f"{tool} not found on PATH. Install it: `brew install ffmpeg` (Mac) "
                "or `sudo apt install ffmpeg` (Linux)."
            )


def _ffmpeg_extract(source: str, out_wav: Path, is_url: bool) -> None:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if is_url:
        # Survive flaky connections on long downloads
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"]
    cmd += ["-i", source, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size == 0:
        msg = (proc.stderr or "").strip().splitlines()
        raise RuntimeError("ffmpeg failed: " + (msg[-1] if msg else "no audio produced"))


def extract_audio(source: str, out_wav: Path, is_url: bool = False) -> Path:
    """Convert any supported input to 16 kHz mono WAV at out_wav."""
    fn = _EXTRACTORS.get(extension_of(source), _ffmpeg_extract)
    fn(source, out_wav, is_url)
    return out_wav


def duration_seconds(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0
