"""Command-line entry point: `transcribe --help`."""
import argparse
import logging
import platform
import sys
from pathlib import Path

from . import __version__
from .audio import INPUT_FORMATS, require_ffmpeg
from .engines import DEFAULT_MODEL, available_engines, pick_engine
from .pipeline import run
from .sources import collect_jobs
from .writers import WRITERS

EXAMPLES = """
examples:
  transcribe call.mp4                          # one file -> transcripts/call.txt
  transcribe ~/Recordings/                     # every supported file in a folder
  transcribe https://example.com/rec.mp4       # a URL (streamed, not saved)
  transcribe --list urls.txt                   # one path/URL per line
  transcribe --list sheet.csv --url-column recording_link --id-column prospect_id
  transcribe --check                           # verify ffmpeg + engine setup
"""


def _check() -> int:
    ok = True
    try:
        require_ffmpeg()
        print("ffmpeg:  OK")
    except RuntimeError as e:
        print(f"ffmpeg:  MISSING - {e}")
        ok = False
    print(f"system:  {platform.system()} {platform.machine()}, Python {platform.python_version()}")
    for name, cls in available_engines().items():
        print(f"engine:  {name:15} {'installed' if cls.is_installed() else 'not installed'}")
    try:
        print(f"default: {pick_engine('auto').name}")
    except RuntimeError as e:
        print(f"default: NONE - {e}")
        ok = False
    print("formats: in  =", ", ".join(sorted(INPUT_FORMATS)))
    print("         out =", ", ".join(WRITERS))
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe video/audio locally with open-source Whisper. Free; audio never leaves your machine.",
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", nargs="*", help="files, folders or http(s) URLs")
    p.add_argument("--list", dest="list_file", help="a .txt (one per line) or .csv file of paths/URLs")
    p.add_argument("--url-column", default="url", help="CSV column holding the path/URL (default: url)")
    p.add_argument("--id-column", help="CSV column used to name outputs, e.g. prospect_id")
    p.add_argument("-o", "--out", default="transcripts", help="output folder (default: ./transcripts)")
    p.add_argument("-f", "--formats", default="txt",
                   help=f"comma-separated output formats: {', '.join(WRITERS)} (default: txt)")
    p.add_argument("-e", "--engine", default="auto", choices=["auto", *available_engines()],
                   help="auto = mlx on Apple Silicon, else faster-whisper")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL,
                   help=f"Whisper model: tiny, base, small, medium, large-v3, large-v3-turbo (default: {DEFAULT_MODEL})")
    p.add_argument("-l", "--language", default=None,
                   help="language code, e.g. en or hi (default: auto-detect per file)")
    p.add_argument("--overwrite", action="store_true", help="redo files that already have outputs")
    p.add_argument("--keep-audio", action="store_true", help="also save the extracted .wav")
    p.add_argument("--check", action="store_true", help="check setup and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("transcribe")

    if args.check:
        return _check()
    if not args.inputs and not args.list_file:
        p.print_help()
        return 2

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    bad = [f for f in formats if f not in WRITERS]
    if bad:
        p.error(f"unknown format(s): {', '.join(bad)}. Options: {', '.join(WRITERS)}")

    try:
        require_ffmpeg()
        jobs = collect_jobs(args.inputs, args.list_file, args.url_column, args.id_column)
        if not jobs:
            log.error("Nothing to transcribe.")
            return 1
        engine = pick_engine(args.engine, args.model)
    except (RuntimeError, ValueError) as e:
        log.error(str(e))
        return 1

    log.info("Loading model %s (first run downloads it, ~1.5 GB for large-v3-turbo)...", args.model)
    try:
        engine.load()
    except Exception as e:
        log.error("Could not load model '%s': %s", args.model, e)
        return 1

    log.info("%d file(s) | engine=%s model=%s language=%s | output -> %s",
             len(jobs), engine.name, args.model, args.language or "auto", Path(args.out).resolve())
    try:
        counts = run(jobs, engine, Path(args.out), formats, args.language,
                     args.overwrite, args.keep_audio)
    except KeyboardInterrupt:
        log.warning("Interrupted. Re-run the same command to resume; finished files are skipped.")
        return 130
    log.info("Finished: %d done, %d skipped, %d failed. See %s",
             counts["done"], counts["skipped"], counts["error"], Path(args.out) / "manifest.csv")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
