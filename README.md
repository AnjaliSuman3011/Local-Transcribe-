# local-transcribe

Transcribe video and audio recordings **on your own machine** with open-source
[Whisper](https://github.com/openai/whisper). There are no API keys and no per-minute cost, and
the audio never leaves your laptop (URL inputs are downloaded straight to your machine).

```
recording.mp4 ──ffmpeg──▶ 16 kHz mono audio ──Whisper (local)──▶ cleanup ──▶ recording.txt
```

- **Input:** `.mp4` (primary), plus `.mov .mkv .webm .mp3 .wav .m4a`. Inputs can be files, folders, URLs, or a CSV/TXT list.
- **Output:** `.txt` by default; `.srt` (subtitles with timestamps) and `.json` (segments) on request.
- **Batch-friendly:** if a file fails, the tool logs it and moves on. Re-running the same command skips finished files, and a `manifest.csv` records the status of every file.

---

## 1. Setup: which one do I use?

| Your computer | Use | Why |
|---|---|---|
| **Mac (Apple Silicon, M1-M4)** | [Native setup](#native-mac-apple-silicon) | Faster: uses the Mac GPU via MLX |
| **Windows / non-Mac** | [Docker setup](#docker-windows) | CPU-only and slower, but no manual Python/ffmpeg setup |

### NATIVE (Mac, Apple Silicon)

You need **Python 3.9+** and **ffmpeg**.

```bash
brew install ffmpeg
git clone https://github.com/AnjaliSuman3011/Local-Transcribe-.git local-transcribe
cd local-transcribe
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[mlx]'
transcribe --check          # expect "ffmpeg: OK" and "default: mlx"
```

Run it:

```bash
transcribe call.mp4 -m small -l en          # clear English audio
transcribe call.mp4 -m large-v3             # heavy Hindi / mixed-language audio
```

The first run of each model downloads it (`small` ~0.5 GB, `large-v3` ~3 GB) and caches it.
Plug in the charger and turn off Low Power Mode, or it runs about 12x slower.
Other platforms (Intel Mac, Linux) can use `pip install -e '.[faster]'` instead.

### DOCKER (Windows)

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/) only. No Python or ffmpeg.

> **CPU only, so slower than the Mac setup.** The container runs `faster-whisper` on CPU
> (MLX cannot use the GPU inside Docker). **Windows users should prefer the `small` or `base`
> model.** Give Docker Desktop 8 GB+ of memory.

```bash
git clone https://github.com/AnjaliSuman3011/Local-Transcribe-.git local-transcribe
cd local-transcribe
docker compose build
```

Local file (put it in the `recordings` folder; transcripts appear in `transcripts`):

```bash
mkdir recordings
copy C:\path\to\call.mp4 recordings\
docker compose run --rm transcribe /recordings/call.mp4 -m small -l en
```

URL (no input folder needed):

```bash
docker compose run --rm transcribe https://host/path/rec.mp4 -m small -l en
```

Models are cached in a Docker volume, so each downloads only once. For heavy Hindi / mixed
audio, use `-m large-v3` (much slower on CPU).

## 2. Run it

```bash
transcribe call.mp4                      # -> transcripts/call.txt
transcribe ~/Recordings/                 # every supported file in the folder (recursive)
transcribe https://host/path/rec.mp4     # a URL
transcribe call.mp4 -f txt,srt           # also write subtitles
transcribe call.mp4 -l en                # force English (default: auto-detect per file)
transcribe --help                        # all options
```

### From a Google Sheet (batch of recording links)

1. In the sheet: **File → Download → Comma-separated values (.csv)** for the tab with the links.
2. Run it, telling the tool which column holds the link and which column to name files by:

```bash
transcribe --list "DE Session - Audit - Mapped to DE.csv" \
  --url-column recording_link --id-column prospect_id -l en
```

Each transcript is saved as `transcripts/<prospect_id>.txt`, so you can match it back to its row.
`transcripts/manifest.csv` lists every file with its status (`done` / `error`), audio length and
processing time.

**Interrupted or a few failed?** Run the exact same command again. Finished files are skipped,
and only the missing or failed ones are processed.

## 3. Choosing a model

| Model | Speed | Accuracy | Use when |
|---|---|---|---|
| `large-v3-turbo` (default) | fast | very good for English | most calls |
| `large-v3` | several times slower | best, especially non-English | heavy Hindi / mixed-language audio |
| `small`, `base`, `tiny` | fastest | noticeably worse | quick tests only |

```bash
transcribe call.mp4 -m large-v3
```

### Which model should I use? (first-time users start here)

Measured on a 3-minute English sales call, Apple Silicon MacBook (16 GB), plugged in:

| Model (`-m` value) | Download | Time | Speed | WER* (lower = better) |
|---|---|---|---|---|
| `mlx-community/whisper-small-mlx-q4` | 0.2 GB | 9 s | 21x real-time | 10.9% |
| `small` | 0.5 GB | 11 s | 16x | 11.4% |
| `medium` | 1.5 GB | 21 s | 8.5x | 12.1% |
| `large-v3` | 3 GB | 38 s | 5x | 13.5% |
| `large-v3-turbo` (default) | 1.6 GB | 24 s | 7x | 13.7% |
| `base` | 0.15 GB | 5 s | 39x | 15.4% |

\*Word error rate against a reference transcript from another tool (not a human transcript),
on one 3-minute English call (422 words). A 1-2 point gap is only a few words, so treat the top
four as roughly tied. Not recommended: `distil-whisper-large-v3`, `whisper-medium.en-mlx` and
`whisper-large-v3-turbo-q4` (all 15-17% WER, English-only or no faster).

- **Clear, mostly English calls:** start with `small` (fast, small download, best score here).
  `base` is fastest if you can accept a few more errors.
  `transcribe call.mp4 -m small -l en`
- **Heavy Hindi or mixed-language (Hinglish) audio:** use `large-v3`. The smaller models make many more mistakes on it.
  **Use this command:** `transcribe call.mp4 -m large-v3`
  For a CSV batch: `transcribe --list file.csv --url-column recording_link --id-column prospect_id -m large-v3`
- Not sure? Leave the default (`large-v3-turbo`), which is what runs when you don't pass `-m`.
- The `large-v3` recommendation for Hindi / mixed audio is untested on our sample (it was English only). Spot-check the first results.

> **Plug in the charger and turn off Low Power Mode** (System Settings → Battery). On battery
> with Low Power Mode on, we measured the same model running about 12x slower.
> The first run of each model also downloads it (`small` ~0.5 GB, `large-v3` ~3 GB), so don't use that run to judge speed.

### Optional: faster-whisper engine (not benchmarked)

We did not benchmark this engine. It is the right choice on **Intel Mac, Linux or Windows**, and it
is available on a Mac too if you want to try it:

```bash
source .venv/bin/activate
pip install -e '.[faster]'          # installs it next to (or instead of) MLX
transcribe --check                  # should now list "faster-whisper  installed"
transcribe call.mp4 --engine faster-whisper -m small -l en
```

Notes: it runs on CPU unless an NVIDIA GPU is present, so it is much slower than MLX on Apple
Silicon. Model names are the standard Whisper names (`small`, `large-v3`, ...) and are downloaded
on first use. The first-run download rules and the Low Power Mode warning above still apply.

**Speed depends on your machine.** Run one file first and check `speed_x_realtime` in
`manifest.csv`. For example, 10.0 means one hour of audio takes about 6 minutes. Multiply that
out to plan a big batch, and keep the laptop plugged in and awake (`caffeinate -i transcribe ...`
on Mac).

## 4. Good to know

- **Repetition cleanup.** On silence, Whisper sometimes loops a phrase ("thank you thank you
  ..."). The tool collapses any phrase repeated 4+ times back-to-back into a single copy, and
  drops runs of identical segments. Real repeats like "no no no no" are shortened too.
- **No speaker labels.** The transcript is one stream of text and doesn't say who is speaking.
- **Hindi / Hinglish:** auto-detect handles it, but accuracy is lower than for English. Use
  `-m large-v3` and spot-check the results.
- **Privacy:** transcripts often contain personal data (names, salaries, phone numbers).
  `transcripts/` and media files are git-ignored. Don't commit them.

## 5. Extending

The tool is built around three small registries, so adding support means writing one function or class:

**New input format.** If ffmpeg can read it, add the extension in `src/local_transcribe/audio.py`:
```python
INPUT_FORMATS[".flac"] = "audio"
```
For something ffmpeg can't read directly, register a custom extractor that writes a 16 kHz mono WAV:
```python
from local_transcribe.audio import register_extractor
register_extractor(".xyz", lambda source, out_wav, is_url: ...)
```

**New output format.** Add a writer in `src/local_transcribe/writers.py`:
```python
@writer("md")
def write_md(t: Transcript, path: Path) -> None:
    path.write_text("\n\n".join(s.text for s in t.segments))
```

**New engine** (e.g. whisper.cpp or a cloud API). Subclass `Engine` in `src/local_transcribe/engines.py`:
```python
@register("my-engine")
class MyEngine(Engine):
    @classmethod
    def is_installed(cls): ...
    def transcribe(self, wav, language) -> Transcript: ...
```
It then appears in `transcribe --check` and `--engine`.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg not found` | `brew install ffmpeg`, then open a new terminal |
| `No transcription engine installed` | Activate the venv (`source .venv/bin/activate`) and install an engine (step 1) |
| `Could not load model` | The first run needs internet to download the model; check your network or VPN |
| A file shows `error` in the manifest | Read the `error` column. An expired or private link is the usual cause. Fix it and re-run |
| Very slow on a Mac | Check `transcribe --check` says `default: mlx`. faster-whisper on CPU is much slower |

## Development

```bash
pip install -e '.[dev]'
pytest -q        # uses a stub engine, so no model download is needed
```

Layout: `cli.py` (arguments) → `sources.py` (inputs → jobs) → `pipeline.py` (per job: `audio.py`
extract → `engines.py` transcribe → `cleanup.py` → `writers.py`, then update the manifest).
