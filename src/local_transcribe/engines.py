"""Speech-to-text backends.

Two open-source Whisper backends ship by default:
  - "mlx":            mlx-whisper, runs on the Apple Silicon GPU (fastest on M-series Macs)
  - "faster-whisper": CTranslate2, runs on CPU anywhere, or NVIDIA GPU if available

To add another backend (e.g. whisper.cpp, a cloud API), subclass Engine and
decorate it with @register("name").
"""
import platform
from pathlib import Path
from typing import Callable, Dict, Optional, Type

from .models import Segment, Transcript

DEFAULT_MODEL = "large-v3-turbo"

_REGISTRY: Dict[str, Type["Engine"]] = {}


def register(name: str) -> Callable[[Type["Engine"]], Type["Engine"]]:
    def deco(cls: Type["Engine"]) -> Type["Engine"]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def available_engines() -> Dict[str, Type["Engine"]]:
    return dict(_REGISTRY)


class Engine:
    name = "base"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    @classmethod
    def is_installed(cls) -> bool:
        return False

    def load(self) -> None:
        """Load/download the model up front so setup problems fail fast."""

    def transcribe(self, wav: Path, language: Optional[str]) -> Transcript:
        raise NotImplementedError


@register("mlx")
class MLXEngine(Engine):
    # Friendly names -> Hugging Face repos with MLX weights
    MODELS = {
        "tiny": "mlx-community/whisper-tiny",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        "turbo": "mlx-community/whisper-large-v3-turbo",
    }

    @classmethod
    def is_installed(cls) -> bool:
        try:
            import mlx_whisper  # noqa: F401
            return True
        except Exception:
            return False

    def transcribe(self, wav: Path, language: Optional[str]) -> Transcript:
        import mlx_whisper

        repo = self.MODELS.get(self.model, self.model)  # "org/repo" passes through
        result = mlx_whisper.transcribe(
            str(wav),
            path_or_hf_repo=repo,
            language=language,
            # Both settings reduce hallucination loops on silent stretches
            condition_on_previous_text=False,
            word_timestamps=True,
            hallucination_silence_threshold=2.0,
        )
        segs = [Segment(float(s["start"]), float(s["end"]), s["text"].strip())
                for s in result.get("segments", [])]
        return Transcript(segs, result.get("language"))


@register("faster-whisper")
class FasterWhisperEngine(Engine):
    def __init__(self, model: str = DEFAULT_MODEL):
        super().__init__(model)
        self._model = None

    @classmethod
    def is_installed(cls) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except Exception:
            return False

    def load(self) -> None:
        self._load()

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            # device="auto" uses an NVIDIA GPU when present, otherwise CPU (int8)
            self._model = WhisperModel(self.model, device="auto", compute_type="auto")
        return self._model

    def transcribe(self, wav: Path, language: Optional[str]) -> Transcript:
        segments, info = self._load().transcribe(
            str(wav),
            language=language,
            beam_size=5,
            vad_filter=True,                  # skip silence -> fewer hallucinations
            condition_on_previous_text=False,  # stops loops from feeding themselves
        )
        segs = [Segment(s.start, s.end, s.text.strip()) for s in segments]
        return Transcript(segs, info.language)


def pick_engine(name: str = "auto", model: str = DEFAULT_MODEL) -> Engine:
    if name != "auto":
        if name not in _REGISTRY:
            raise ValueError(f"Unknown engine '{name}'. Options: {', '.join(_REGISTRY)}")
        cls = _REGISTRY[name]
        if not cls.is_installed():
            raise RuntimeError(f"Engine '{name}' is not installed. See README > Install.")
        return cls(model)

    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    order = ["mlx", "faster-whisper"] if apple_silicon else ["faster-whisper"]
    for n in order:
        if _REGISTRY[n].is_installed():
            return _REGISTRY[n](model)
    hint = "pip install -e '.[mlx]'" if apple_silicon else "pip install -e '.[faster]'"
    raise RuntimeError(f"No transcription engine installed. Run: {hint}")
