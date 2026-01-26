from dataclasses import dataclass
from typing import Any

import torch

MUQ_SAMPLE_RATE = 24000

DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_device(device: str | None) -> str:
    """Select compute device with mps > cuda > cpu priority."""
    if device is not None:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(frozen=True, slots=True)
class MuqEmbedder:
    """Wrapper for MuQ-MuLan audio-text embedding model."""

    model: Any
    device: str
    sample_rate: int

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "MuqEmbedder":
        """Load MuQ-MuLan model from HuggingFace hub."""
        # deferred import; avoids dependency at module level
        from muq import MuQMuLan

        resolved_device = _resolve_device(device)

        model = MuQMuLan.from_pretrained(model_id, torch_dtype=dtype)
        model = model.to(resolved_device)
        model.eval()

        return cls(
            model=model,
            device=resolved_device,
            sample_rate=MUQ_SAMPLE_RATE,
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        model_name: str,
    ) -> "MuqEmbedder":
        """Load MuQ-MuLan model from config dict."""
        models = config.get("models", {})
        model_spec = models.get(model_name)

        if model_spec is None:
            raise ValueError(f"model '{model_name}' not found in config")

        model_id = model_spec.get("model_id")
        if not model_id:
            raise ValueError(f"model_id not specified for '{model_name}'")

        device = model_spec.get("device")
        dtype_str = model_spec.get("dtype", "float32")
        dtype = DTYPE_MAP.get(dtype_str, torch.float32)

        return cls.from_pretrained(model_id, device=device, dtype=dtype)

    @torch.no_grad()
    def embed_audio(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Compute audio embeddings from raw waveforms."""
        waveforms = waveforms.to(self.device)
        return self.model(wavs=waveforms)

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> torch.Tensor:
        """Compute text embeddings from string descriptions."""
        return self.model(texts=texts)
