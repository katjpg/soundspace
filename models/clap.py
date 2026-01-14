from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModel, AutoProcessor

CLAP_SAMPLE_RATE = 48000

DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True, slots=True)
class ClapEmbedder:
    model: Any
    processor: Any
    device: str
    sample_rate: int

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "ClapEmbedder":
        """Load CLAP model from HuggingFace hub."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        model = model.to(device)
        model.eval()
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        return cls(
            model=model,
            processor=processor,
            device=device,
            sample_rate=CLAP_SAMPLE_RATE,
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        model_name: str,
    ) -> "ClapEmbedder":
        """Load CLAP model from config dict."""
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
