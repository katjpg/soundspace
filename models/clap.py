from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModel, AutoProcessor


@dataclass(frozen=True, slots=True)
class ClapEmbedder:
    model: Any
    processor: Any
    device: str
    sample_rate: int


def from_pretrained(
    model_id: str,
    *,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> ClapEmbedder:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    sample_rate = 48000
    
    return ClapEmbedder(
        model=model,
        processor=processor,
        device=device,
        sample_rate=sample_rate,
    )


def from_config(config: dict[str, Any], model_name: str) -> ClapEmbedder:
    model_spec = config["models"].get(model_name)
    if model_spec is None:
        raise ValueError(f"model {model_name!r} not found in config")
    
    model_id = model_spec.get("model_id")
    if not model_id:
        raise ValueError(f"model_id not specified for {model_name!r}")
    
    device = model_spec.get("device")
    dtype_str = model_spec.get("dtype", "float32")
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)
    
    return from_pretrained(model_id, device=device, dtype=dtype)
