"""The stress tagger: ByT5 encoder + one linear head that scores every character; inside each
word with >= 2 vowels the highest-scoring vowel is stressed. One encoder pass per sentence,
and the letters of the text can never change."""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from .text import encode_bytes

FORMAT = "naholos-tagger/1"


class Tagger(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers import T5EncoderModel
        self.enc = T5EncoderModel(config)
        self.head = nn.Linear(config.d_model, 1)

    @classmethod
    def from_encoder(cls, name_or_path: str):
        """New tagger on top of a pretrained ByT5 (or a ByT5 seq2seq checkpoint)."""
        from transformers import T5EncoderModel
        enc = T5EncoderModel.from_pretrained(name_or_path)
        m = cls(enc.config)
        m.enc = enc
        return m

    def forward(self, ids, attention_mask):
        h = self.enc(input_ids=ids, attention_mask=attention_mask).last_hidden_state
        return self.head(h).squeeze(-1).float()

    # ----------------------------------------------------------------- save / load
    def save(self, path: str, dtype=torch.bfloat16, meta: dict | None = None):
        from safetensors.torch import save_file
        os.makedirs(path, exist_ok=True)
        self.enc.config.save_pretrained(path)
        sd = {k: v.detach().to(dtype).contiguous() for k, v in self.state_dict().items()}
        save_file(sd, os.path.join(path, "model.safetensors"), metadata={"format": FORMAT})
        json.dump({"format": FORMAT, "max_bytes": 480, **(meta or {})},
                  open(os.path.join(path, "naholos.json"), "w"), indent=1, ensure_ascii=False)

    @staticmethod
    def is_tagger(name_or_path: str) -> bool:
        """True if the dir / HF repo holds a naholos tagger (has naholos.json)."""
        if os.path.isdir(name_or_path):
            return os.path.isfile(os.path.join(name_or_path, "naholos.json"))
        from huggingface_hub import file_exists
        try:
            return file_exists(name_or_path, "naholos.json")
        except Exception:
            return False

    @classmethod
    def load(cls, name_or_path: str, device: str = "cpu", dtype=None):
        from safetensors.torch import load_file
        from transformers import T5Config
        path = name_or_path
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(name_or_path, allow_patterns=["*.json", "*.safetensors"])
        m = cls(T5Config.from_pretrained(path))
        m.load_state_dict(load_file(os.path.join(path, "model.safetensors")))
        meta = json.load(open(os.path.join(path, "naholos.json")))
        dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
        return m.to(device=device, dtype=dtype).eval(), meta


@torch.no_grad()
def score(model: Tagger, texts: list[str], device: str, max_batch_bytes: int = 16000):
    """Per text: list of per-character scores (higher = more likely stressed)."""
    enc = [encode_bytes(t) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(enc[i][0]))
    out: list = [None] * len(texts)
    b = 0
    while b < len(order):
        L = len(enc[order[b]][0])
        e = b + 1
        while e < len(order) and len(enc[order[e]][0]) * (e - b + 1) <= max_batch_bytes:
            e += 1
        idx = order[b:e]
        L = max(len(enc[i][0]) for i in idx)
        ids = torch.zeros(len(idx), L, dtype=torch.long)
        for k, i in enumerate(idx):
            ids[k, :len(enc[i][0])] = torch.tensor(enc[i][0])
        s = model(ids.to(device), (ids > 0).long().to(device)).float().cpu()
        for k, i in enumerate(idx):
            last = enc[i][1]
            out[i] = s[k, last].tolist() if last else []
        b = e
    return out
