#!/usr/bin/env python3
"""Upload an exported tagger dir + model_card.md to the HF Hub:
    python scripts/upload_model.py models/naholos-tagger-v1 Mikhailo/naholos-tagger
Only config.json, model.safetensors and naholos.json are uploaded (no training data)."""
import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi

src, repo = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True)
with tempfile.TemporaryDirectory() as d:
    for f in ("config.json", "model.safetensors", "naholos.json"):
        os.link(os.path.join(src, f), os.path.join(d, f))
    shutil.copy(os.path.join(here, "model_card.md"), os.path.join(d, "README.md"))
    api.upload_folder(folder_path=d, repo_id=repo, commit_message="naholos-tagger v1")
print("https://huggingface.co/" + repo)
