#!/usr/bin/env python3
"""Convert a tagger trained with the mova-tts / uk-stress scripts (dir with the ByT5 encoder
safetensors + head.pt) into the naholos model format (config.json, model.safetensors,
naholos.json), optionally checking that the scores are unchanged."""
import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--meta", default="{}", help="extra JSON for naholos.json")
    a = ap.parse_args()
    import json
    from transformers import T5EncoderModel
    from naholos.model import Tagger, score
    enc = T5EncoderModel.from_pretrained(a.src)
    m = Tagger(enc.config)
    m.enc = enc
    m.head.load_state_dict(torch.load(os.path.join(a.src, "head.pt"), map_location="cpu"))
    m.eval()
    m.save(a.out, meta=json.loads(a.meta))
    from naholos.text import word_spans
    s = ["Наш атлас доріг лежав поруч із шовковим атласом.", "Учора визирнуло сонце, але вже зранку."]
    ref = score(m, s, "cpu")
    back, _ = Tagger.load(a.out, "cpu", dtype=torch.float32)
    got = score(back, s, "cpu")
    same = tot = 0
    for t, r, g in zip(s, ref, got):
        for _, _, vow in word_spans(t):
            tot += 1
            same += max(vow, key=r.__getitem__) == max(vow, key=g.__getitem__)
    diff = max(abs(x - y) for r, g in zip(ref, got) for x, y in zip(r, g))
    print(f"exported to {a.out}; same stressed vowel after bf16 round trip {same}/{tot}, "
          f"max score diff {diff:.3f}")
    assert same == tot


if __name__ == "__main__":
    main()
