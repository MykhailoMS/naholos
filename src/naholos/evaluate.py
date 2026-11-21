#!/usr/bin/env python3
"""Word-level stress accuracy of naholos modes on a marked JSON Lines file (same format as
training: {"text": "...со́нце..."}). Counts words with >= 2 vowels and exactly one gold mark.

    python -m naholos.evaluate gold.jsonl --modes dictionary,model,hybrid
"""
import argparse
import json
import time

from .core import DEFAULT_MODEL, Naholos
from .text import marks_of, word_spans


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("--modes", default="dictionary,model,hybrid")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--plain-dictionary", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    rows = []
    for line in open(a.gold, encoding="utf-8"):
        r = json.loads(line)
        rows.append(marks_of(r.get("text") or r.get("target")))
        if a.limit and len(rows) >= a.limit:
            break
    print(f"| mode | word accuracy | words | ms / sentence |\n|---|---|---|---|")
    for mode in a.modes.split(","):
        n = Naholos(mode=mode, model=a.model, device=a.device,
                    disambiguation="plain" if a.plain_dictionary else "stanza")
        t0 = time.perf_counter()
        preds = [marks_of(n(p))[1] for p, _ in rows]
        dt = (time.perf_counter() - t0) * 1000 / max(1, len(rows))
        ok = tot = 0
        for (plain, gold), pred in zip(rows, preds):
            for _, _, vow in word_spans(plain):
                g = [v for v in vow if v in gold]
                if len(g) != 1:
                    continue
                tot += 1
                ok += [v for v in vow if v in pred] == g
        print(f"| {mode} | {ok / max(1, tot):.4f} | {tot} | {dt:.1f} |", flush=True)


if __name__ == "__main__":
    main()
