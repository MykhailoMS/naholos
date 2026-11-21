#!/usr/bin/env python3
"""Train / fine-tune the naholos tagger.

Data: JSON Lines, one sentence per line, the stressed vowel marked with U+0301:
    {"text": "Учо́ра над мі́стом ви́зирнуло со́нце."}
Only words with >= 2 vowels and exactly one mark are trained on; leave a word unmarked when
you do not know its stress - it is then simply skipped (no "unstressed" signal).
({"input": ..., "target": ...} pairs are accepted too; target is the marked text.)

    python -m naholos.train --train train.jsonl --val val.jsonl --out my-tagger \
        --init Mikhailo/naholos-tagger            # fine-tune the released tagger
        --init google/byt5-small                  # or start a new head on a plain ByT5

The output dir loads with Naholos(model="my-tagger").
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import Tagger
from .text import encode_bytes, marks_of, word_spans


def load(path, limit=0):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        marked = r.get("text") or r.get("target")
        plain, marks = marks_of(marked)
        rows.append((plain, marks))
        if limit and len(rows) >= limit:
            break
    return rows


def example(plain, marks):
    ids, last = encode_bytes(plain)
    lab = np.full(len(ids), -1, np.int64)
    for _, _, vow in word_spans(plain):
        hit = [v for v in vow if v in marks]
        if len(hit) != 1:
            continue                           # unknown / compound stress: not supervised
        for v in vow:
            lab[last[v]] = int(v == hit[0])
    return ids, lab


def batches(rows, budget, rng):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0].encode()))
    out, cur, mx = [], [], 0
    for i in order:
        L = len(rows[i][0].encode()) + 1
        if cur and max(mx, L) * (len(cur) + 1) > budget:
            out.append(cur)
            cur, mx = [], 0
        cur.append(i)
        mx = max(mx, L)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def collate(exs, dev):
    L = max(len(e[0]) for e in exs)
    ids = torch.zeros(len(exs), L, dtype=torch.long)
    lab = torch.full((len(exs), L), -1, dtype=torch.long)
    for k, (i, l) in enumerate(exs):
        ids[k, :len(i)] = torch.tensor(i)
        lab[k, :len(l)] = torch.from_numpy(l)
    return ids.to(dev), (ids > 0).long().to(dev), lab.to(dev)


@torch.no_grad()
def word_accuracy(model, rows, dev, budget=16000):
    """Accuracy of the argmax vowel on supervised (one-mark, multi-vowel) words."""
    model.eval()
    ok = tot = 0
    for b in batches(rows, budget, random.Random(0)):
        ids, att, lab = collate([example(*rows[i]) for i in b], dev)
        with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            s = model(ids, att).float()
        for k, i in enumerate(b):
            plain, marks = rows[i]
            last = encode_bytes(plain)[1]
            for _, _, vow in word_spans(plain):
                hit = [v for v in vow if v in marks]
                if len(hit) != 1:
                    continue
                tot += 1
                ok += max(vow, key=lambda v: float(s[k, last[v]])) == hit[0]
    model.train()
    return ok / max(1, tot), tot


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="Mikhailo/naholos-tagger",
                    help="a naholos tagger (fine-tune) or a ByT5 model (new head)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--budget", type=int, default=16000, help="padded bytes per batch")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--grad-ckpt", action="store_true", help="less memory, ~30%% slower")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    rng = random.Random(0)
    tr, va = load(a.train), load(a.val)
    print(f"train {len(tr)} sentences, val {len(va)}, device {dev}", flush=True)
    if Tagger.is_tagger(a.init):
        model, _ = Tagger.load(a.init, dev, dtype=torch.float32)
        print(f"fine-tuning tagger {a.init}", flush=True)
        head_lr = a.lr
    else:
        model = Tagger.from_encoder(a.init).to(dev)
        print(f"new head on {a.init}", flush=True)
        head_lr = a.lr * 20
    model.train()                       # from_pretrained/load give eval mode (no checkpointing)
    if a.grad_ckpt:
        model.enc.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr},
                             {"params": model.head.parameters(), "lr": head_lr}], weight_decay=0.01,
                            fused=dev.startswith("cuda"))
    per_epoch = len(batches(tr, a.budget, random.Random(1)))
    total = a.max_steps or a.epochs * per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / min(300, max(1, total // 10)))
        * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, s / total)))))
    acc, n = word_accuracy(model, va, dev)
    print(f"[eval] step 0 val word accuracy {acc:.4f} ({n} words)", flush=True)
    best, step, t0 = acc, 0, time.time()
    model.save(a.out, meta={"init": a.init, "step": 0, "val_word_accuracy": acc})
    while step < total:
        for b in batches(tr, a.budget, rng):
            ids, att, lab = collate([example(*tr[i]) for i in b], dev)
            with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                s = model(ids, att)
            m = lab >= 0
            if not m.any():
                continue
            loss = F.binary_cross_entropy_with_logits(s[m], lab[m].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 or step == total:
                print(f"step {step}/{total} loss {loss.item():.4f} {(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % a.eval_every == 0 or step == total:
                acc, n = word_accuracy(model, va, dev)
                print(f"[eval] step {step} val word accuracy {acc:.4f} ({n} words)", flush=True)
                if acc >= best:
                    best = acc
                    model.save(a.out, meta={"init": a.init, "step": step, "val_word_accuracy": acc})
            if step >= total:
                break
    print(f"best val word accuracy {best:.4f}, saved to {a.out}")


if __name__ == "__main__":
    main()
