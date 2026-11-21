"""Show all modes on a few sentences with timings: python scripts/demo.py [model] [device]"""
import sys
import time

from naholos import Naholos

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Mikhailo/naholos-tagger"
DEVICE = sys.argv[2] if len(sys.argv) > 2 else None
S = [
    "Наш атлас доріг лежав поруч із шовковим атласом.",
    "Він зачинив замок на дверях і пішов дивитися на старовинний замок.",
    "Учора ввечері над містом пройшла сильна злива, але вже зранку визирнуло сонце.",
]
for mode in ("hybrid", "model", "dictionary"):
    n = Naholos(mode=mode, model=MODEL, device=DEVICE)
    n(S[0])
    t = time.perf_counter()
    out = n(S)
    print(f"{mode}: {(time.perf_counter() - t) * 1000 / len(S):.0f} ms/sentence")
    for o in out:
        print("   ", o)
