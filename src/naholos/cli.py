"""naholos "текст"   or   echo "текст" | naholos"""
import argparse
import sys

from .core import DEFAULT_MODEL, MODES, Naholos


def main(argv=None):
    ap = argparse.ArgumentParser(prog="naholos", description="Ukrainian word stress (U+0301 after the stressed vowel)")
    ap.add_argument("text", nargs="*", help="text; read from stdin, line by line, if omitted")
    ap.add_argument("--mode", choices=MODES, default="hybrid")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local dir")
    ap.add_argument("--device", default=None)
    ap.add_argument("--plain-dictionary", action="store_true", help="dictionary without Stanza")
    ap.add_argument("--monosyllables", action="store_true", help="also mark one-vowel words")
    ap.add_argument("--user-dict", metavar="FILE",
                    help="words to always stress your way, one per line: за́мок or за+мок")
    ap.add_argument("--no-plus", action="store_true",
                    help='do not read "+" after a vowel as a stress mark in the input')
    a = ap.parse_args(argv)
    n = Naholos(mode=a.mode, model=a.model, device=a.device, monosyllables=a.monosyllables,
                disambiguation="plain" if a.plain_dictionary else "stanza",
                user_dict=a.user_dict, plus_marks=not a.no_plus)
    if a.text:
        print(n(" ".join(a.text)))
        return
    for line in sys.stdin:
        print(n(line.rstrip("\n")))


if __name__ == "__main__":
    main()
