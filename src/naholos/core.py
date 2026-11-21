"""Naholos: Ukrainian word stress = dictionary + neural tagger."""
from __future__ import annotations

import logging
import os

from .text import (ACUTE, CLITIC_OVERRIDE, VOWELS, WORD, apply_marks, chunks, marks_of,
                   n_vowels, strip_marks, user_marks, word_spans)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Mikhailo/naholos-tagger"
MODES = ("hybrid", "model", "dictionary")


class Naholos:
    """Put U+0301 after the stressed vowel of Ukrainian words.

    mode:
      "hybrid" (default) - the dictionary for words it allows only one stress, the tagger for
                 heteronyms (за́мок / замо́к) and unknown words. Closest to the literary norm.
      "model"  - the tagger for every word. Closer to how narrators actually speak (it was
                 trained on stress heard in recordings), also where that departs from the norm.
      "dictionary" - ukrainian-word-stress only (no neural model).
    disambiguation: for mode="dictionary" only: "stanza" (default; POS/morphology resolves
      some heteronyms, ~500 MB models) or "plain" (dictionary lookup only).
    monosyllables: also mark words with one vowel (ві́н). Default False.

    Your own stress always wins over the dictionary and the model:
      - in the text: a word you already marked keeps your mark - "за́мок", "за´мок" or "за+мок"
        (keep_marks=True; plus_marks=False turns the "+" notation off);
      - user_dict: words to always stress your way - {"замок": "за́мок"}, a list ["за+мок"],
        or a path to a UTF-8 file with one stressed word per line ("#" starts a comment).
        Matching ignores case. n.add_words("Мико́ла", "по+ки") adds more later.
    """

    def __init__(self, mode: str = "hybrid", model: str = DEFAULT_MODEL, device: str | None = None,
                 disambiguation: str = "stanza", monosyllables: bool = False, symbol: str = ACUTE,
                 user_dict=None, keep_marks: bool = True, plus_marks: bool = True):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.monosyllables, self.symbol = mode, monosyllables, symbol
        self.keep_marks, self.plus_marks = keep_marks, plus_marks
        self.user_dict: dict[str, frozenset[int]] = {}
        if user_dict is not None:
            self.add_words(user_dict)
        self._dict = self._trie = None
        if mode != "model":
            from ukrainian_word_stress import Disambiguation, Stressifier, StressSymbol
            from ukrainian_word_stress import stressify_ as _s
            # hybrid only takes the dictionary's single-stress words, where POS changes nothing,
            # so Stanza is loaded for the pure dictionary mode only
            stanza = disambiguation == "stanza" and mode == "dictionary"
            self._dict = Stressifier(stress_symbol=StressSymbol.CombiningAcuteAccent,
                                     disambiguation=Disambiguation.Stanza if stanza
                                     else Disambiguation.Dictionary)
            self._trie = self._dict.dict
            self._trie_value, self._parse_value = _s._trie_value, _s._parse_dictionary_value
        self.tagger = None
        if mode != "dictionary":
            import torch
            from .model import Tagger
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tagger, meta = Tagger.load(model, self.device)
            self.max_bytes = int(meta.get("max_bytes", 480))

    # ---------------------------------------------------------------- dictionary
    def n_positions(self, word: str) -> int:
        """Distinct stress variants of a surface form in the dictionary (0 = unknown).

        A capitalised word is also looked up in lower case: upstream matches the exact case
        first, so a sentence-initial "Хотів" would otherwise hit only the place name Хо́тів."""
        variants = set()
        for w in {word, word.lower()}:
            v = self._trie_value(self._trie, w)
            if v is not None:
                variants |= {tuple(a) for _, a in self._parse_value(v[0])}
        return len(variants)

    def single_stress(self, word: str) -> int | None:
        """Index of the stressed vowel if the dictionary allows exactly one stress, else None.

        Entries with two marks (ро́ків / рокі́в stored as one reading, compounds) do not count."""
        pos = set()
        for w in {word, word.lower()}:
            v = self._trie_value(self._trie, w)
            if v is not None:
                for _, a in self._parse_value(v[0]):
                    pos.add(tuple(a))
        if len(pos) != 1:
            return None
        (a,) = pos
        if len(a) != 1 or not 0 < a[0] <= len(word) or word[a[0] - 1] not in VOWELS:
            return None
        return a[0] - 1                             # trie stores the index after the vowel

    def _dict_marks(self, text: str) -> set[int]:
        plain, marks = marks_of(self._dict(text))
        if plain != text:                          # should not happen; never trust a shifted text
            log.warning("dictionary changed the text, ignoring its marks: %r", text)
            return set()
        return marks

    # ---------------------------------------------------------------- user stress
    def add_words(self, *words):
        """Always stress these words your way. Each item is a stressed word ("за́мок", "за+мок"),
        a {word: stressed} dict, a list of such words, or a path to a file with one per line."""
        for w in words:
            if isinstance(w, dict):
                self.add_words(*w.values())
            elif isinstance(w, (list, tuple, set)):
                self.add_words(*w)
            elif isinstance(w, os.PathLike) or (isinstance(w, str) and os.path.isfile(w)):
                with open(w, encoding="utf-8") as f:
                    self.add_words(*[s for s in (ln.split("#")[0].strip() for ln in f) if s])
            else:
                plain, marks = user_marks(w.strip(), True)
                if not WORD.fullmatch(plain) or not marks:
                    raise ValueError(f"user_dict entry must be one word with a stress mark: {w!r}")
                self.user_dict[plain.lower()] = frozenset(marks)

    # ---------------------------------------------------------------- inference
    def __call__(self, text):
        if isinstance(text, str):
            return self.batch([text])[0]
        return self.batch(list(text))

    def batch(self, texts: list[str]) -> list[str]:
        parsed = [user_marks(t, self.plus_marks) if self.keep_marks else (strip_marks(t), set())
                  for t in texts]
        plain = [p for p, _ in parsed]
        scores = self._scores(plain) if self.tagger is not None else [None] * len(plain)
        return [self._one(p, s, um) for (p, um), s in zip(parsed, scores)]

    def _scores(self, texts):
        """Per-character tagger scores for each text (long texts in <= max_bytes pieces)."""
        from .model import score
        pieces, owners = [], []
        for k, t in enumerate(texts):
            for off, p in chunks(t, self.max_bytes):
                pieces.append(p)
                owners.append((k, off))
        out = [[0.0] * len(t) for t in texts]
        for (k, off), s in zip(owners, score(self.tagger, pieces, self.device)):
            out[k][off:off + len(s)] = s
        return out

    def _one(self, text: str, sc, user: set[int] = frozenset()) -> str:
        marks = self._dict_marks(text) if self.mode == "dictionary" else set()
        final = set()
        for m in WORD.finditer(text):
            a, b = m.start(), m.end()
            word = m.group(0)
            low = word.lower()
            mine = {i for i in user if a <= i < b}
            if mine:                                # the user marked this word in the text
                final |= mine
                continue
            if low in self.user_dict:
                final |= {a + i for i in self.user_dict[low]}
                continue
            vow = [a + i for i, c in enumerate(word) if c in VOWELS]
            if len(vow) < 2:
                if self.monosyllables and len(vow) == 1:
                    final.add(vow[0])
                continue
            if low in CLITIC_OVERRIDE:
                final.add(a + CLITIC_OVERRIDE[low])
                continue
            if self.mode == "dictionary":
                final |= {i for i in marks if a <= i < b}
                continue
            if self.mode == "hybrid":
                k = self.single_stress(word)
                if k is not None:
                    final.add(a + k)
                    continue
            final.add(max(vow, key=lambda i: sc[i]))
        return apply_marks(text, final, self.symbol)


def stressify(text, **kw):
    """One-off helper: Naholos(**kw)(text). Keep a Naholos instance for repeated calls."""
    return Naholos(**kw)(text)


__all__ = ["Naholos", "stressify", "DEFAULT_MODEL", "word_spans", "n_vowels"]
