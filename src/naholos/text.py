"""Text helpers shared by inference and training: words, vowels, stress marks, byte ids."""
from __future__ import annotations

import re

ACUTE = "\u0301"
VOWELS = frozenset("аеиіоуяюєїАЕИІОУЯЮЄЇ")
# a word: letters with apostrophes inside; hyphenated compounds are separate words
WORD = re.compile(r"[^\W\d_]+(?:['’ʼ`][^\W\d_]+)*", re.U)
BYTE_OFFSET, EOS_ID = 3, 1          # ByT5: token id = byte + 3, </s> = 1

# One-vowel function words never carry a mark in running speech; a few multi-vowel
# clitics where the dictionary is wrong in isolation (checked against expert marks).
CLITIC_OVERRIDE = {"його": 3, "поміж": 3}      # index of the stressed vowel CHARACTER


def strip_marks(text: str) -> str:
    return text.replace(ACUTE, "").replace("\u00b4", "")


def n_vowels(word: str) -> int:
    return sum(c in VOWELS for c in word)


def word_spans(text: str):
    """[(start, end, [char indices of the word's vowels])] for words with >= 2 vowels."""
    out = []
    for m in WORD.finditer(text):
        v = [m.start() + i for i, c in enumerate(m.group(0)) if c in VOWELS]
        if len(v) >= 2:
            out.append((m.start(), m.end(), v))
    return out


def marks_of(marked: str) -> tuple[str, set[int]]:
    """'со́нце' -> ('сонце', {1}): plain text and indices of the stressed characters."""
    plain, marks = [], set()
    for ch in marked:
        if ch in (ACUTE, "\u00b4"):
            if plain:
                marks.add(len(plain) - 1)
        else:
            plain.append(ch)
    return "".join(plain), marks


_LETTER = re.compile(r"[^\W\d_]", re.U)


def user_marks(text: str, plus: bool = True) -> tuple[str, set[int]]:
    """Stress the user already put in the text: U+0301, U+00B4 (´) or, with plus=True, a "+"
    right after a vowel inside a word ("за+мок", "молоко+"). Returns (plain text, indices of
    the stressed characters). Other "+" signs ("2+2", "C++") are left alone."""
    plain, marks = [], set()
    n = len(text)
    for i, ch in enumerate(text):
        if ch in (ACUTE, "\u00b4"):
            if plain:
                marks.add(len(plain) - 1)
            continue
        if (plus and ch == "+" and plain and plain[-1] in VOWELS
                and ((len(plain) > 1 and _LETTER.match(plain[-2]))
                     or (i + 1 < n and _LETTER.match(text[i + 1])))):
            marks.add(len(plain) - 1)
            continue
        plain.append(ch)
    return "".join(plain), marks


def apply_marks(text: str, marks: set[int], symbol: str = ACUTE) -> str:
    return "".join(ch + (symbol if i in marks else "") for i, ch in enumerate(text))


def encode_bytes(text: str) -> tuple[list[int], list[int]]:
    """ByT5 ids of the text (+EOS) and, per character, the position of its LAST byte."""
    ids, last = [], []
    for ch in text:
        b = ch.encode("utf-8")
        ids.extend(x + BYTE_OFFSET for x in b)
        last.append(len(ids) - 1)
    return ids + [EOS_ID], last


_SENT = re.compile(r"(?<=[.!?…])\s+")


def chunks(text: str, max_bytes: int):
    """Split long text into pieces of <= max_bytes UTF-8 bytes, at sentence ends, then at
    spaces. Yields (offset, piece)."""
    pos = 0
    for sent in _SENT.split(text):
        start = text.index(sent, pos) if sent else pos
        pos = start + len(sent)
        if len(sent.encode()) <= max_bytes:
            if sent:
                yield start, sent
            continue
        a = 0
        while a < len(sent):
            b = len(sent)
            while len(sent[a:b].encode()) > max_bytes:
                cut = sent.rfind(" ", a, b - 1)
                b = cut if cut > a else b - 1
            yield start + a, sent[a:b]
            a = b
