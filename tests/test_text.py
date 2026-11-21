import torch

from naholos.text import ACUTE, apply_marks, chunks, encode_bytes, marks_of, strip_marks, word_spans


def test_marks_round_trip():
    s = "Учо́ра над мі́стом ви́зирнуло со́нце."
    plain, marks = marks_of(s)
    assert plain == strip_marks(s) == "Учора над містом визирнуло сонце."
    assert apply_marks(plain, marks) == s


def test_word_spans_skip_monosyllables_and_split_hyphens():
    spans = word_spans("Він пішов куди-небудь, а м'ята росте.")
    words = ["Він пішов куди-небудь, а м'ята росте."[a:b] for a, b, _ in spans]
    assert words == ["пішов", "куди", "небудь", "м'ята", "росте"]


def test_encode_bytes_matches_byt5_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/byt5-small")
    s = "Учора визирнуло сонце, ну й ну! 2026 р."
    ids, last = encode_bytes(s)
    assert tok(s)["input_ids"] == ids
    assert len(last) == len(s) and last[-1] == len(ids) - 2


def test_chunks_cover_text_within_limit():
    text = ("Це довге речення про погоду і місто. " * 30).strip()
    pieces = list(chunks(text, 120))
    assert all(len(p.encode()) <= 120 for _, p in pieces)
    assert all(text[o:o + len(p)] == p for o, p in pieces)
    assert "".join(p for _, p in pieces).replace(" ", "") == text.replace(" ", "")


class _FakeTagger(torch.nn.Module):
    """Scores the LAST vowel of every word highest - enough to check the plumbing."""

    def forward(self, ids, att):
        return torch.arange(ids.shape[1], dtype=torch.float32).repeat(ids.shape[0], 1)


def _fake(**kw):
    from naholos.core import Naholos
    n = Naholos.__new__(Naholos)
    n.mode, n.monosyllables, n.symbol = "model", False, ACUTE
    n.keep_marks, n.plus_marks, n.user_dict = kw.get("keep_marks", True), kw.get("plus_marks", True), {}
    n._dict = None
    n.tagger, n.device, n.max_bytes = _FakeTagger(), "cpu", 480
    return n


def test_model_mode_plumbing():
    n = _fake()
    assert n("молоко і вода") == "молоко́ і вода́"
    assert n(["його", "сонце"]) == ["його́", "сонце́"]


def test_user_marks_notations():
    from naholos.text import user_marks
    assert user_marks("за+мок і моло+ко") == ("замок і молоко", {1, 11})
    assert user_marks("молоко+ 2+2 C++") == ("молоко 2+2 C++", {5})
    assert user_marks("за́мок за´мок") == ("замок замок", {1, 7})
    assert user_marks("за+мок", plus=False) == ("за+мок", set())


def test_marks_in_text_win_over_model():
    n = _fake()
    assert n("Він за́мок зачинив") == "Він за́мок зачини́в"
    assert n("Він за+мок зачинив, 2+2") == "Він за́мок зачини́в, 2+2"
    assert n("ві́н сказав") == "ві́н сказа́в"           # a marked one-vowel word keeps the mark
    assert _fake(keep_marks=False)("за́мок") == "замо́к"


def test_user_dict_wins_over_model(tmp_path):
    n = _fake()
    n.add_words({"замок": "за́мок"}, ["Мико+ла"])
    f = tmp_path / "words.txt"
    f.write_text("# my words\nпо+ки  # comment\n\n", encoding="utf-8")
    n.add_words(str(f))
    assert n("Замок, Микола, поки") == "За́мок, Мико́ла, по́ки"
    assert n("за́мок замок") == "за́мок за́мок"
    import pytest
    with pytest.raises(ValueError):
        n.add_words("замок")                        # no mark
