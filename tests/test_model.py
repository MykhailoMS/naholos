"""End-to-end with the released model: pytest -m slow (downloads the model + Stanza)."""
import os

import pytest

MODEL = os.environ.get("NAHOLOS_MODEL", "Mikhailo/naholos-tagger")


@pytest.fixture(scope="module")
def hybrid():
    from naholos import Naholos
    return Naholos(model=MODEL)


@pytest.mark.slow
def test_heteronyms_in_context(hybrid):
    out = hybrid("Наш атлас доріг лежав поруч із шовковим атласом.")
    assert "атла́с" in out or "а́тлас" in out
    assert hybrid("Учора визирнуло сонце.").endswith("со́нце.")


@pytest.mark.slow
def test_letters_never_change(hybrid):
    from naholos import strip_marks
    s = "«Ну й ну!» — сказав він, і пішов геть, у ніч, о 22:15."
    assert strip_marks(hybrid(s)) == s
