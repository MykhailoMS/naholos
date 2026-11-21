# naholos

Ukrainian word stress (*наголос*): puts U+0301 after the stressed vowel.

```python
from naholos import Naholos

n = Naholos()
n(["Мені потрібен атлас світу.", "Вона одягла сукню з шовкового атласу."])
# ['Мені́ потрі́бен а́тлас сві́ту.', 'Вона́ одягла́ су́кню з шовко́вого атла́су.']
```

It combines the [ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress)
dictionary with a small neural tagger (ByT5 encoder + one linear head over the characters).
The dictionary handles words with a single possible stress. The tagger reads the whole
sentence and resolves heteronyms (за́мок / замо́к, а́тлас / атла́с) and words that are not
in the dictionary. The tagger needs one encoder pass per sentence, about 4 ms on a GPU, and it
cannot change the letters of the text.

## Install

```bash
pip install naholos
```

The model ([Mikhailo/naholos-tagger](https://huggingface.co/Mikhailo/naholos-tagger), ~830 MB)
downloads on first use. The default `hybrid` mode uses only the dictionary lookup.
`mode="dictionary"` also loads Stanza for POS disambiguation (~500 MB of models on first
use), unless you pass `disambiguation="plain"`.

## Usage

```python
Naholos(mode="hybrid")       # default: dictionary where unambiguous, tagger elsewhere
Naholos(mode="model")        # tagger for every word: stress as narrators actually say it
Naholos(mode="dictionary")   # ukrainian-word-stress only, no neural model
Naholos(monosyllables=True)  # also mark one-vowel words (ві́н); off by default
n(["перше речення", "друге"])        # batches
```

```bash
naholos "Учора визирнуло сонце."
cat book.txt | naholos --mode model > book.stressed.txt
```

Long texts are split at sentence ends.

## Fixing a stress yourself

When the model gets a word wrong, mark it yourself. Your stress always wins over the
dictionary and the model.

In the text you send, a word you already marked keeps your mark. All three notations work:
`за́мок` (U+0301), `за´мок` (U+00B4) or `за+мок`. The output always uses U+0301, or the
`symbol=` you chose.

```python
n("Старий за+мок на горі, ключ у замку.")
# 'Стари́й за́мок на горі́, ключ у за́мку.'
```

A "+" counts as a mark only right after a vowel inside a word, so `2+2` and `C++` are left
alone. Pass `plus_marks=False` (CLI: `--no-plus`) if your text uses "+" otherwise.
`keep_marks=False` ignores every mark in the input and stresses the text from scratch.

For words you correct often, use a user dictionary. Matching ignores case:

```python
n = Naholos(user_dict={"замок": "за́мок"})       # or a list ["за+мок", "Мико+ла"]
n.add_words("по+ки", "Мико́ла")                  # add more at any time
n = Naholos(user_dict="my_words.txt")           # one stressed word per line, "#" = comment
```

```bash
naholos --user-dict my_words.txt "Ключ від замка"
```

A mark in the text beats the user dictionary. The user dictionary beats the built-in rules.

## Accuracy

The table counts word accuracy on words with two or more vowels whose stress is known, using
`python -m naholos.evaluate` (RTX 4090; ms per sentence in the last column).

| set | words | dictionary | model | hybrid |
|---|---:|---:|---:|---:|
| expert-stressed literary text, every word in 3000 sentences (private set) | 9666 | 0.889 | 0.917 | **0.935** |
| [lang-uk](https://github.com/lang-uk) human gold, ambiguous words | 1472 | 0.521 | 0.751 | **0.755** |
| held-out audiobook forms, stress heard in recordings | 3102 | 0.741 | 0.924 | **0.929** |
| Common Voice, stress heard in recordings\* | 3001 | 0.638 | 0.939 | **0.939** |
| ms / sentence | | 10–14 | 4.5–5 | 4.5–5 |

`dictionary` is ukrainian-word-stress 2.1 with Stanza disambiguation. It leaves a word
unmarked when it cannot decide, and that counts as an error. Without Stanza
(`disambiguation="plain"`) it scores 0.843 on the expert-stressed set. The lang-uk, audiobook and Common Voice
sets label only hard words (heteronyms and other ambiguous forms), so their numbers are much
lower than on running text.

\* Some of these Common Voice sentences were in the training data, so this row is optimistic.

`hybrid` follows the literary norm wherever the dictionary allows only one stress, and lets
the tagger decide everything else. `model` is closer to how people actually read: narrators
sometimes stress "unambiguous" words differently from the dictionary.

Known weakness: when one sentence uses the same heteronym in two senses ("старий замок на
горі, а на дверях новий замок"), both usually get the same stress.

## Training

Data is JSON Lines, one sentence per line, with the stressed vowel marked:

```json
{"text": "Учо́ра над мі́стом ви́зирнуло со́нце."}
```

Only words with two or more vowels and exactly one mark are trained on. Leave a word unmarked
if you don't know its stress, and it is skipped. A small example (Common Voice sentences, CC0,
with stress taken from the recordings) is in `examples/`:

```bash
# fine-tune the released tagger on your data
python -m naholos.train --train examples/train.jsonl --val examples/val.jsonl \
    --out my-tagger --init Mikhailo/naholos-tagger --max-steps 50 --budget 4000

# or train a new head on a plain ByT5
python -m naholos.train --train data/train.jsonl --val data/val.jsonl \
    --out my-tagger --init google/byt5-base --epochs 2 --lr 5e-5 --grad-ckpt

# evaluate any marked file
python -m naholos.evaluate examples/val.jsonl --model my-tagger
```

```python
Naholos(model="my-tagger")
```

On an RTX 4090, ByT5-base trains at ~1.2 s per step with `--budget 16000 --grad-ckpt`, using
about 12 GB of memory.

## How the model was trained

The model is `google/byt5-base`, first fine-tuned as a seq2seq stress model, then turned into
the tagger (encoder + head) and trained on about 800k sentences:

- stress heard in recordings: ~1.5M audiobook and Common Voice utterances, aligned with the
  Montreal Forced Aligner and labelled by an acoustic stress detector. Only confident labels
  are used, and whole sentences with an unsure word are dropped.
- ambiguous words from modern audiobooks, labelled by the stress their narrators read,
  provided by the author of [accentuk](https://github.com/vacuum-in/accentuk) for this model.
  That data is not redistributed.

## License

The code is MIT. The model is Apache-2.0, like its base `google/byt5-base`.
