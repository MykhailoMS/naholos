---
language:
- uk
license: apache-2.0
base_model: google/byt5-base
library_name: naholos
pipeline_tag: token-classification
tags:
- ukrainian
- stress
- accentuation
- text-to-speech
---

# naholos-tagger

Ukrainian word stress (*наголос*) tagger. It is a ByT5-base encoder with one linear head that
scores every character of a sentence. Inside each word with two or more vowels, the
highest-scoring vowel is stressed. One encoder pass handles a whole sentence (about 4 ms on an
RTX 4090), and the letters of the text can never change.

Use it through the [naholos](https://github.com/MykhailoMS/naholos) package. That package
combines this model with the
[ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress) dictionary:

```bash
pip install git+https://github.com/MykhailoMS/naholos
```

```python
from naholos import Naholos

n = Naholos()                 # hybrid: dictionary where it allows one stress, this model elsewhere
n("Мені потрібен атлас світу.")      # 'Мені́ потрі́бен а́тлас сві́ту.'
Naholos(mode="model")         # this model for every word
n("Старий за+мок на горі.")   # your own marks (за́мок / за+мок) always win; see user_dict too
```

## Files

- `config.json`: T5 encoder config.
- `model.safetensors`: bf16 weights, the encoder (`enc.*`) and `head.weight` / `head.bias`.
- `naholos.json`: format tag and the maximum input length in bytes.

## Accuracy

Word accuracy on words with two or more vowels whose stress is known:

| set | words | ukrainian-word-stress 2.1 (+Stanza) | this model | naholos hybrid |
|---|---:|---:|---:|---:|
| expert-stressed literary text, every word (private set) | 9666 | 0.889 | 0.917 | **0.935** |
| lang-uk human gold, ambiguous words | 1472 | 0.521 | 0.751 | **0.755** |
| held-out audiobook forms (stress heard) | 3102 | 0.741 | 0.924 | **0.929** |
| Common Voice (stress heard)\* | 3001 | 0.638 | 0.939 | **0.939** |

\* Partly overlaps the training data.

## Training

`google/byt5-base` was first fine-tuned as a seq2seq stress model, then converted into this
tagger (encoder + head) and trained on about 800k sentences:

- **Stress heard in recordings:** about 1.5M audiobook and Common Voice utterances, aligned
  with the Montreal Forced Aligner and labelled by an acoustic stress detector. Only confident
  labels are used, and a sentence with any unsure word is dropped.
- **Ambiguous words from modern audiobooks:** labelled by the stress their narrators read.
  The author of [accentuk](https://github.com/vacuum-in/accentuk) provided this data for
  training this model. The data itself is not published and is not redistributed here.

The loss is per-vowel binary cross-entropy, applied only to words with exactly one known
stress. To fine-tune on your own data, see `python -m naholos.train` in the package README.

## Limitations

- Stress follows how narrators read. In `model` mode that can differ from the dictionary
  norm. The hybrid mode keeps the norm for words that have a single stress.
- When one sentence uses the same heteronym in two senses, both usually get the same stress.
- One-vowel words are left unmarked by default.
- Inputs longer than 480 bytes are split at sentence boundaries.

## License

Apache-2.0, like the base model. The naholos code is MIT.
