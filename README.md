<div align="center">

# WeTextProcessing

**Rule-based text normalization and inverse text normalization for speech applications.**

[Quick start](#quick-start) · [Languages](#supported-languages) · [Python API](#python-api) · [Rule architecture](docs/python-rule-architecture.md) · [C++ runtime](runtime/README.md)

[![Unit tests](https://github.com/wenet-e2e/WeTextProcessing/actions/workflows/unittest.yml/badge.svg)](https://github.com/wenet-e2e/WeTextProcessing/actions/workflows/unittest.yml)

</div>

WeTextProcessing converts between written and spoken text using weighted finite-state transducers (WFSTs), built with [Pynini](https://www.openfst.org/twiki/bin/view/GRM/Pynini) and [OpenFst](https://www.openfst.org/twiki/bin/view/FST/WebHome).

- **TN and ITN** — normalize numbers, dates, times, currencies, units, and other structured expressions.
- **Multilingual rules** — language-specific grammars for every language in [Supported languages](#supported-languages).
- **Explainable output** — inspect raw tagged fields, trace input/output spans, and request unique n-best results.
- **Reusable graphs** — automatic, configuration-aware FST caching and a separate C++ deployment runtime.

For background and motivation, read [WeTextProcessing: Production First & Production Ready Text Processing Toolkit](https://mp.weixin.qq.com/s/q_11lck78qcjylHCi6wVsQ) (Chinese).

## What are TN and ITN?

### Text normalization

TN converts written expressions into their spoken forms, for example `12` → `十二`. It is commonly used before speech synthesis.

<div align=center><img src="https://user-images.githubusercontent.com/13466943/193439861-acfba531-13d1-4fca-b2f2-6e47fc10f195.png" alt="Cover" width="50%"/></div>

### Inverse text normalization

ITN converts spoken expressions into written forms, for example `十二` → `12`. It is commonly used after speech recognition.

<div align=center><img src="https://user-images.githubusercontent.com/13466943/193439870-634c44a3-bd62-4311-bcf2-1427758d5f62.png" alt="Cover" width="50%"/></div>

## Supported languages

| Language | CLI code | TN | ITN | Python package |
| --- | --- | --- | --- | --- |
| Arabic | `ar` | Yes | Yes | `arabic` |
| Armenian | `hy` | Yes | Yes | `armenian` |
| Chinese | `zh` | Yes | Yes | `chinese` |
| English | `en` | Yes | Yes | `english` |
| French | `fr` | Yes | Yes | `french` |
| German | `de` | Yes | Yes | `german` |
| Hindi | `hi` | Yes | Yes | `hindi` |
| Japanese | `ja` | Yes | Yes | `japanese` |
| Kinyarwanda | `rw` | Yes | Yes | `kinyarwanda` |
| Korean | `ko` | Yes | Yes | `korean` |
| Portuguese | `pt` | Yes | Yes | `portuguese` |
| Russian | `ru` | Yes | Yes | `russian` |
| Spanish | `es` | Yes | Yes | `spanish` |
| Swedish | `sv` | Yes | Yes | `swedish` |
| Vietnamese | `vi` | Yes | Yes | `vietnamese` |

Chinese is the CLI default. Rule coverage and configurable options vary by language and direction.

The published package includes Chinese, English, and Japanese. The other languages are included in this source checkout; install from source to use them before the next package release.

## Quick start

### Install

Install the published package:

```bash
pip install WeTextProcessing
```

For the latest rules, including every language beyond Chinese, English, and Japanese, install from source:

```bash
git clone https://github.com/wenet-e2e/WeTextProcessing.git
cd WeTextProcessing
pip install -e .
```

Pynini/OpenFst must be available in your Python environment. If a compatible wheel is unavailable, follow the [Pynini installation instructions](https://www.openfst.org/twiki/bin/view/GRM/Pynini). On Windows, the project's CI uses `conda install -c conda-forge pynini` before installing WeTextProcessing.

### Run the CLI

```bash
wetn --text "2.5平方电线"
weitn --text "二点五平方电线"

wetn --language en --text "13 dollars"
weitn --language en --text "thirteen dollars"

wetn --language de --text "23"
weitn --language de --text "dreiundzwanzig"

wetn --language ja --text "１２時"
weitn --language ja --text "十二時"

wetn --language ko --text "오후 3시 30분"
weitn --language ko --text "오천원"
```

Each input produces two lines: the **tagged representation**, then the **verbalized result**. The tagger preserves raw field values; the verbalizer performs the conversion.

The same interfaces are available as `python -m tn` and `python -m itn`. Both accept `--text`, `--file PATH`, or UTF-8 text from standard input when neither option is supplied:

```bash
python -m tn --language en --file input.txt
python -m itn --language ko < input.txt
```

Only input line endings are removed before processing. Other spaces are passed to the selected language's rules.

### Language-specific options

Inspect the options for the selected language and direction:

```bash
wetn --language zh --help
weitn --language ko --help
```

For example:

```bash
wetn --language zh --no-remove-erhua --text "这儿"
weitn --language zh --enable-0-to-9 --text "一二三"
```

Boolean options accept `--option` / `--no-option` and the legacy `--option True|False` form. Options from another language or direction are not accepted.

## Python API

Import the normalizer for the language you need and reuse the instance:

```python
from tn.chinese.normalizer import Normalizer
from itn.chinese.inverse_normalizer import InverseNormalizer

normalizer = Normalizer()
inverse_normalizer = InverseNormalizer()

print(normalizer.normalize("今天中午12点"))
print(inverse_normalizer.normalize("二点五平方电线"))
```

Other languages use the same class names under their packages. For example, Korean:

```python
from tn.korean.normalizer import Normalizer
from itn.korean.inverse_normalizer import InverseNormalizer

normalizer = Normalizer()
inverse_normalizer = InverseNormalizer()

print(normalizer.normalize("오후 3시 30분"))
print(inverse_normalizer.normalize("오천원"))
```

### Inspect tags

Tagging and verbalization are independently accessible:

```python
tagged = normalizer.tag("오후 3시 30분")
print(tagged)
print(normalizer.verbalize(tagged))
```

The hours and minutes remain written values in the tags, rather than being replaced with their Korean readings.

### Input/output span mapping

`normalize_with_mapping()` returns normalized text and the spans matched by grammar rules:

```python
from tn.chinese.normalizer import Normalizer

normalizer = Normalizer()
result = normalizer.normalize_with_mapping("今天中午12点")

print(result.output_text)
for mapping in result.mappings:
    print(mapping.token_type, mapping.input_text, "=>", mapping.output_text)
```

Output:

```text
今天中午十二点
math 12 => 十二
```

Offsets are half-open Unicode character offsets, not UTF-8 byte offsets. Use `include_identity=True` to include unchanged tokens, or `result.as_dict()` to obtain a serializable representation. Mappings are traced through the WFSTs, not inferred from a text diff.

### N-best results

```python
outputs = normalizer.normalize("3/4", nbest=3)
results = normalizer.normalize_with_mapping("3/4", nbest=3)
```

`nbest=1` returns a string or a single `NormalizationResult`; `nbest>1` returns a list of up to that many unique outputs. The mapping API follows the same output ranking.

## FST caching

Graphs are cached automatically outside the source tree. Cache bundles record the pipeline's configuration and grammar-source fingerprint; changing either causes a new bundle to be built.

| Setting | Behavior |
| --- | --- |
| `cache_dir=None` | Use the platform's default user cache |
| `cache_dir="/path/to/cache"` | Choose an explicit cache root |
| `cache_dir=False` | Build in memory without persistent caching |
| `overwrite_cache=True` | Force rebuilding the selected bundle |

```python
from tn.chinese.normalizer import Normalizer

normalizer = Normalizer(cache_dir="/path/to/wetext-cache")
uncached = Normalizer(cache_dir=False)
```

The CLI exposes `--cache-dir PATH` and `--overwrite-cache`. Normal use does not require forced rebuilds.

## Development and deployment

Install development dependencies and run the tests:

```bash
pip install -e '.[dev]'
pre-commit install
pytest -q
```

Before adding or changing a language rule, read the [Python rule architecture guide](docs/python-rule-architecture.md). Rules preserve raw fields in the tagger, apply semantic conversion in the verbalizer, and use pipeline-owned rule inventories and field-order schemas.

| Guide | Scope |
| --- | --- |
| [TN rules](tn/README.md) | Text normalization rules and examples |
| [ITN rules](itn/README.md) | Inverse text normalization rules and examples |
| [Python rule architecture](docs/python-rule-architecture.md) | Tagging, verbalization, registration, and span mapping contracts |
| [C++ runtime](runtime/README.md) | Native build and deployment instructions |

Python cache bundles are **not** C++ runtime exports. Follow the runtime's model and build instructions for native deployment.

## Community

Use [GitHub Issues](https://github.com/wenet-e2e/WeTextProcessing/issues) for questions, bug reports, and feature requests.

For Chinese users, follow the WeNet official account using the QR code on the left. Scan the personal QR code on the right to contact the person who can invite you to the WeChat discussion group.

| <img src="https://github.com/robin1001/qr/blob/master/wenet.jpeg" width="250px"> | <img src="https://user-images.githubusercontent.com/13466943/203046432-f637180e-4c87-40cc-be05-ce48c65dd1ef.jpg" width="250px"> |
| ---- | ---- |

## Acknowledgments

- [OpenFst](https://www.openfst.org/twiki/bin/view/FST/WebHome) and [Pynini](https://www.openfst.org/twiki/bin/view/GRM/Pynini) for the finite-state toolkit.
- The [NeMo Text Processing](https://github.com/NVIDIA/NeMo-text-processing) team and community for text-processing grammars and resources, including the [Chinese TN reference](https://github.com/NVIDIA/NeMo-text-processing/tree/main/nemo_text_processing/text_normalization/zh).
- [Zhenxiang Ma](https://github.com/mzxcpp), [Jiayu Du](https://github.com/dophist), and [SpeechColab](https://github.com/SpeechColab) for their contributions.
- [Pynini](https://github.com/kylebgorman/pynini) for reference implementations of FAR loading and shortest-path decoding in the C++ runtime.
- [chinese_text_normalization](https://github.com/speechio/chinese_text_normalization/tree/master/thrax/src/cn) for Chinese ITN grammar resources.

## License

[Apache License 2.0](LICENSE).
