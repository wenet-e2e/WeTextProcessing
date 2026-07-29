## Text Normalization & Inverse Text Normalization

### 0. Brief Introduction

```diff
- **Must Read Doc** (In Chinese): https://mp.weixin.qq.com/s/q_11lck78qcjylHCi6wVsQ
```

[WeTextProcessing: Production First & Production Ready Text Processing Toolkit](https://mp.weixin.qq.com/s/q_11lck78qcjylHCi6wVsQ)

#### 0.1 Text Normalization

<div align=center><img src="https://user-images.githubusercontent.com/13466943/193439861-acfba531-13d1-4fca-b2f2-6e47fc10f195.png" alt="Cover" width="50%"/></div>

#### 0.2 Inverse Text Normalization

<div align=center><img src="https://user-images.githubusercontent.com/13466943/193439870-634c44a3-bd62-4311-bcf2-1427758d5f62.png" alt="Cover" width="50%"/></div>

### 1. How To Use

#### 1.1 Quick Start:
```bash
# install
pip install WeTextProcessing
```

Command-usage:

```bash
wetn --text "2.5平方电线"
weitn --text "二点五平方电线"
```

Python usage:

```py
from itn.chinese.inverse_normalizer import InverseNormalizer
from itn.english.inverse_normalizer import InverseNormalizer as EnInverseNormalizer
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer

# FST 缓存会记录构建参数及规则数据指纹；配置或规则变化时会自动重新构图。
# 日常使用不需要 overwrite_cache；只在需要无条件重建时将它设为 True。

zh_tn_text = "你好 WeTextProcessing 1.0，船新版本儿，船新体验儿，简直666，9和10"
zh_itn_text = "你好 WeTextProcessing 一点零，船新版本儿，船新体验儿，简直六六六，九和六"
en_tn_text = "Hello WeTextProcessing 1.0, life is short, just use wetext, 666, 9 and 10"
en_itn_text = "call me at five five five one two three four"
zh_tn_model = ZhNormalizer(remove_erhua=True)
zh_itn_model = InverseNormalizer(enable_0_to_9=False)
en_tn_model = EnNormalizer()
en_itn_model = EnInverseNormalizer()
print("中文 TN (去除儿化音，重新在线构图):\n\t{} => {}".format(zh_tn_text, zh_tn_model.normalize(zh_tn_text)))
print("中文ITN (小于10的单独数字不转换，重新在线构图):\n\t{} => {}".format(zh_itn_text, zh_itn_model.normalize(zh_itn_text)))
print("英文 TN (暂时还没有可控的选项，后面会加...):\n\t{} => {}\n".format(en_tn_text, en_tn_model.normalize(en_tn_text)))
print("英文 ITN:\n\t{} => {}\n".format(en_itn_text, en_itn_model.normalize(en_itn_text)))

zh_tn_model = ZhNormalizer(remove_erhua=False, overwrite_cache=True)
zh_itn_model = InverseNormalizer(enable_0_to_9=True, overwrite_cache=True)
print("中文 TN (不去除儿化音，重新在线构图):\n\t{} => {}".format(zh_tn_text, zh_tn_model.normalize(zh_tn_text)))
print("中文ITN (小于10的单独数字也进行转换，重新在线构图):\n\t{} => {}\n".format(zh_itn_text, zh_itn_model.normalize(zh_itn_text)))
```

FST graphs are cached automatically outside the source tree. Use
`cache_dir="/path/to/cache"` to choose another cache directory,
`cache_dir=False` to disable persistent caching, or `overwrite_cache=True` to
force a rebuild.

To get the changed spans between the input and normalized text:

```py
result = zh_tn_model.normalize_with_mapping("今天中午12点")
print(result.output_text)
# 今天中午十二点

for mapping in result.mappings:
    print(mapping.token_type, mapping.input_text, "=>", mapping.output_text)
# math 12 => 十二

print(result.as_dict())
# {
#   "input": "今天中午12点",
#   "output": "今天中午十二点",
#   "mappings": [{
#     "kind": "replace",
#     "token_type": "math",
#     "input": {"start": 4, "end": 6, "text": "12"},
#     "output": {"start": 4, "end": 6, "text": "十二"}
#   }]
# }
```

Offsets are Unicode character offsets, and `token_type` identifies the matched
grammar rule. Use `include_identity=True` to include unchanged tokens.

Both APIs also support n-best output:

```py
outputs = zh_tn_model.normalize("输入文本", nbest=3)
results = zh_tn_model.normalize_with_mapping("输入文本", nbest=3)
```

`nbest=1` returns a single result; `nbest>1` returns a list.

#### 1.2 Advanced Usage:

DIY your own rules && Deploy WeTextProcessing with cpp runtime !!

For users who want modifications and adapt tn/itn rules to fix badcase, please try:

``` bash
git clone https://github.com/wenet-e2e/WeTextProcessing.git
cd WeTextProcessing
pip install -r requirements.txt
pre-commit install # for clean and tidy code
# `overwrite_cache` will rebuild all rules according to
#   your modifications on tn/chinese/rules/xx.py (itn/chinese/rules/xx.py).
#   The resulting content-addressed bundle is stored in your user cache.
python -m tn --text "2.5平方电线" --overwrite_cache
python -m itn --text "二点五平方电线" --overwrite_cache
```

Both commands also accept `--file PATH`, or read UTF-8 text from standard
input when neither `--text` nor `--file` is supplied. Each input line produces
two output lines: the tagged representation followed by the verbalized result.
Only the input line ending is removed; leading, trailing, and internal spaces
are preserved.

Options are limited to the selected direction and language. For example:

```bash
python -m tn --language zh --no-remove-erhua --text "这儿"
python -m tn --language ja --transliterate --text "WeNet"
python -m itn --language zh --enable-0-to-9 --text "一二三"
python -m itn --language ja --full-to-half True --text "１２時"
```

Boolean options support both switch-style `--option` / `--no-option` and the
legacy `--option True|False` form. Use `python -m tn --language LANG --help`
or `python -m itn --language LANG --help` to see only the options supported by
that pipeline. The installed `wetn` and `weitn` commands expose the same
interfaces.

To use an explicit Python cache root:

```py
# tn usage
>>> from tn.chinese.normalizer import Normalizer
>>> normalizer = Normalizer(cache_dir="/path/to/wetext-cache")
>>> normalizer.normalize("2.5平方电线")
# itn usage
>>> from itn.chinese.inverse_normalizer import InverseNormalizer
>>> invnormalizer = InverseNormalizer(cache_dir="/path/to/wetext-cache")
>>> invnormalizer.normalize("二点五平方电线")
# Disable persistent caching:
>>> uncached = Normalizer(cache_dir=False)
```

Python cache files are not C++ runtime exports. Rule contributors should read
the [Python rule architecture guide](docs/python-rule-architecture.md).

### 2. TN Pipeline

Please refer to [TN.README](tn/README.md)

### 3. ITN Pipeline

Please refer to [ITN.README](itn/README.md)

## Discussion & Communication

For Chinese users, you can aslo scan the QR code on the left to follow our offical account of WeNet.
We created a WeChat group for better discussion and quicker response.
Please scan the personal QR code on the right, and the guy is responsible for inviting you to the chat group.

| <img src="https://github.com/robin1001/qr/blob/master/wenet.jpeg" width="250px"> | <img src="https://user-images.githubusercontent.com/13466943/203046432-f637180e-4c87-40cc-be05-ce48c65dd1ef.jpg" width="250px"> |
| ---- | ---- |

Or you can directly discuss on [Github Issues](https://github.com/wenet-e2e/WeTextProcessing/issues).

## Acknowledge

1. Thank the authors of foundational libraries like [OpenFst](https://www.openfst.org/twiki/bin/view/FST/WebHome) & [Pynini](https://www.openfst.org/twiki/bin/view/GRM/Pynini).
3. Thank [NeMo](https://github.com/NVIDIA/NeMo) team & NeMo open-source community.
2. Thank [Zhenxiang Ma](https://github.com/mzxcpp), [Jiayu Du](https://github.com/dophist), and [SpeechColab](https://github.com/SpeechColab) organization.
3. Referred [Pynini](https://github.com/kylebgorman/pynini) for reading the FAR, and printing the shortest path of a lattice in the C++ runtime.
4. Referred [TN of NeMo](https://github.com/NVIDIA/NeMo/tree/main/nemo_text_processing/text_normalization/zh) for the data to build the tagger graph.
5. Referred [ITN of chinese_text_normalization](https://github.com/speechio/chinese_text_normalization/tree/master/thrax/src/cn) for the data to build the tagger graph.
