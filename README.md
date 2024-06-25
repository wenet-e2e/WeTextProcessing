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
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer

# NOTE(xcsong): 和默认参数不一致时，必须重新构图，要重新构图请务必指定 `overwrite_cache=True`
#               When the parameters differ from the defaults, it is mandatory to re-compose. To re-compose, please ensure you specify `overwrite_cache=True`.

zh_tn_text = "你好 WeTextProcessing 1.0，船新版本儿，船新体验儿，简直666，9和10"
zh_itn_text = "你好 WeTextProcessing 一点零，船新版本儿，船新体验儿，简直六六六，九和六"
en_tn_text = "Hello WeTextProcessing 1.0, life is short, just use wetext, 666, 9 and 10"
zh_tn_model = ZhNormalizer(remove_erhua=True, overwrite_cache=True)
zh_itn_model = InverseNormalizer(enable_0_to_9=False, overwrite_cache=True)
en_tn_model = EnNormalizer(overwrite_cache=True)
print("中文 TN (去除儿化音，重新在线构图):\n\t{} => {}".format(zh_tn_text, zh_tn_model.normalize(zh_tn_text)))
print("中文ITN (小于10的单独数字不转换，重新在线构图):\n\t{} => {}".format(zh_itn_text, zh_itn_model.normalize(zh_itn_text)))
print("英文 TN (暂时还没有可控的选项，后面会加...):\n\t{} => {}\n".format(en_tn_text, en_tn_model.normalize(en_tn_text)))

zh_tn_model = ZhNormalizer(overwrite_cache=False)
zh_itn_model = InverseNormalizer(overwrite_cache=False)
en_tn_model = EnNormalizer(overwrite_cache=False)
print("中文 TN (复用之前编译好的图):\n\t{} => {}".format(zh_tn_text, zh_tn_model.normalize(zh_tn_text)))
print("中文ITN (复用之前编译好的图):\n\t{} => {}".format(zh_itn_text, zh_itn_model.normalize(zh_itn_text)))
print("英文 TN (复用之前编译好的图):\n\t{} => {}\n".format(en_tn_text, en_tn_model.normalize(en_tn_text)))

zh_tn_model = ZhNormalizer(remove_erhua=False, overwrite_cache=True)
zh_itn_model = InverseNormalizer(enable_0_to_9=True, overwrite_cache=True)
print("中文 TN (不去除儿化音，重新在线构图):\n\t{} => {}".format(zh_tn_text, zh_tn_model.normalize(zh_tn_text)))
print("中文ITN (小于10的单独数字也进行转换，重新在线构图):\n\t{} => {}\n".format(zh_itn_text, zh_itn_model.normalize(zh_itn_text)))
```

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
#   After rebuild, you can find new far files at `$PWD/tn` and `$PWD/itn`.
python -m tn --text "2.5平方电线" --overwrite_cache
python -m itn --text "二点五平方电线" --overwrite_cache
```

Once you successfully rebuild your rules, you can deploy them either with your installed pypi packages:

```py
# tn usage
>>> from tn.chinese.normalizer import Normalizer
>>> normalizer = Normalizer(cache_dir="PATH_TO_GIT_CLONED_WETEXTPROCESSING/tn")
>>> normalizer.normalize("2.5平方电线")
# itn usage
>>> from itn.chinese.inverse_normalizer import InverseNormalizer
>>> invnormalizer = InverseNormalizer(cache_dir="PATH_TO_GIT_CLONED_WETEXTPROCESSING/itn")
>>> invnormalizer.normalize("二点五平方电线")
```

Or with cpp runtime:

```bash
cmake -B build -S runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build
# tn usage
cache_dir=PATH_TO_GIT_CLONED_WETEXTPROCESSING/tn
./build/processor_main --tagger $cache_dir/zh_tn_tagger.fst --verbalizer $cache_dir/zh_tn_verbalizer.fst --text "2.5平方电线"
# itn usage
cache_dir=PATH_TO_GIT_CLONED_WETEXTPROCESSING/itn
./build/processor_main --tagger $cache_dir/zh_itn_tagger.fst --verbalizer $cache_dir/zh_itn_verbalizer.fst --text "二点五平方电线"
```

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
