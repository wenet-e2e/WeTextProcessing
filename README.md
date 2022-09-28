## Text Normalization & Inverse Text Normalization

### 1. How To Use

#### 1.1 Quick Start:
```bash
# install
pip install WeTextProcessing
```

```py
# tn usage
>>> from tn.chinese.normalizer import Normalizer
>>> normalizer = Normalizer()
>>> normalizer.normalize("2.5平方电线")
# itn usage
>>> from itn.chinese.inverse_normalizer import InverseNormalizer
>>> invnormalizer = InverseNormalizer()
>>> invnormalizer.normalize("二点五平方电线")
```

#### 1.2 Advanced Usage:

DIY your own rules && Deploy WeTextProcessing with cpp runtime !!

For users who want modifications and adapt tn/itn rules to fix badcase, please try:

``` bash
git clone https://github.com/wenet-e2e/WeTextProcessing.git
cd WeTextProcessing
# `overwrite_cache` will rebuild all rules according to
#   your modifications on tn/chinese/rules/xx.py (itn/chinese/rules/xx.py).
#   After rebuild, you can find new far files at `$PWD/tn` and `$PWD/itn`.
python normalize.py --text "2.5平方电线" --overwrite_cache
python inverse_normalize.py --text "二点五平方电线" --overwrite_cache
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
./build/bin/processor_main --far PATH_TO_GIT_CLONED_WETEXTPROCESSING/tn/zh_tn_normalizer.far --text "2.5平方电线"
# itn usage
./build/bin/processor_main --far PATH_TO_GIT_CLONED_WETEXTPROCESSING/itn/zh_itn_normalizer.far --text "二点五平方电线"
```

### 2. TN Pipeline

Please refer to [TN.README](tn/README.md)

### 3. ITN Pipeline

Please refer to [ITN.README](itn/README.md)

## Acknowledge

1. Thank the authors of foundational libraries like [OpenFst](https://www.openfst.org/twiki/bin/view/FST/WebHome) & [Pynini](https://www.openfst.org/twiki/bin/view/GRM/Pynini).
3. Thank [NeMo](https://github.com/NVIDIA/NeMo) team & NeMo open-source community.
2. Thank [Zhenxiang Ma](https://github.com/mzxcpp), [Jiayu Du](https://github.com/dophist), and [SpeechColab](https://github.com/SpeechColab) organization.
3. Referred [Pynini](https://github.com/kylebgorman/pynini) for reading the FAR, and printing the shortest path of a lattice in the C++ runtime.
4. Referred [TN of NeMo](https://github.com/NVIDIA/NeMo/tree/main/nemo_text_processing/text_normalization/zh) for the data to build the tagger graph.
5. Referred [ITN of chinese_text_normalization](https://github.com/speechio/chinese_text_normalization/tree/master/thrax/src/cn) for the data to build the tagger graph.
