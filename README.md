## Text Normalization & Inverse Text Normalization

### 1. How To Use

#### Usage-1: from pip
```bash
# install
pip install WeTextProcessing
```

```py
# tn usage
from tn.chinese.normalizer import Normalizer
normalizer = Normalizer()
normalizer.normalize("2.5平方电线")
# itn usage
from itn.chinese.inverse_normalizer import InverseNormalizer
invnormalizer = InverseNormalizer()
invnormalizer.normalize("二点五平方电线")
```

#### Usage-2: from source code

``` bash
# install
git clone https://github.com/wenet-e2e/WeTextProcessing.git
cd WeTextProcessing
wget -P tn https://github.com/wenet-e2e/WeTextProcessing/releases/download/WeTextProcessing/zh_tn_normalizer.far
wget -P itn https://github.com/wenet-e2e/WeTextProcessing/releases/download/WeTextProcessing/zh_itn_normalizer.far
```

```bash
# tn usage
python normalize.py --text "2.5平方电线"
# itn usage
python inverse_normalize.py --text "二点五平方电线"
```

**Advanced usage**: For users who want modifications and adapt tn/itn rules to fix badcase, please try:

```bash
# overwrite_cache will rebuild all rules according to
# your modifications on xx/xx/rules/xx.py.
python normalize.py --text "2.5平方电线" --overwrite_cache
python inverse_normalize.py --text "二点五平方电线" --overwrite_cache
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
