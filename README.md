## Text Normalization & Inverse Text Normalization

### 1. How To Use

#### 1.1 pip install
```bash
pip install WeTextProcessing
```

```py
# tn
from tn.chinese.normalizer import Normalizer
normalizer = Normalizer()
normalizer.normalize("2.5平方电线")
# itn
from itn.chinese.inverse_normalizer import InverseNormalizer
invnormalizer = InverseNormalizer()
invnormalizer.normalize("二点五平方电线")
```

#### 1.2 source code compilation

``` bash
git clone https://github.com/wenet-e2e/WeTextProcessing.git
cd WeTextProcessing
```

```bash
python normalize.py --text "2.5平方电线"
python inverse_normalize.py --text "二点五平方电线"
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
