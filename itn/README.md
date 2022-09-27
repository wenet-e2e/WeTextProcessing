## Inverse Text Normalization

### 1. How To Use

``` bash
$ python inverse_normalize.py --text "text to be denormalized"
```

### 2. ITN Pipeline

There are 2 components in ITN pipeline:

* pre-processing (before tagger)
* non-standard word denormalization

#### 2.1 Pre-Processing

| Pre-Processing                       | Raw                        | Denormalized              | Note                                       |
| ------------------------------------ | -------------------------- | ----------------------- | ------------------------------------------ |
| Blacklist (Removal)                  | 呃这个呃啊我不知道         | 这个我不知道            | via `data/default/blacklist.tsv`           |

#### 2.2 Non-Standard-Words (NSW) Denormalization

| NSW type                | Raw                 | Denormalized                       | Note                           |
| ----------------------- | ------------------- | -------------------------------- | ------------------------------ |
| Numbers                 | 共四百六十五篇约三百一十五万字  | 共465篇约315万字 |                                |
|                         | 共计六点四二万人        |        共计6.42万人          |                                |
|                         | 幸运一百 | 幸运100 |       enable_standalone_number=True (default) ([ref](https://github.com/wenet-e2e/WeTextProcessing/blob/master/itn/chinese/rules/cardinal.py#L73-L75))                        |
|                         |  | 幸运一百 |          enable_standalone_number=False                      |
| Fraction                | 总量的五分之一以上       |        总量的1/5以上        |                                |
|                         | 相当于头发丝的十六分之一  |     相当于头发丝的1/16     |                                |
| Percentage              | 同比增长百分之六点三        |      同比增长6.3%        |                                |
| Date                    | 二零零二年一月二十八日          |   2002/01/28         |                                |
|                         | 二零零二年一月             |        2002/01            |                                |
| Time                    | 八月十六号十二点之前    |      08/16 12点之前        |                                |
|                         | 我是五点零二分开始的      |       我是5:02开始的       |                                |
|                         | 于五点三十五分三十六秒发射       |   于5:35:36发射     |                                |
|                         | 早上八点半准时开会   |         8:30a.m.准时开会         |                                |
| Math                    | 比分定格在七十八比九十六     |      比分定格在78:96    |                                |
|                         | 计算负二的绝对值是二   |      计算-2的绝对值是2        |     enable_standalone_number=True (default)                           |
|                         |    |      计算负二的绝对值是二        |        enable_standalone_number=False                        |
|                         | 正负二的平方都是四       |        ±2的平方都是4        |         enable_standalone_number=True (default)                       |
|                         |        |        正负二的平方都是四        |           enable_standalone_number=False                     |
| Money                   | 价格是十三点五元        |      价格是¥13.5            |                                |
|                         | 价格是十三点五美元         |     价格是$13.5           |                                |
| Measure                 | 重达二十五千克            |      重达25kg              |                                |
|                         | 最高气温三十八摄氏度        |      最高气温38°C        |                                |
|                         | 速度是每小时十公里 | 速度是10km/h |                                |
|                         | 一年后 | 一年后 |      exclue_one=True (default) ([ref](https://github.com/wenet-e2e/WeTextProcessing/blob/master/itn/chinese/rules/measure.py#L43-L44))                      |
|                         |  | 1年后 |       exclue_one=False                         |
| Number series           | 可以拨打幺二三零六来咨询 |    可以拨打12306来咨询      |                                |
| Whitelist (Replacement) | 三心二意                 | 三心二意                            |                                |

## Acknowledge

1. Thank the authors of foundational libraries like [OpenFst](https://www.openfst.org/twiki/bin/view/FST/WebHome) & [Pynini](https://www.openfst.org/twiki/bin/view/GRM/Pynini).
3. Thank [NeMo](https://github.com/NVIDIA/NeMo) team & NeMo open-source community.
2. Thank [Jiayu Du](https://github.com/dophist), and [SpeechColab](https://github.com/SpeechColab) organization.
3. Referred [Pynini](https://github.com/kylebgorman/pynini) for reading the FAR, and printing the shortest path of a lattice in the C++ runtime.
4. Referred [ITN of chinese_text_normalization](https://github.com/speechio/chinese_text_normalization/tree/master/thrax/src/cn) for the data to build the tagger graph.
