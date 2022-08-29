## Chinese Text Normalization

### 1. How To Use

``` bash
$ python normalize.py --text "text to be normalized"
```

### 2. TN Pipeline

There are 3 components in TN pipeline:

* pre-processing (before tagger)
* non-standard word normalization
* post-processing (after verbalizer)

#### 2.1 Pre-Processing

| Pre-Processing                       | Raw                        | Normalized              | Note                                       |
| ------------------------------------ | -------------------------- | ----------------------- | ------------------------------------------ |
| Char Width Conversion (全角 => 半角) | 苹果宣布发布新ＩＰＨＯＮＥ | 苹果宣布发布新IPHONE    |                                            |
| Mapping symbols                      | 他说：“我们已经吃过了！”。 | 他说:"我们已经吃过了!". | via `data/char/fullwidth_to_halfwidth.tsv` |
| Blacklist (Removal)                  | 呃这个呃啊我不知道         | 这个我不知道            | via `data/default/blacklist.tsv`           |

#### 2.2 Non-Standard-Words (NSW) Normalization

| NSW type                | Raw                 | Normalized                       | Note                           |
| ----------------------- | ------------------- | -------------------------------- | ------------------------------ |
| Numbers                 | 共465篇，约315万字  | 共四百六十五篇，约三百一十五万字 |                                |
|                         | 共计6.42万人        | 共计六点四二万人                 |                                |
| Fraction                | 总量的1/5以上       | 总量的五分之一以上               |                                |
|                         | 相当于头发丝的1/16  | 相当于头发丝的十六分之一         |                                |
| Percentage              | 同比增长6.3%        | 同比增长百分之六点三             |                                |
| Date                    | 2002/01/28          | 二零零二年一月二十八日           |                                |
|                         | 2002-01-28          | 二零零二年一月二十八日           |                                |
|                         | 2002.01.28          | 二零零二年一月二十八日           |                                |
|                         | 2002/01             | 二零零二年一月                   |                                |
| Time                    | 8月16号12:00之前    | 八月十六号十二点之前             |                                |
|                         | 我是5:02开始的      | 我是五点零二分开始的             |                                |
|                         | 于5:35:36发射       | 于五点三十五分三十六秒发射       |                                |
|                         | 8:00 a.m.准时开会   | 早上八点准时开会                 |                                |
| Math                    | 比分定格在78:96     | 比分定格在七十八比九十六         |                                |
|                         | 计算-2的绝对值是2   | 计算负二的绝对值是二             |                                |
|                         | ±2的平方都是4       | 正负二的平方都是四               |                                |
| Money                   | 价格是￥13.5        | 价格是十三点五元                 |                                |
|                         | 价格是$13.5         | 价格是十三点五美元               |                                |
| Measure                 | 重达25kg            | 重达二十五千克                   |                                |
|                         | 最高气温38°C        | 最高气温三十八摄氏度             |                                |
|                         | 速度是10km/h | 速度是每小时十公里 |                                |
| Number series           | 可以拨打12306来咨询 | 可以拨打幺二三零六来咨询         |                                |
| Erhua (儿化音) Removal  | 我儿子喜欢这地儿    | 我儿子喜欢这地                   | via `data/erhua/whitelist.tsv` |
| Whitelist (Replacement) | CEO                 | C E O                            |                                |
|                         | O2O                 | O to O                           |                                |

#### 2.3 Post-Processing

| Post-Processing                | Raw      | Normalized                       | Note                                                         |
| ------------------------------ | -------- | -------------------------------- | ------------------------------------------------------------ |
| Punctuation Removal            | 你好!    | 你好                             |                                                              |
| Out-Of-Vocabulary (OOV) Tagger | 我们안녕 | `我们<oov>안</oov><oov>녕</oov>` | default charset (national standard) [通用规范汉字表](https://zh.wikipedia.org/wiki/通用规范汉字表) |
|                                | 雪の花   | `雪<oov>の</oov>花`              | extend charset  `data/char/charset_extension.tsv`            |

### 3. Credits

Author: Zhendong Peng @ Tsinghua University

The authors of this work would like to thank:

* The authors of foundational libraries like OpenFst & Pynini
* NeMo team and NeMo open-source community
* Zhenxiang MA
