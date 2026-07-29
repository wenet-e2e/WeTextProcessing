# Copyright (c) 2022 Xingchen Song (sxc19@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pynini import accep, cross, string_file, union
from pynini.lib.pynutil import add_weight, delete, insert

from itn.chinese.rules.cardinal import Cardinal
from tn.processor import Processor
from itn.utils import get_abs_path


class Measure(Processor):

    def __init__(self, exclude_one=True, enable_0_to_9=True, cardinal=None):
        super().__init__(name="measure")
        self.exclude_one = exclude_one
        self.enable_0_to_9 = enable_0_to_9
        self.cardinal = cardinal or Cardinal()
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        units_en = string_file(get_abs_path("chinese/data/measure/units_en.tsv"))
        units_en |= add_weight(cross("度", "°"), -0.001)
        units_zh = string_file(get_abs_path("chinese/data/measure/units_zh.tsv"))
        sign = string_file(get_abs_path("chinese/data/number/sign.tsv"))  # + -
        digit = string_file(get_abs_path("chinese/data/number/digit.tsv"))  # 1 ~ 9
        digit_zh = string_file(get_abs_path("chinese/data/number/digit_zh.tsv"))  # 1 ~ 9
        addzero = insert("0")
        to = cross("到", "~") | cross("到百分之", "~")

        units = add_weight((accep("亿") | accep("兆") | accep("万")), -0.5).ques + units_zh
        units |= add_weight((cross("亿", "00M") | cross("兆", "T") | cross("万", "W")), -0.5).ques + (add_weight(units_en, -1.0))

        number = self.cardinal.number if self.enable_0_to_9 else self.cardinal.number_exclude_0_to_9
        # 百分之三十, 百分三十, 百分之百，百分之三十到四十, 百分之三十到百分之五十五
        percent = ((sign + delete("的").ques).ques + delete("百分") + delete("之").ques +
                   ((self.cardinal.number + (to + self.cardinal.number).ques)
                    | ((self.cardinal.number + to).ques + cross("百", "100"))) + insert("%"))

        # 十千米每小时 => 10km/h, 十一到一百千米每小时 => 11~100km/h
        measure = number + (to + number).ques + units

        # XXX: 特殊case处理, ignore enable_standalone_number
        # digit + union("百", "千", "万") + digit + unit
        unit_sp_case1 = [
            "年",
            "月",
            "个月",
            "周",
            "天",
            "位",
            "次",
            "个",
            "顿",
        ]
        if self.enable_0_to_9:
            measure_sp = add_weight(
                ((digit + delete("百") + add_weight(addzero**2, 1.0))
                 | (digit + delete("千") + add_weight(addzero**3, 1.0))
                 | (digit + delete("万") + add_weight(addzero**4, 1.0))) + insert(" ") + digit + union(*unit_sp_case1),
                -0.5,
            )
        else:
            measure_sp = add_weight(
                ((digit + delete("百") + add_weight(addzero**2, 1.0))
                 | (digit + delete("千") + add_weight(addzero**3, 1.0))
                 | (digit + delete("万") + add_weight(addzero**4, 1.0))) + digit_zh + union(*unit_sp_case1),
                -0.5,
            )

        # X块Y, X块Y毛: 十五块八 => 15块8, 三块五毛 => 3块5毛
        money_colloquial = (self.cardinal.number + accep("块") + digit + (accep("毛") | delete("角")).ques)

        self.value = measure | measure_sp | percent | money_colloquial
        self.denominator = delete("每") + units
        self.numerator = measure
        tagger = self.tag_field("value", self.value)
        # 每小时十千米 => 10km/h, 每小时三十到三百一十一千米 => 30~311km/h
        tagger |= (self.tag_field("denominator", self.denominator) + insert(" ") + self.tag_field("numerator", self.numerator))

        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        value = self.verbalize_field("value", self.value)
        self.verbalizer = self.delete_tokens(value)
        numerator = self.verbalize_field("numerator", self.numerator)
        denominator = self.verbalize_field("denominator", self.denominator, leading_space=True)
        verbalizer = numerator + insert("/") + denominator
        self.verbalizer |= self.delete_tokens(verbalizer)
