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

from pynini import string_file
from pynini.lib.pynutil import add_weight, delete, insert

from itn.chinese.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Fraction(Processor):

    def __init__(self):
        super().__init__(name="fraction")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        number = Cardinal().number
        sign = string_file(get_abs_path("../itn/chinese/data/number/sign.tsv"))  # + -

        # NOTE(xcsong): default weight = 1.0,  set to -1.0 means higher priority
        #   For example,
        #       1.0, 负二分之三 -> { sign: "" denominator: "-2" numerator: "3" }
        #       -1.0,负二分之三 -> { sign: "-" denominator: "2" numerator: "3" }
        tagger = (
            insert('sign: "')
            + add_weight(sign, -1.0).ques
            + insert('" denominator: "')
            + number
            + delete("分之")
            + insert('" numerator: "')
            + number
            + insert('"')
        )
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        sign = delete('sign: "') + self.SIGMA + delete('"')
        numerator = delete(' numerator: "') + self.SIGMA + delete('"')
        denominator = delete(' denominator: "') + self.SIGMA + delete('"')
        verbalizer = sign + numerator + insert("/") + denominator
        self.verbalizer = self.delete_tokens(verbalizer)
