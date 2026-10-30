# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026 Zhendong Peng (pzd17@tsinghua.org.cn)
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

from pynini import cross, union
from pynini.lib.pynutil import add_weight, delete, insert

from itn.korean.rules.utils import josa_graph
from tn.processor import Processor


class Fraction(Processor):
    """Keep operands as independent raw fields; the parser swaps their order."""

    def __init__(self, cardinal, decimal):
        super().__init__(
            name="fraction", ordertype="itn",
            token_orders={"fraction": ["negative", "integer_part", "numerator", "denominator", "suffix"]},
        )
        number = decimal.just_decimal
        root = union("√") | cross("루트", "√")
        expression = (number | root + number | cardinal.just_cardinals + root + number).optimize()
        mixed = expression + delete(union("와", "과")).ques
        sign = (self.tag_field("negative", cardinal.negative) + insert(" ")).ques
        integer = add_weight(self.tag_field("integer_part", mixed), 1.1) + insert(" ")
        denominator = self.tag_field("denominator", expression) + delete(" ").ques
        numerator = self.tag_field("numerator", expression) + delete(" ").ques
        suffix = (delete(" ").ques + insert(" ") + self.tag_field("suffix", josa_graph())).ques
        fields = sign + integer.ques + denominator + delete("분의") + insert(" ") + numerator + suffix
        self.tagger = self.add_tokens(fields)
        sign_out = (self.verbalize_field("negative", cardinal.negative) + delete(" ")).ques
        integer_out = (self.verbalize_field("integer_part", mixed) + delete(" ") + insert(" ")).ques
        # TOKEN_ORDERS moves numerator ahead of denominator, for unbounded decimals.
        fields_out = self.verbalize_field("numerator", expression) + delete(" ") + insert("/")
        fields_out += self.verbalize_field("denominator", expression)
        fields_out += (delete(" ") + insert(" ") + self.verbalize_field("suffix", josa_graph())).ques
        self.verbalizer = self.delete_tokens(sign_out + integer_out + fields_out)
