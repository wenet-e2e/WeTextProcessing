# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pynini
from pynini.lib.pynutil import add_weight, delete, insert

from itn.french.rules.cardinal import load
from itn.french.rules.decimal import WHITE_SPACE
from tn.processor import Processor


TOKEN_ORDER = {
    "measure": [
        "negative", "integer", "integer_part", "fractional_part", "quantity", "numerator", "denominator", "units",
    ],
}


class Measure(Processor):
    """``moins soixante-six kilogrammes`` -> ``measure { negative: "moins" integer: "soixante-six"
    units: "kilogrammes" }`` -> ``-66 kg``. Values may also be decimals or fractions."""

    def __init__(self, cardinal, decimal, fraction):
        super().__init__("measure", ordertype="itn", token_orders=TOKEN_ORDER)
        delete_space = delete(WHITE_SPACE.star)
        singular = load("measurements/measurements")
        unit = load("measurements/magnitudes").ques + (singular | (self.VSIGMA + delete("s")) @ singular)
        per = insert("/") + delete(pynini.union("par", "à")) + delete_space + unit
        units = pynini.union(unit, per, add_weight(unit + delete_space + per, 0.01)).optimize()
        number = cardinal.graph_no_exception

        values = [
            (self.tag_field("integer", number), self.verbalize_field("integer", number @ decimal.group_integer)),
            (decimal.number_tagger, decimal.number_verbalizer),
            (fraction.number_tagger, fraction.number_verbalizer),
        ]
        sign_tagger = decimal.sign_tagger.ques
        sign_verbalizer = decimal.sign_verbalizer.ques
        units_tagger = delete(WHITE_SPACE.plus) + insert(" ") + self.tag_field("units", units)
        units_verbalizer = pynini.accep(" ") + self.verbalize_field("units", units)
        self.tagger = self.add_tokens(sign_tagger + pynini.union(*[tagger for tagger, _ in values]) + units_tagger)
        self.verbalizer = self.delete_tokens(
            sign_verbalizer + pynini.union(*[verbalizer for _, verbalizer in values]) + units_verbalizer)
