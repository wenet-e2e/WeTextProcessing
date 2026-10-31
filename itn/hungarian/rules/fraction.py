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
from pynini.lib.pynutil import delete

from tn.processor import Processor

TOKEN_ORDER = {}


class Fraction(Processor):
    """Fractions, e.g. ``kettő harmad`` -> ``2/3``, ``egy és egy fél`` -> ``1 1/2``."""

    def __init__(self, cardinal, ordinal):
        super().__init__("fraction", ordertype="itn", token_orders=TOKEN_ORDER)
        numerator = cardinal.graph
        denominator = ordinal.fractional

        integer_t = self.tag_field("integer_part", cardinal.graph) + delete(" és") + pynini.accep(" ")
        fraction_t = (self.tag_field("numerator", numerator) + pynini.accep(" ")
                      + self.tag_field("denominator", denominator))
        self.unsigned_tagger = (integer_t.ques + fraction_t).optimize()
        self.tagger = self.add_tokens(cardinal.sign_tagger + self.unsigned_tagger)

        integer_v = self.verbalize_field("integer_part", cardinal.graph) + pynini.accep(" ")
        fraction_v = (self.verbalize_field("numerator", numerator) + pynini.cross(" ", "/")
                      + self.verbalize_field("denominator", denominator))
        self.unsigned_verbalizer = (integer_v.ques + fraction_v).optimize()
        self.verbalizer = self.delete_tokens(cardinal.sign_verbalizer + self.unsigned_verbalizer)
