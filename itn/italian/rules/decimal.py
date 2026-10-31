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

from itn.italian.rules.utils import load_rows
from tn.processor import Processor

# Penalty per digit group, so ``trecento quarantadue`` reads as one group (342).
GROUP_WEIGHT = 0.1


class Decimal(Processor):
    """Decimals and quantities, e.g. ``quattro virgola cinquantasei milioni`` -> ``4,56 milioni``.

    The tagger keeps the spoken integer, the spoken fractional digit groups
    (dropping ``virgola``) and the quantity word; the verbalizer writes the
    digits around a decimal comma.
    """

    def __init__(self, cardinal):
        super().__init__("decimal", token_orders={})
        self.cardinal = cardinal
        self.integer = cardinal.number
        group = add_weight(cardinal.below_thousand, GROUP_WEIGHT)
        self.fractional = (group + (delete(" ") + group).star).optimize()
        self.quantity = pynini.union(*[row[0] for row in load_rows("italian/data/numbers/quantities.tsv")]).optimize()
        # Bare quantities count below one thousand, e.g. ``un milione`` -> ``1 milione``.
        self.quantity_integer = (cardinal.below_thousand | pynini.cross("un", "1")).optimize()

        quantity_tag = pynini.accep(" ") + self.tag_field("quantity", self.quantity)
        fraction_tag = self.tag_field("integer_part", self.integer) + pynini.cross(" virgola ", " ")
        fraction_tag += self.tag_field("fractional_part", self.fractional)
        whole_tag = self.tag_field("integer_part", self.quantity_integer) + quantity_tag
        self.fields_tagger = (fraction_tag + quantity_tag.ques | whole_tag).optimize()
        self.fraction_tagger = fraction_tag.optimize()
        sign_tag = (self.tag_field("negative", cardinal.sign) + pynini.accep(" ")).ques
        self.tagger = self.add_tokens(sign_tag + self.fields_tagger)

        quantity_verb = pynini.accep(" ") + self.verbalize_field("quantity", self.quantity)
        fraction_verb = self.verbalize_field("integer_part", self.integer) + delete(" ") + insert(",")
        fraction_verb += self.verbalize_field("fractional_part", self.fractional)
        whole_verb = self.verbalize_field("integer_part", self.quantity_integer) + quantity_verb
        self.fields_verbalizer = (fraction_verb + quantity_verb.ques | whole_verb).optimize()
        self.fraction_verbalizer = fraction_verb.optimize()
        sign_verb = (self.verbalize_field("negative", cardinal.sign) + delete(" ")).ques
        self.verbalizer = self.delete_tokens(sign_verb + self.fields_verbalizer)
