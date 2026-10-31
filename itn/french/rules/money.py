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
from pynini.lib.pynutil import delete, insert

from itn.french.rules.cardinal import load
from itn.french.rules.decimal import WHITE_SPACE
from tn.processor import Processor


TOKEN_ORDER = {"money": ["integer_part", "fractional_part", "quantity", "currency", "currency_minor"]}


class Money(Processor):
    """Amounts and currency names stay verbatim in the tag.

    ``deux euros et vingt centimes`` -> ``money { integer_part: "deux" currency: "euros"
    fractional_part: "vingt" currency_minor: "centimes" }`` -> ``2,20 €``. Minor units
    alone (``cinq centimes`` -> ``0,05 €``) and round amounts introduced by ``de``
    (``deux millions de dollars`` -> ``2 millions de dollars``) are covered too.
    """

    def __init__(self, cardinal, decimal):
        super().__init__("money", ordertype="itn", token_orders=TOKEN_ORDER)
        number = cardinal.graph_no_exception
        major = load("money/currency_major")
        minor = load("money/currency_minor")
        minor_names = self.input_projection(minor)
        names = self.input_projection(major | minor)
        round_currency = (pynini.union("de ", "des ", "d'") + names).optimize()
        cents = (number @ (self.DIGIT**2 | insert("0") + self.DIGIT)).optimize()
        delete_space = delete(WHITE_SPACE.star)
        delete_extra_space = delete(WHITE_SPACE.plus)

        def field(name, graph, separator=delete_space):
            return separator + insert(" ") + self.tag_field(name, graph)

        major_tagger = self.tag_field("integer_part", number) + field("currency", major)
        major_tagger += (delete_extra_space + delete("et")).ques + field("fractional_part", cents).ques
        major_tagger += field("currency_minor", minor_names, delete_extra_space).ques
        minor_tagger = self.tag_field("fractional_part", cents) + field("currency", minor, delete_extra_space)
        round_tagger = decimal.number_tagger + field("currency", round_currency)
        self.tagger = self.add_tokens(major_tagger | minor_tagger | round_tagger)

        def currency(graph):
            return insert(" ") + self.verbalize_field("currency", graph, leading_space=True)

        cents_out = insert(",") + self.verbalize_field("fractional_part", cents, leading_space=True)
        major_verbalizer = self.verbalize_field("integer_part", number @ decimal.group_integer) + cents_out.ques
        major_verbalizer += currency(major)
        major_verbalizer += self.verbalize_field("currency_minor", delete(minor_names), leading_space=True).ques
        minor_verbalizer = insert("0,") + self.verbalize_field("fractional_part", cents) + currency(minor)
        round_verbalizer = decimal.number_verbalizer + currency(round_currency)
        self.verbalizer = self.delete_tokens(major_verbalizer | minor_verbalizer | round_verbalizer)
