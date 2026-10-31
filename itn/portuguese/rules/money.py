# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from itn.utils import get_abs_path
from tn.processor import Processor


TOKEN_ORDER = {
    "money": ["currency", "integer_part", "point", "fractional_part", "quantity", "cents", "currency_min"],
}


class Money(Processor):
    """Raw spoken amounts and currency names, e.g. "doze dólares e cinco centavos" -> $ 12,05.

    Cents without a major currency default to reais: "vinte centavos" -> R$ 0,20.
    """

    def __init__(self, cardinal, decimal):
        super().__init__("money", ordertype="itn", token_orders=TOKEN_ORDER)
        singular = pynini.string_file(get_abs_path("portuguese/data/currency_singular.tsv")).invert()
        plural = pynini.string_file(get_abs_path("portuguese/data/currency_plural.tsv")).invert()
        currency = (singular | plural).optimize()
        number = cardinal.graph_no_exception
        one = pynini.cross("um", "1") | pynini.cross("uma", "1")
        other = pynini.difference(self.VSIGMA, pynini.union("um", "uma")) @ number
        pad = self.DIGIT**2 | insert("0") + self.DIGIT
        cents_one = pynini.cross("um", "01")
        cents_other = add_weight(other @ pad, -0.7)
        cents_suffix = add_weight(number @ pad, -0.7)
        self.cents_graph = (cents_one | cents_other | cents_suffix).optimize()
        minor_one = pynini.accep("centavo")
        minor_other = pynini.accep("centavos")
        connector = (delete(pynini.union("com", "e")) + delete(self.SPACE.plus)).ques
        space = self.DELETE_SPACE
        gap = delete(self.SPACE.plus) + insert(" ")

        def field(name, graph):
            return self.tag_field(name, graph)

        cents_standalone = (
            field("cents", cents_other) + gap + field("currency_min", minor_other)
            | field("cents", cents_one) + gap + field("currency_min", minor_one)
        )
        optional_cents = (
            gap + connector + cents_standalone
            | gap + connector + field("cents", cents_suffix)
        ).ques
        integer = (
            field("integer_part", other) + gap + field("currency", plural)
            | field("integer_part", one) + gap + field("currency", singular)
        ) + optional_cents
        cents_only = (
            cents_standalone
            | add_weight(cents_standalone + gap + delete("de") + space + field("currency", singular), -0.1)
        )
        decimal_amount = (
            decimal.unsigned_tagger + gap + (delete("de") + delete(self.SPACE.plus)).ques + field("currency", plural)
        )
        self.tagger = self.add_tokens(integer | cents_only | decimal_amount)

        currency_v = self.verbalize_field("currency", currency) + delete(" ") + insert(" ")
        cents_v = (
            insert(",") + self.verbalize_field("cents", self.cents_graph, leading_space=True)
            + self.verbalize_field("currency_min", delete(minor_one | minor_other), leading_space=True).ques
        )
        integer_v = currency_v + self.verbalize_field("integer_part", one | other) + cents_v.ques
        cents_only_v = (
            (currency_v | insert("R$ ")) + insert("0")
            + insert(",") + self.verbalize_field("cents", self.cents_graph)
            + self.verbalize_field("currency_min", delete(minor_one | minor_other), leading_space=True)
        )
        decimal_v = currency_v + decimal.unsigned_verbalizer
        self.verbalizer = self.delete_tokens(integer_v | cents_only_v | decimal_v)
