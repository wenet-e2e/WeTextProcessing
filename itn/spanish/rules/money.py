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
from pynini.lib import pynutil

from itn.spanish.rules.utils import DIGIT, SIGMA, cased, data_labels, restrict
from tn.processor import Processor

TOKEN_ORDER = {
    "money": [
        "currency", "minor_currency", "integer_part", "point", "fractional_part", "quantity", "cents",
    ],
}


def _currencies(*names):
    """Spoken currency names to symbols; a later table row wins on ties."""
    mapping = {}
    for name in names:
        for symbol, spoken in data_labels("money/" + name + ".tsv"):
            mapping[spoken] = symbol
    return pynini.string_map(sorted(mapping.items())).optimize()


class Money(Processor):
    """Raw spoken amounts and currency names; V writes "$12,05".

    The connectors "y"/"con" before the cents and "de" before the currency
    are not part of any field.
    """

    def __init__(self, cardinal, decimal, cased_input=False):
        super().__init__("money", ordertype="itn", token_orders=TOKEN_ORDER)
        singular = cased(_currencies("currency_major_singular"), cased_input)
        plural = cased(_currencies("currency_major_plural"), cased_input)
        self.currency_graph = (singular | plural).optimize()
        minor = cased(_currencies("currency_minor_singular", "currency_minor_plural"), cased_input)
        self.minor_graph = minor

        numbers = cardinal.graph_no_exception
        one = cased(pynini.cross(pynini.union("un", "una"), "1"), cased_input)
        others = restrict(numbers, one)
        self.integer_graph = (one | others).optimize()
        two_digits = DIGIT + DIGIT | pynutil.insert("0") + DIGIT
        # The cardinal accepts a leading "y"; here it is a connector instead.
        no_connector = pynini.difference(SIGMA, pynini.union("y", "Y") + " " + SIGMA)
        cents_number = (no_connector @ others @ two_digits).optimize()
        self.cents_graph = (cents_number | (one @ pynini.cross("1", "01"))).optimize()

        space = pynini.accep(" ")
        conjunction = cased(pynini.union("y", "con"), cased_input)
        con = cased(pynini.accep("con"), cased_input)
        currency_t = self.tag_field("currency", self.currency_graph)
        minor_t = self.tag_field("minor_currency", self.minor_graph)
        cents_t = self.tag_field("cents", pynutil.add_weight(cents_number, -0.7) | (one @ pynini.cross("1", "01")))
        standalone = (
            space + pynini.closure(pynutil.delete(conjunction + " "), 0, 1) + cents_t + space + minor_t
        )
        suffix = (
            space + pynini.closure(pynutil.delete(con + " "), 0, 1)
            + self.tag_field("cents", pynutil.add_weight(cents_number, -0.7))
        )
        cents_optional = pynini.closure(standalone | suffix, 0, 1)
        graph = (
            self.tag_field("integer_part", others) + space + self.tag_field("currency", plural)
            | self.tag_field("integer_part", one) + space + self.tag_field("currency", singular)
        ) + cents_optional
        graph |= cents_t + space + minor_t
        of = pynini.closure(pynutil.delete(cased(pynini.accep("de"), cased_input) + " "), 0, 1)
        graph |= decimal.unsigned_tagger + space + of + self.tag_field("currency", plural)
        self.tagger = self.add_tokens(graph)

        delete_space = pynutil.delete(" ")
        currency_v = self.verbalize_field("currency", self.currency_graph)
        minor_v = self.verbalize_field("minor_currency", self.minor_graph)
        integer_v = delete_space + self.verbalize_field("integer_part", self.integer_graph)
        cents_v = pynutil.insert(",") + delete_space + self.verbalize_field("cents", self.cents_graph)
        graph = currency_v + integer_v + pynini.closure(cents_v, 0, 1)
        graph |= currency_v + delete_space + pynutil.delete(pynini.project(minor_v, "input")) + integer_v + cents_v
        graph |= minor_v + pynutil.insert("0") + cents_v
        graph |= currency_v + delete_space + decimal.unsigned_verbalizer
        self.verbalizer = self.delete_tokens(graph)
