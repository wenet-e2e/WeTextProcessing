# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from itn.hindi.rules.utils import DEDH, DELETE_EXTRA_SPACE, DELETE_SPACE, spoken_to_written
from tn.processor import Processor

# The written currency symbol precedes the amount, while the spoken currency follows it.
TOKEN_ORDER = {"money": ["currency", "value", "fractional_part", "minor_currency"]}

LAKH = "लाख"
CRORE = "करोड़"


class Money(Processor):
    """``बारह रुपये पचास पैसे`` ->
    ``money { value: "बारह" currency: "रुपये" fractional_part: "पचास" minor_currency: "पैसे" }`` -> ``₹१२.५०``.
    """

    def __init__(self, cardinal, decimal):
        super().__init__("money", ordertype="itn", token_orders=TOKEN_ORDER)
        number = cardinal.graph_no_exception
        small = cardinal.digit | cardinal.teens_and_ties
        paune = cardinal.paune
        currency = spoken_to_written("money/currency.tsv")
        spoken_currency = currency.copy().project("input").optimize()

        def scaled(prefix, integer, digits, scale, weight=None):
            graph = (delete(prefix) + DELETE_SPACE + integer + DELETE_SPACE + insert(digits, weight=-0.1)
                     + DELETE_SPACE + delete(scale))
            return graph if weight is None else add_weight(graph, weight)

        self.amount = pynini.union(
            decimal.graph_wo_negative,
            number,
            delete("साढ़े") + DELETE_SPACE + small + DELETE_SPACE + insert(".५०"),
            delete("सवा") + DELETE_SPACE + small + DELETE_SPACE + insert(".२५"),
            delete(DEDH) + DELETE_SPACE + insert("१") + DELETE_SPACE + insert(".५०"),
            delete("ढाई") + DELETE_SPACE + insert("२") + DELETE_SPACE + insert(".५०"),
            scaled("साढ़े", small, "५००००", LAKH, 0.01),
            scaled("सवा", small, "२५०००", LAKH, 0.01),
            scaled("पौने", paune, "७५०००", LAKH),
            scaled(DEDH, insert(""), "१५००००", LAKH),
            scaled("ढाई", insert(""), "२५००००", LAKH),
            scaled("साढ़े", small, "५००००००", CRORE),
            scaled("सवा", small, "२५०००००", CRORE),
            scaled("पौने", paune, "७५०००००", CRORE),
            scaled("ढाई", insert(""), "२५००००००", CRORE),
            scaled(DEDH, insert(""), "१५००००००", CRORE),
        ).optimize()

        def field(name, graph):
            return DELETE_EXTRA_SPACE + insert(" ") + self.tag_field(name, graph)

        currency_field = field("currency", spoken_currency)
        simple = self.tag_field("value", self.amount) + currency_field
        # "बारह रुपये और पचास पैसे": the optional "और" is dropped, the minor currency is not read.
        paise = (self.tag_field("value", number) + currency_field + (DELETE_EXTRA_SPACE + delete("और")).ques
                 + field("fractional_part", number) + field("minor_currency", spoken_currency))
        self.tagger = self.add_tokens(simple | paise)

        symbol = self.verbalize_field("currency", currency) + delete(" ")
        simple = symbol + self.verbalize_field("value", self.amount)
        paise = (symbol + self.verbalize_field("value", number) + insert(".")
                 + self.verbalize_field("fractional_part", number, leading_space=True)
                 + self.verbalize_field("minor_currency", delete(spoken_currency), leading_space=True))
        self.verbalizer = self.delete_tokens(simple | paise)
