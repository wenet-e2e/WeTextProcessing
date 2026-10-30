# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


class Money(Processor):
    """Amounts with currency symbols or words (``10,5$``, ``12.345 đồng``, ``50.000đ/ngày``).

    A currency glued to the amount is tagged as ``currency_maj`` and may carry
    minor units (``10,5$`` -> "mười đô la năm mươi xu"); a currency written as a
    separate word is tagged as ``currency`` and keeps the decimal reading
    (``10,5 đồng`` -> "mười phẩy năm đồng").
    """

    def __init__(self, cardinal, decimal):
        super().__init__("money", token_orders={})
        major_labels = load_labels(get_abs_path("vietnamese/data/money/currency.tsv"))
        minor_names = dict(load_labels(get_abs_path("vietnamese/data/money/currency_minor.tsv")))
        abbreviations = pynini.string_file(get_abs_path("vietnamese/data/numbers/quantity_abbr.tsv"))
        self.currency_graph = pynini.string_map(major_labels).optimize()
        self.per_unit_graph = self._per_unit_graph()

        nonzero = self.DIGIT - "0"
        strip = pynini.closure(self.DIGIT) + nonzero + pynini.closure(delete("0"))
        pad = delete("0") + nonzero | nonzero + insert("0") | nonzero + self.DIGIT
        self.cents_graph = ((strip @ pad) @ cardinal.graph).optimize()
        positive = pynini.difference((self.DIGIT | ".").plus, "0") @ cardinal.graph

        sep = insert(" ")
        integer = self.tag_field("integer_part", cardinal.raw)
        cents = delete(",") + sep + self.tag_field("fractional_part", self.cents_graph)
        majors = {}
        for symbol, name in major_labels:
            majors.setdefault(symbol, name)
        minor_symbols = [symbol for symbol in majors if symbol in minor_names]
        glued = integer + sep + self.tag_field("currency_maj", self.currency_graph)
        glued |= integer + cents + sep + self.tag_field("currency_maj", pynini.union(*minor_symbols))

        space = delete(self.SPACE.star)
        currency = space + sep + self.tag_field("currency", self.currency_graph)
        quantity = space + sep + self.tag_field("quantity", abbreviations)
        spaced = (decimal.with_integer_tagger | integer | integer + quantity) + currency
        per_unit = sep + self.tag_field("per_unit", self.per_unit_graph)
        self.fields_tagger = ((glued | add_weight(spaced, 0.1)) + per_unit.ques).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)

        integer_v = self.verbalize_field("integer_part", cardinal.graph)
        glued_v = integer_v + insert(" ") + self.verbalize_field("currency_maj", self.currency_graph, True)
        cents_v = self.verbalize_field("fractional_part", self.cents_graph, True)
        # The glued symbol comes last, so each branch speaks the names of the symbol it then consumes.
        minor_only, major_minor = [], []
        for symbol in minor_symbols:
            minor = insert(" " + minor_names[symbol]) + self.verbalize_field("currency_maj", delete(symbol), True)
            minor_only.append(cents_v + minor)
            major_minor.append(insert(" " + majors[symbol] + " ") + cents_v + minor)
        glued_v |= self.verbalize_field("integer_part", delete("0")) + pynini.union(*minor_only)
        glued_v |= self.verbalize_field("integer_part", positive) + pynini.union(*major_minor)
        number_v = decimal.with_integer_verbalizer | self.verbalize_field("integer_part", decimal.integer_graph)
        spaced_v = number_v + insert(" ") + self.verbalize_field("currency", self.currency_graph, True)
        per_unit_v = insert(" ") + self.verbalize_field("per_unit", self.per_unit_graph, True)
        self.fields_verbalizer = ((glued_v | spaced_v) + per_unit_v.ques).optimize()
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)

    @staticmethod
    def _per_unit_graph():
        prefixes = pynini.string_file(get_abs_path("vietnamese/data/money/per_unit_prefixes.tsv"))
        bases = pynini.string_file(get_abs_path("vietnamese/data/money/per_unit_bases.tsv"))
        others = pynini.string_file(get_abs_path("vietnamese/data/money/per_unit_non_metric.tsv"))
        one = delete("/") + insert("một ")
        return (one + prefixes + insert(" ") + bases | one + bases | others).optimize()
