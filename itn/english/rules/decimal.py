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

from pynini import closure, cross, string_file, union
from pynini.lib.pynutil import delete, insert

from itn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


class Decimal(Processor):

    def __init__(self, cardinal=None):
        super().__init__(name="decimal", ordertype="itn")
        self.cardinal = cardinal or Cardinal()
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        digit = string_file(get_abs_path("../itn/english/data/numbers/digit.tsv"))
        zero = string_file(get_abs_path("../itn/english/data/numbers/zero.tsv"))
        ds = delete(" ")

        # fractional part: digit by digit, "o" => 0
        frac_digit = digit | zero | cross("o", "0")
        frac_graph = closure(frac_digit + ds) + frac_digit

        optional_negative = closure(
            insert('negative: "true" ') + delete("minus") + ds, 0, 1
        )
        integer_part = insert('integer_part: "') + self.cardinal.graph + insert('"')
        frac_part = insert('fractional_part: "') + frac_graph + insert('"')
        point = delete("point")

        graph = optional_negative + closure(integer_part + ds, 0, 1) + point + ds + frac_part

        # quantity: "five point two million" => 5.2 million
        quantities = load_labels(get_abs_path("../itn/english/data/numbers/thousands.tsv"))
        quantity_all = union(*[x[0] for x in quantities])
        quantity_no_thousand = union(*[x[0] for x in quantities if x[0] != "thousand"])
        # decimal + quantity: five point two million, 164.58 thousand
        quantity_graph = (
            optional_negative + integer_part + ds + point + ds + frac_part
            + ds + insert(' quantity: "') + quantity_all + insert('"')
        )
        # cardinal (up to 999) + quantity: four hundred million, five million
        # exclude thousand to let cardinal handle "ten thousand" => 10000
        cardinal_small = self.cardinal.up_to_999
        cardinal_quantity = (
            optional_negative + insert('integer_part: "') + cardinal_small + insert('"')
            + ds + insert(' quantity: "') + quantity_no_thousand + insert('"')
        )
        graph |= quantity_graph | cardinal_quantity

        self.tagger = self.add_tokens(graph)

    def build_verbalizer(self):
        optional_sign = closure(cross('negative: "true"', "-") + self.DELETE_SPACE, 0, 1)
        integer = delete('integer_part:') + self.DELETE_SPACE + delete('"') + self.NOT_QUOTE.plus + delete('"')
        optional_integer = closure(integer + self.DELETE_SPACE, 0, 1)
        fractional = (
            insert(".") + delete('fractional_part:') + self.DELETE_SPACE
            + delete('"') + self.NOT_QUOTE.plus + delete('"')
        )
        optional_fractional = closure(fractional + self.DELETE_SPACE, 0, 1)
        quantity = (
            insert(" ") + delete('quantity:') + self.DELETE_SPACE
            + delete('"') + self.NOT_QUOTE.plus + delete('"')
        )
        optional_quantity = closure(quantity + self.DELETE_SPACE, 0, 1)
        graph = optional_sign + optional_integer + optional_fractional + optional_quantity
        self.numbers = graph
        self.verbalizer = self.delete_tokens(graph)
