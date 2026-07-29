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
from pynini.lib.pynutil import add_weight, delete, insert

from itn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from itn.utils import get_abs_path, load_labels


class Decimal(Processor):

    def __init__(self, cardinal=None):
        super().__init__(name="decimal", ordertype="itn")
        self.cardinal = cardinal or Cardinal()
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        digit = string_file(get_abs_path("english/data/numbers/digit.tsv"))
        zero = string_file(get_abs_path("english/data/numbers/zero.tsv"))
        ds = delete(" ")

        # fractional part: digit by digit, "o" => 0
        frac_digit = digit | zero | cross("o", "0")
        frac_graph = closure(frac_digit + ds) + frac_digit

        self.negative = cross("minus", "-")
        self.integer = self.cardinal.graph
        self.fractional = delete("point") + ds + frac_graph
        optional_negative = closure(self.tag_field("negative", self.negative) + ds + insert(" "), 0, 1)
        integer_part = self.tag_field("integer_part", self.integer)
        frac_part = self.tag_field("fractional_part", self.fractional)

        graph = optional_negative + closure(integer_part + ds + insert(" "), 0, 1) + frac_part

        # quantity: "five point two million" => 5.2 million
        quantities = load_labels(get_abs_path("english/data/numbers/thousands.tsv"))
        quantity_all = union(*[x[0] for x in quantities])
        quantity_no_thousand = union(*[x[0] for x in quantities if x[0] != "thousand"])
        # decimal + quantity: five point two million, 164.58 thousand
        self.quantity = quantity_all
        quantity_graph = (optional_negative + integer_part + ds + insert(" ") + frac_part + ds + insert(" ") +
                          self.tag_field("quantity", self.quantity))
        # cardinal (up to 999) + quantity: four hundred million, five million
        # exclude thousand to let cardinal handle "ten thousand" => 10000
        cardinal_small = self.cardinal.up_to_999
        cardinal_quantity = (optional_negative + self.tag_field("integer_part", cardinal_small) + ds + insert(" ") +
                             self.tag_field("quantity", quantity_no_thousand))
        # Prefer quantity structure over expanding the same magnitude as a bare
        # cardinal. The classification weight is outside every raw field.
        graph |= add_weight(quantity_graph | cardinal_quantity, -0.001)

        self.tagger = self.add_tokens(graph)

    def build_verbalizer(self):
        optional_sign = closure(
            self.verbalize_field("negative", self.negative) + self.DELETE_SPACE,
            0,
            1,
        )
        integer = self.verbalize_field("integer_part", self.integer | self.cardinal.up_to_999)
        optional_integer = closure(integer + self.DELETE_SPACE, 0, 1)
        fractional = insert(".") + self.verbalize_field("fractional_part", self.fractional)
        optional_fractional = closure(fractional + self.DELETE_SPACE, 0, 1)
        quantity = insert(" ") + self.verbalize_field("quantity", self.quantity)
        optional_quantity = closure(quantity + self.DELETE_SPACE, 0, 1)
        graph = optional_sign + optional_integer + optional_fractional + optional_quantity
        self.numbers = graph
        self.verbalizer = self.delete_tokens(graph)
