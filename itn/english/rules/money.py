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

from pynini import accep, closure, compose, cross, string_file, union
from pynini.lib.pynutil import add_weight, delete, insert

from itn.english.rules.cardinal import Cardinal
from itn.english.rules.decimal import Decimal
from itn.english.rules.time import _num_to_word
from tn.processor import Processor
from itn.utils import get_abs_path, load_labels


class Money(Processor):

    def __init__(self, cardinal=None, decimal=None):
        super().__init__(name="money", ordertype="itn")
        self.cardinal = cardinal or Cardinal()
        self.decimal = decimal or Decimal(cardinal=self.cardinal)
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        cardinal_graph = self.cardinal.graph
        cardinal_small = self.cardinal.up_to_999
        ds = delete(" ")

        currency_labels = load_labels(get_abs_path("english/data/currency.tsv"))
        singular_pairs = [(name, symbol) for symbol, name in currency_labels]
        plural_pairs = []
        for name, symbol in singular_pairs:
            if name.endswith("s"):
                plural_pairs.append((name + "es", symbol))
            else:
                plural_pairs.append((name + "s", symbol))
        currency_singular = union(*[cross(name, symbol) for name, symbol in singular_pairs]).optimize()
        currency_plural = union(*[cross(name, symbol) for name, symbol in singular_pairs + plural_pairs]).optimize()

        cent = cross("cent", "") | cross("cents", "")
        magnitudes = load_labels(get_abs_path("english/data/magnitudes.tsv"))
        magnitude = union(*[name for symbol, name in magnitudes if name != "thousand"])

        # "two dollars"
        # add "one fifty five" => "one hundred fifty five" => 155
        with_hundred = compose(
            closure(self.NOT_SPACE) + accep(" ") + insert("hundred ") + self.VSIGMA,
            compose(cardinal_graph, self.DIGIT**3),
        )
        cardinal_with_hundred = cardinal_graph | with_hundred
        not_one = self.DIGIT**(2, ...) | (self.DIGIT - accep("1"))
        cardinal_plural = compose(cardinal_with_hundred, not_one)
        # "one dollar" (singular) vs "two dollars" (plural)
        one = cross("one", "1")
        integer_graph = (self.tag_field("value", cardinal_plural) + ds + insert(" ") +
                         self.tag_field("currency", currency_plural))
        integer_graph |= (self.tag_field("value", one) + ds + insert(" ") + self.tag_field("currency", currency_singular))
        # "fifty million dollars" / "four hundred billion won"
        quantity_graph = (self.tag_field("value", cardinal_small) + ds + insert(" ") + self.tag_field("quantity", magnitude) +
                          ds + insert(" ") + self.tag_field("currency", currency_plural))
        # "two point five billion dollars"
        digit = string_file(get_abs_path("english/data/numbers/digit.tsv"))
        zero = string_file(get_abs_path("english/data/numbers/zero.tsv"))
        frac_d = digit | zero | cross("o", "0")
        frac = closure(frac_d + ds) + frac_d
        decimal_value = cardinal_graph + ds + delete("point") + ds + insert(".") + frac
        decimal_quantity_graph = (self.tag_field("value", decimal_value) + ds + insert(" ") +
                                  self.tag_field("quantity", magnitude) + ds + insert(" ") +
                                  self.tag_field("currency", currency_plural))
        # "twenty point five o six dollars" (decimal without quantity)
        decimal_graph = (self.tag_field("value", decimal_value) + ds + insert(" ") +
                         self.tag_field("currency", currency_plural))
        # "point five o six dollars"
        decimal_no_int_value = insert(".") + delete("point") + ds + frac
        decimal_no_int = (self.tag_field("value", decimal_no_int_value) + ds + insert(" ") +
                          self.tag_field("currency", currency_plural))
        # "one fifty five dollars" => $155 (missing "hundred")
        # cents
        cents_graph = union(*[cross(_num_to_word(x), f"{x:02d}") for x in range(1, 100) if _num_to_word(x)])
        decimal_with_cents = ((delete("and") + ds).ques + cents_graph + ds + cent)
        with_cents = (self.tag_field("value", cardinal_graph) + ds + insert(" ") +
                      self.tag_field("currency", currency_plural) + ds + insert(" ") +
                      self.tag_field("decimal", decimal_with_cents))
        # "seventy five dollars sixty three" (no "cents" word)
        dollars_amount = (self.tag_field("value", cardinal_graph) + ds + insert(" ") +
                          self.tag_field("currency", currency_plural) + ds + insert(" ") +
                          self.tag_field("decimal", cents_graph))
        cents_only_decimal = cents_graph + ds + cent
        cents_only = (self.tag_field("currency", insert("$")) + insert(" ") + self.tag_field("decimal", cents_only_decimal))

        self.value = (cardinal_plural | one | cardinal_small | decimal_value | decimal_no_int_value | cardinal_graph)
        self.currency = currency_plural | currency_singular | insert("$")
        self.decimal = decimal_with_cents | cents_graph | cents_only_decimal
        self.quantity = magnitude
        graph = (integer_graph | add_weight(quantity_graph, -1) | add_weight(decimal_quantity_graph, -1)
                 | decimal_graph | decimal_no_int
                 | with_cents | dollars_amount | cents_only)
        self.tagger = self.add_tokens(graph)

    def build_verbalizer(self):
        currency = self.verbalize_field("currency", self.currency)
        value = self.verbalize_field("value", self.value, leading_space=True)
        decimal = self.verbalize_field("decimal", self.decimal, leading_space=True)
        quantity = self.verbalize_field("quantity", self.quantity, leading_space=True)

        graph = currency + value
        graph += closure(insert(".") + self.DELETE_SPACE + decimal, 0, 1)
        graph += closure(insert(" ") + self.DELETE_SPACE + quantity, 0, 1)
        graph |= currency + insert("0.") + self.DELETE_SPACE + decimal

        self.verbalizer = self.delete_tokens(graph)
