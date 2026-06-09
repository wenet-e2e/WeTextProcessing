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
from tn.utils import get_abs_path, load_labels


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

        currency_labels = load_labels(get_abs_path("../itn/english/data/currency.tsv"))
        currency_pairs = []
        for symbol, name in currency_labels:
            currency_pairs.append((name, symbol))
            if name.endswith("s"):
                currency_pairs.append((name + "es", symbol))
            else:
                currency_pairs.append((name + "s", symbol))
        currency = union(*[cross(name, symbol) for name, symbol in currency_pairs]).optimize()

        cent = cross("cent", "") | cross("cents", "")
        magnitudes = load_labels(get_abs_path("../itn/english/data/magnitudes.tsv"))
        magnitude = union(*[name for symbol, name in magnitudes if name != "thousand"])

        # "two dollars"
        # add "one fifty five" => "one hundred fifty five" => 155
        with_hundred = compose(
            closure(self.NOT_SPACE) + accep(" ") + insert("hundred ") + self.VSIGMA,
            compose(cardinal_graph, self.DIGIT ** 3),
        )
        cardinal_with_hundred = cardinal_graph | with_hundred
        integer_graph = (
            insert('value: "') + cardinal_with_hundred + insert('"')
            + ds + insert(' currency: "') + currency + insert('"')
        )
        # "fifty million dollars" / "four hundred billion won"
        quantity_graph = (
            insert('value: "') + cardinal_small + insert('"')
            + ds + insert(' quantity: "') + magnitude + insert('"')
            + ds + insert(' currency: "') + currency + insert('"')
        )
        # "two point five billion dollars"
        digit = string_file(get_abs_path("../itn/english/data/numbers/digit.tsv"))
        zero = string_file(get_abs_path("../itn/english/data/numbers/zero.tsv"))
        frac_d = digit | zero | cross("o", "0")
        frac = closure(frac_d + ds) + frac_d
        decimal_quantity_graph = (
            insert('value: "') + cardinal_graph + insert(".")
            + ds + delete("point") + ds + frac + insert('"')
            + ds + insert(' quantity: "') + magnitude + insert('"')
            + ds + insert(' currency: "') + currency + insert('"')
        )
        # "twenty point five o six dollars" (decimal without quantity)
        decimal_graph = (
            insert('value: "') + cardinal_graph + insert(".")
            + ds + delete("point") + ds + frac + insert('"')
            + ds + insert(' currency: "') + currency + insert('"')
        )
        # "point five o six dollars"
        decimal_no_int = (
            insert('value: ".') + delete("point") + ds + frac + insert('"')
            + ds + insert(' currency: "') + currency + insert('"')
        )
        # "one fifty five dollars" => $155 (missing "hundred")
        with_hundred = (
            insert('value: "') + cardinal_small + insert('"')
            + ds + insert(' currency: "') + currency + insert('"')
        )

        # cents
        cents_graph = union(*[cross(_num_to_word(x), f"{x:02d}") for x in range(1, 100) if _num_to_word(x)])
        with_cents = (
            insert('value: "') + cardinal_graph + insert('"')
            + ds + insert(' currency: "') + currency + insert('"')
            + ds + (delete("and") + ds).ques
            + insert(' decimal: "') + cents_graph + insert('"')
            + ds + cent
        )
        # "seventy five dollars sixty three" (no "cents" word)
        dollars_amount = (
            insert('value: "') + cardinal_graph + insert('"')
            + ds + insert(' currency: "') + currency + insert('"')
            + ds + insert(' decimal: "') + cents_graph + insert('"')
        )
        cents_only = (
            insert('currency: "$" decimal: "') + cents_graph + insert('"')
            + ds + cent
        )

        graph = (
            integer_graph | add_weight(quantity_graph, -1) | add_weight(decimal_quantity_graph, -1)
            | decimal_graph | decimal_no_int
            | with_cents | dollars_amount | cents_only
        )
        self.tagger = self.add_tokens(graph)

    def build_verbalizer(self):
        currency = delete('currency: "') + self.NOT_QUOTE.plus + delete('"')
        value = delete(' value: "') + self.NOT_QUOTE.plus + delete('"')
        decimal = delete(' decimal: "') + self.NOT_QUOTE.plus + delete('"')
        quantity = delete(' quantity: "') + self.NOT_QUOTE.plus + delete('"')

        graph = currency + value
        graph += closure(insert(".") + self.DELETE_SPACE + decimal, 0, 1)
        graph += closure(insert(" ") + self.DELETE_SPACE + quantity, 0, 1)
        graph |= currency + insert("0.") + self.DELETE_SPACE + decimal

        self.verbalizer = self.delete_tokens(graph)
