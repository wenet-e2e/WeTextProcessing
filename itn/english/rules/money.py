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

from pynini import closure, cross, union
from pynini.lib.pynutil import delete, insert

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
        magnitude = union(*[cross(name, "") for symbol, name in magnitudes])

        integer_graph = (
            insert('value: "') + cardinal_graph + insert('"')
            + ds + insert(' currency: "') + currency + insert('"')
        )
        quantity_graph = (
            insert('value: "') + cardinal_graph + insert('"')
            + ds + insert(' quantity: "') + magnitude + insert('"')
            + ds + insert(' currency: "') + currency + insert('"')
        )
        # cents: pad single digit (1-9 => 01-09)
        cents_graph = union(*[cross(_num_to_word(x), f"{x:02d}") for x in range(1, 100) if _num_to_word(x)])
        with_cents = (
            insert('value: "') + cardinal_graph + insert('"')
            + ds + insert(' currency: "') + currency + insert('"')
            + ds + (delete("and") + ds).ques
            + insert(' decimal: "') + cents_graph + insert('"')
            + ds + cent
        )
        cents_only = (
            insert('currency: "$" decimal: "') + cents_graph + insert('"')
            + ds + cent
        )

        graph = integer_graph | quantity_graph | with_cents | cents_only
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
