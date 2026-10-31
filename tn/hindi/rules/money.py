# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from tn.hindi.rules.utils import ALL_DIGIT, ALL_ZERO, hindi_path
from tn.processor import Processor
from tn.utils import load_labels

TOKEN_ORDER = {"money": ["integer_part", "currency_maj", "fractional_part"]}


class Money(Processor):
    """``₹५०.५०`` -> ``money { currency_maj: "₹" integer_part: "५०" fractional_part: "५०" }``.

    The verbalizer chooses between rupees/paise, singular and decimal readings from the raw fields.
    """

    def __init__(self, cardinal):
        super().__init__("money", token_orders=TOKEN_ORDER)
        number = cardinal.final_graph
        one = pynini.union("1", "१") @ number
        ascii_digit = pynini.union(*"0123456789")
        ascii_nonzero = pynini.union(*"123456789")
        to_ascii = pynini.invert(pynini.string_file(hindi_path("ordinal/en_to_hi_digit.tsv")))
        to_ascii = (to_ascii | ascii_digit).plus
        strip_zeros = ascii_digit.star + ascii_nonzero + delete("0").star
        canonical = (delete("0") + ascii_nonzero) | (ascii_nonzero + insert("0")) | (ascii_nonzero + ascii_digit)
        minor = (to_ascii @ strip_zeros @ canonical @ number).optimize()
        nonzero = pynini.union(ascii_nonzero, *"१२३४५६७८९")
        digits = ALL_DIGIT @ cardinal.single_digits_graph
        spelled = ((ALL_DIGIT + ALL_DIGIT + nonzero + ALL_DIGIT.star) @ (digits + (insert(" ") + digits).star))
        zeros = delete(ALL_ZERO.plus)

        plural = load_labels(hindi_path("money/currency.tsv"))
        singular = dict(load_labels(hindi_path("money/currency_singular.tsv")))
        minor_names = dict(load_labels(hindi_path("money/major_minor_currencies.tsv")))

        # Tag each field graph once; the per-currency patterns only concatenate them.
        tagged = {name: self.tag_field("integer_part", graph) for name, graph in (("one", one), ("number", number))}
        tagged.update({name: delete(".") + insert(" ") + self.tag_field("fractional_part", graph)
                       for name, graph in (("zeros", zeros), ("spelled", spelled), ("minor", minor))})

        def fields(symbol, integer, fraction=None):
            tagger = self.tag_field("currency_maj", pynini.accep(symbol)) + insert(" ") + tagged[integer]
            return tagger if fraction is None else tagger + tagged[fraction]

        def currency(symbol, before, fraction=None, after=""):
            # Reads the currency field (and the optional fraction field) that follow the integer.
            verbalizer = self.verbalize_field("currency_maj", delete(symbol), leading_space=True) + insert(before)
            if fraction is not None:
                verbalizer += self.verbalize_field("fractional_part", fraction, leading_space=True)
            return verbalizer + insert(after)

        # (integer reading, fraction reading, tagger weight) of every NeMo pattern; the
        # verbalizer below factors the shared integer reading out of the currency branches.
        taggers, plural_branches, singular_branches, minor_only = [], [], [], []
        for symbol, major in plural:
            major_one = singular[symbol]
            taggers += [
                add_weight(fields(symbol, "one"), -0.001),
                add_weight(fields(symbol, "one", "zeros"), -0.001),
                fields(symbol, "number"),
                fields(symbol, "number", "zeros"),
                add_weight(fields(symbol, "number", "spelled"), -0.0005),
            ]
            singular_branches += [currency(symbol, " " + major_one), currency(symbol, " " + major_one, zeros)]
            plural_branches += [
                currency(symbol, " " + major),
                currency(symbol, " " + major, zeros),
                add_weight(currency(symbol, " दशमलव ", spelled, " " + major), -0.0005),
            ]
            for name, integer, branches in ((major_one, "one", singular_branches), (major, "number", plural_branches)):
                if name in minor_names:
                    weight = -0.001 if integer == "one" else 0
                    cents = minor + insert(" " + minor_names[name])
                    taggers.append(add_weight(fields(symbol, integer, "minor"), weight))
                    branches.append(currency(symbol, " " + name + " ", cents))
                    # NeMo's verbalizer reads "शून्य <major> X <minor>" as just "X <minor>".
                    minor_only.append(add_weight(currency(symbol, "", cents), weight - 0.1))
        self.tagger = self.add_tokens(pynini.union(*taggers))

        def integer(graph):
            return self.verbalize_field("integer_part", graph)

        verbalizer = pynini.union(
            integer(number) + pynini.union(*plural_branches),
            add_weight(integer(one), -0.001) + pynini.union(*singular_branches),
            integer(delete(ALL_ZERO)) + pynini.union(*minor_only),
        )
        self.verbalizer = self.delete_tokens(verbalizer)
