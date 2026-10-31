# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib.pynutil import delete, insert

from tn.hungarian.rules.utils import DIGIT, NONZERO, data_labels, naive_inflector
from tn.processor import Processor

TOKEN_ORDER = {}

# Names of the fractional positions after "tized" and "század".
DECIMAL_NAMES = ["ezred", "milliomod", "milliárdod", "billiomod", "billiárdod", "trilliomod", "trilliárdod"]


def quantities():
    """Quantity words with their inflected and abbreviated forms, e.g. ``mrd``."""

    output = []
    for row in data_labels("number/quantities.tsv"):
        if len(row) == 2:
            output.append((row[0], row[1]))
            output += naive_inflector(row[0], row[1], True)
        else:
            output.append((row[0], row[0]))
            real = [form[1] for form in naive_inflector(".", row[0], True)]
            output += [(form, form) for form in real]
            if "lli" in row[0]:
                output.append((row[0].replace("lli", "li"), row[0]))
                output += [(form.replace("lli", "li"), form) for form in real]
    return pynini.string_map(output).optimize()


class Decimal(Processor):
    """Decimals with a comma, e.g. ``11,04006`` -> ``tizenegy egész négyezer-hat százezred``.

    Fields stay raw: ``integer_part``, ``fractional_part`` and an optional
    ``quantity`` (``1 mrd`` -> ``egy milliárd``).
    """

    def __init__(self, cardinal):
        super().__init__("decimal", token_orders=TOKEN_ORDER)
        names = ["tized", "század"] + [prefix + name for name in DECIMAL_NAMES for prefix in ["", "tíz", "száz"]]
        fractional = None
        leading = pynini.accep("")
        for name in names:
            # The last non-zero digit fixes the denominator; trailing zeros are dropped.
            number = (leading + NONZERO) @ (delete("0").star + cardinal.plain)
            leading += DIGIT
            branch = number + delete("0").star + insert(" " + name)
            fractional = branch if fractional is None else fractional | branch
        self.fractional = fractional.optimize()
        self.integer = cardinal.graph
        self.quantity = quantities()
        self.quantity_integer = cardinal.up_to_hundreds
        self.sign = cardinal.sign

        sign_t = cardinal.sign_tagger
        integer_t = self.tag_field("integer_part", self.integer)
        fractional_t = self.tag_field("fractional_part", self.fractional)
        quantity_t = pynini.accep(" ") + self.tag_field("quantity", self.quantity)
        decimal_t = integer_t + delete(",") + insert(" ") + fractional_t
        self.unsigned_tagger = (decimal_t + quantity_t.ques | self.tag_field("integer_part", self.quantity_integer)
                                + quantity_t).optimize()
        self.tagger = self.add_tokens(sign_t + self.unsigned_tagger)

        integer_v = self.verbalize_field("integer_part", self.integer)
        fractional_v = self.verbalize_field("fractional_part", self.fractional)
        quantity_v = pynini.accep(" ") + self.verbalize_field("quantity", self.quantity)
        self.unsigned_verbalizer = (integer_v + pynini.cross(" ", " egész ") + fractional_v + quantity_v.ques
                                    | integer_v + quantity_v).optimize()
        self.signed_verbalizer = cardinal.sign_verbalizer + self.unsigned_verbalizer
        self.verbalizer = self.delete_tokens(self.signed_verbalizer)
