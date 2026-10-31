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

from itn.hungarian.rules.utils import DIGIT, NONZERO, data_labels, naive_inflector
from tn.processor import Processor

TOKEN_ORDER = {}

# Names of the fractional positions after "tized" and "század".
DECIMAL_NAMES = ["ezred", "milliomod", "milliárdod", "billiomod", "billiárdod", "trilliomod", "trilliárdod"]


def quantities():
    """Spoken quantity words (with case endings) -> written, e.g. ``milliárd`` -> ``mrd``."""

    written = {}
    for row in data_labels("number/quantities.tsv"):
        if len(row) == 2:
            pairs = [(row[0], row[1])] + naive_inflector(row[0], row[1], True)
        else:
            pairs = [(row[0], row[0])] + [(form, form) for _, form in naive_inflector(".", row[0], True)]
        for key, spoken in pairs:
            written.setdefault(spoken, key)
    return pynini.string_map([(spoken, key) for spoken, key in written.items()]).optimize()


class Decimal(Processor):
    """Decimals, e.g. ``tizenegy egész négyezer-hat százezred`` -> ``11,04006``.

    Raw spoken ``integer_part``/``fractional_part`` and an optional
    ``quantity`` (``egy milliárd`` -> ``1 mrd``).
    """

    def __init__(self, cardinal):
        super().__init__("decimal", ordertype="itn", token_orders=TOKEN_ORDER)
        names = ["tized", "század"] + [prefix + name for name in DECIMAL_NAMES for prefix in ["", "tíz", "száz"]]
        spoken = None
        leading = pynini.accep("")
        for name in names:
            # The position of the last digit fixes the denominator.
            branch = (leading + NONZERO) @ (delete("0").star + cardinal.spoken) + insert(" " + name)
            spoken = branch if spoken is None else spoken | branch
            leading += DIGIT
        self.fractional = pynini.invert(spoken).optimize()
        self.integer = cardinal.graph
        self.quantity = quantities()
        self.quantity_integer = pynini.invert(cardinal.up_to_hundreds).optimize()

        integer_t = self.tag_field("integer_part", self.integer)
        fractional_t = self.tag_field("fractional_part", self.fractional)
        quantity_t = pynini.accep(" ") + self.tag_field("quantity", self.quantity)
        decimal_t = integer_t + delete(" egész") + pynini.accep(" ") + fractional_t
        self.unsigned_tagger = (
            decimal_t + quantity_t.ques
            | self.tag_field("integer_part", self.quantity_integer) + quantity_t
        ).optimize()
        self.tagger = self.add_tokens(cardinal.sign_tagger + self.unsigned_tagger)

        integer_v = self.verbalize_field("integer_part", self.integer)
        fractional_v = self.verbalize_field("fractional_part", self.fractional)
        quantity_v = pynini.accep(" ") + self.verbalize_field("quantity", self.quantity)
        self.unsigned_verbalizer = (
            integer_v + pynini.cross(" ", ",") + fractional_v + quantity_v.ques
            | integer_v + quantity_v
        ).optimize()
        self.verbalizer = self.delete_tokens(cardinal.sign_verbalizer + self.unsigned_verbalizer)
