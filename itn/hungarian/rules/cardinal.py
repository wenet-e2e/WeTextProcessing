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

from itn.hungarian.rules.utils import DIGIT, NONZERO, data_map
from tn.processor import Processor

TOKEN_ORDER = {}

# Long-scale powers above a thousand, lowest first.
POWERS = ["millió", "milliárd", "billió", "billiárd", "trillió", "trilliárd"]


def group_digits(min_grouped=5):
    """Writes numbers of at least ``min_grouped`` digits in space-separated triples."""

    triple = insert(" ") + DIGIT ** 3
    grouped = pynini.closure(DIGIT, 1, 3) + triple.plus
    short = pynini.closure(DIGIT, 1, min_grouped - 1)
    long = pynini.closure(DIGIT, min_grouped)
    return (short | pynini.project(long, "input") @ grouped).optimize()


class Cardinal(Processor):
    """Cardinals, e.g. ``kettőezer-tíz`` / ``kétezer tíz`` -> ``2010``.

    The tagger keeps the spoken words; the verbalizer writes digits, grouping
    numbers of five or more digits with spaces (``tízezer`` -> ``10 000``).
    A bare ``egy`` (also the indefinite article) is not tagged on its own.
    """

    def __init__(self):
        super().__init__("cardinal", ordertype="itn", token_orders=TOKEN_ORDER)
        self.build_numbers()
        self.sign = pynini.cross("mínusz", "-")
        self.sign_tagger = (self.tag_field("negative", self.sign) + pynini.accep(" ")).ques
        self.sign_verbalizer = (self.verbalize_field("negative", self.sign) + delete(" ")).ques
        standalone = pynini.difference(pynini.project(self.graph, "input"), "egy") @ self.graph
        self.tagger = self.add_tokens(self.sign_tagger + self.tag_field("integer", standalone))
        self.verbalizer = self.delete_tokens(self.sign_verbalizer + self.verbalize_field("integer", self.graph))

    def build_numbers(self):
        """Builds ``self.spoken``: digits -> every accepted spoken form."""

        two = pynini.cross("2", "két")
        digit = data_map("number/digit.tsv", invert=True)
        digit_inline = data_map("number/digit_inline.tsv", invert=True) | pynini.cross("2", "kettő")
        tens = data_map("number/tens.tsv", invert=True)
        tens_inline = data_map("number/tens_inline.tsv", invert=True)
        zero = data_map("number/zero.tsv", invert=True)
        digit_word = (digit | two).optimize()
        self.digit = digit_word
        digit_no_one = (NONZERO - "1") @ digit_word
        digit_inline_no_one = (NONZERO - "1") @ digit_inline

        graph_tens = (tens_inline + digit_word | tens).optimize()
        self.tens = graph_tens
        one = pynini.union("", "egy")
        base_hundreds = pynini.cross("1", one + "száz") | digit_inline_no_one + insert("száz")
        hundreds = (base_hundreds + (delete("00") | graph_tens | delete("0") + digit_word)).optimize()
        self.hundreds = hundreds
        comp = (hundreds | delete("0") + graph_tens | delete("00") + digit_word).optimize()
        comp_no_one = (hundreds | delete("0") + graph_tens | delete("00") + digit_no_one).optimize()
        head = (digit_word | graph_tens | hundreds).optimize()
        head_no_one = (digit_no_one | graph_tens | hundreds).optimize()
        self.up_to_hundreds = head

        joint = insert(pynini.union("-", " ", ""))
        level = (
            delete("000") + comp
            | comp_no_one + insert("ezer") + (joint + comp | delete("000"))
            | pynini.cross("001", one + "ezer") + (joint + comp | delete("000"))
        ).optimize()
        top = head | (head_no_one + insert("ezer") | pynini.cross("1", one + "ezer")) + (joint + comp | delete("000"))
        width = 6
        for word in POWERS:
            rest = joint + level | delete("0" * width)
            top |= (pynini.cross("1", one + word) | head_no_one + insert(word)) + rest
            level = (
                delete("000") + level | (pynini.cross("001", one + word) | comp_no_one + insert(word)) + rest
            ).optimize()
            width += 3
        self.zero = zero
        self.spoken = (zero | top).optimize()
        self.digits = pynini.invert(self.spoken).optimize()
        self.graph = (self.digits @ group_digits()).optimize()
