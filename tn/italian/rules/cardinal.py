# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.italian.rules.utils import preferred_map
from tn.processor import Processor


class Cardinal(Processor):
    """Cardinals, e.g. ``3544`` -> ``tremila cinquecento quarantaquattro``.

    The tagger keeps the written digits (including ``.``/space thousands
    separators such as ``1.000.000``) and an optional ``-`` sign; the
    verbalizer reads them. As in NeMo, ``milioni`` counts in groups of six
    digits (``3453243534`` -> ``tremila quattrocento cinquantatre milioni
    ...``), so ``miliardo`` is 10^12 and ``trilione`` is 10^18.
    """

    def __init__(self):
        super().__init__("cardinal", token_orders={})
        self.build_graphs()
        self.sign = pynini.cross("-", "meno")
        sign_tag = (self.tag_field("negative", self.sign) + insert(" ")).ques
        sign_verb = (self.verbalize_field("negative", self.sign) + delete(" ") + insert(" ")).ques
        self.tagger = self.add_tokens(sign_tag + self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(sign_verb + self.verbalize_field("integer", self.graph))

    def build_graphs(self):
        digit = self.DIGIT
        nonzero = digit - "0"
        space = insert(" ")
        zero = pynini.cross("0", "zero")
        digits = preferred_map("italian/data/numbers/digit.tsv", invert=True)
        teens = preferred_map("italian/data/numbers/teen.tsv", invert=True)
        tens = preferred_map("italian/data/numbers/tens.tsv", invert=True)
        tens_one = preferred_map("italian/data/numbers/tens_one.tsv", invert=True)
        tens_eight = preferred_map("italian/data/numbers/tens_eight.tsv", invert=True)
        hundreds = preferred_map("italian/data/numbers/hundreds.tsv", invert=True)

        # ``ventuno``/``ventotto`` elide the final vowel of the ten.
        plain_units = pynini.union(*"2345679") @ digits
        two_digits = teens | tens + (delete("0") | plain_units) | tens_one | tens_eight
        self.digits = digits.optimize()
        self.two_digits = two_digits.optimize()
        hundred = pynini.cross("1", "cento") | hundreds
        three_digits = hundred + (delete("00") | space + two_digits | delete("0") + space + digits)
        self.three_digits = three_digits.optimize()
        self.up_to_hundreds = (digits | two_digits | three_digits).optimize()

        # Exact three-digit groups with value 1..999, optionally excluding 1.
        group = three_digits | delete("0") + two_digits | delete("00") + digits
        group_no_one = (pynini.closure(digit, 3, 3) - "001") @ group
        rest3 = delete("000") | space + group
        # Exact six-digit groups with value 1..999999.
        six = (
            delete("000") + group
            | group_no_one + insert("mila") + rest3
            | pynini.cross("001", "mille") + rest3
        ).optimize()
        six_no_one = (pynini.closure(digit, 6, 6) - "000001") @ six

        def head(length):
            # Unpadded leading digits (1..length) padded to a six-digit group.
            padded = pynini.union(*[
                insert("0" * (length - size)) + nonzero + pynini.closure(digit, size - 1, size - 1)
                for size in range(1, length + 1)
            ])
            return padded.optimize()

        below_million = head(6) @ six
        rest = delete("000000") | space + six
        scales = [("un milione", " milioni"), ("un miliardo", " miliardi"), ("un trilione", " trilioni")]
        graph = below_million
        for one, many in scales:
            block = pynini.cross("000001", one) | six_no_one + insert(many)
            leading = head(6) @ block
            graph |= leading + rest
            rest = delete("000000") + rest | space + block + rest
            rest = rest.optimize()
        self.number = (zero | graph).optimize()

        # Thousands separators: ``.`` or a space between three-digit groups.
        separator = delete(pynini.union(".", " "))
        grouped = pynini.closure(digit, 1, 3) + (separator + pynini.closure(digit, 3, 3)).plus
        self.graph = ((digit.plus | grouped) @ self.number).optimize()
