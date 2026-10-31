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

from itn.italian.rules.utils import load_rows, string_map, table_map
from tn.processor import Processor


def two_digit_pairs():
    """Returns ``(written, spoken)`` pairs for 1..99 with two-digit written forms.

    Tens elide their final vowel before ``uno``/``otto`` (``ventuno``,
    ``ventotto``); compounds of ``tre`` accept both ``ventitre`` and
    ``ventitré``.
    """

    units = {int(written): spoken for written, spoken in load_rows("italian/data/numbers/digit.tsv")}
    pairs = [("0{}".format(value), spoken) for value, spoken in units.items()]
    pairs += [(written, spoken) for written, spoken in load_rows("italian/data/numbers/teen.tsv")]
    for written, ten in load_rows("italian/data/numbers/tens.tsv"):
        pairs.append((written + "0", ten))
        for value, unit in units.items():
            stem = ten[:-1] if unit[0] in "uo" else ten
            pairs.append((written + str(value), stem + unit))
            if unit == "tre":
                pairs.append((written + str(value), stem + "tré"))
    return pairs


class Cardinal(Processor):
    """Cardinals, e.g. ``tremila cinquecento quarantaquattro`` -> ``3544``.

    The tagger keeps the spoken words; the verbalizer writes the digits.
    Components may be joined (``centocinquantacinque``) or spaced (``cento
    cinquantacinque``). ``miliardo`` is 10^9; NeMo's TN style of counting
    thousands of millions (``tremila ... milioni``) is accepted too.
    """

    def __init__(self):
        super().__init__("cardinal", token_orders={})
        self.build_graphs()
        self.sign = pynini.cross("meno", "-")
        sign_tag = (self.tag_field("negative", self.sign) + pynini.accep(" ")).ques
        sign_verb = (self.verbalize_field("negative", self.sign) + delete(" ")).ques
        self.tagger = self.add_tokens(sign_tag + self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(sign_verb + self.verbalize_field("integer", self.graph))

    def build_graphs(self):
        digit = self.DIGIT
        join = delete(" ").ques
        space = delete(" ")
        two = string_map(two_digit_pairs())
        hundreds = table_map("italian/data/numbers/hundreds.tsv")
        # ``centotto``/``centottanta`` drop the final ``o`` of the hundred.
        elided = string_map([(written, spoken[:-1]) for written, spoken in load_rows("italian/data/numbers/hundreds.tsv")])
        two_ott = ("ott" + self.VSIGMA) @ two
        three = (
            insert("0") + two
            | hundreds + insert("00")
            | hundreds + join + two
            | elided + two_ott
        ).optimize()
        three_no_one = (three @ (pynini.closure(digit, 3, 3) - "001")).optimize()
        rest3 = insert("000") | join + three
        six = (
            insert("000") + three
            | pynini.cross("mille", "001") + rest3
            | three_no_one + join + delete("mila") + rest3
        ).optimize()
        rest6 = insert("000000") | space + six
        millions = pynini.cross("un milione", "001") | three_no_one + delete(" milioni")
        billions = pynini.cross("un miliardo", "001") | three_no_one + delete(" miliardi")
        thousand_millions = six @ ((pynini.closure(digit, 3, 3) - "000") + pynini.closure(digit, 3, 3))
        padded = (
            billions + (space + millions | insert("000")) + rest6
            | insert("000") + millions + rest6
            | insert("000000") + six
            | thousand_millions + delete(" milioni") + rest6
        )
        strip = delete("0").star + (digit - "0") + digit.star
        self.number = (pynini.cross("zero", "0") | padded @ strip).optimize()
        self.graph = self.number
        self.two = two
        # Unpadded readings below one thousand, e.g. for decimal digit groups.
        self.below_thousand = (pynini.cross("zero", "0") | three @ strip).optimize()
        # Counting contexts (measures, money) also say ``un``/``una`` for 1.
        self.counted = (self.number | pynini.cross(pynini.union("un", "una"), "1")).optimize()
