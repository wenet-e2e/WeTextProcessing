# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib import pynutil

from tn.processor import Processor
from tn.swedish.rules.utils import alt, rewrite, squeeze_spaces, tsv

TOKEN_ORDER = {}

_SCALES = ("triljard", "triljon", "biljard", "biljon", "miljard", "miljon")


class Cardinal(Processor):
    """Raw Swedish cardinal digits; the verbalizer spells them out."""

    def __init__(self, deterministic=True):
        super().__init__("cardinal", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        self.build_graphs()
        sign = pynini.cross("-", "minus")
        self.sign = sign
        sign_t = (self.tag_field("negative", sign) + pynutil.insert(" ")).ques
        sign_v = (self.verbalize_field("negative", sign) + pynutil.delete(" ") + pynutil.insert(" ")).ques
        integer = self.graph
        if not deterministic:
            integer = integer | alt(self.graph_en) | alt(self.single_digits_graph)
        self.tagger = self.add_tokens(sign_t + self.tag_field("integer", integer))
        self.verbalizer = self.delete_tokens(sign_v + self.verbalize_field("integer", integer))

    def build_graphs(self):
        det = self.deterministic
        digit_any = self.DIGIT
        zero = tsv("numbers/zero").invert()
        digit = tsv("numbers/digit").invert()
        teen = tsv("numbers/teen").invert()
        ties = tsv("numbers/ties").invert()
        ett_to_en = pynini.cross("ett", "en")

        digits_no_one = (digit_any - "1") @ digit
        final_digit = digit if det else digit | alt(pynini.cross("1", "en"))
        graph_digit = final_digit
        self.digit = final_digit
        self.zero = zero
        single = graph_digit | zero
        self.single_digits_graph = (single + (pynutil.insert(" ") + single).star).optimize()

        if not det:
            alt_ties = rewrite(ties, pynini.string_map([("go", "gi"), ("tio", "ti")]), "", "[EOS]")
            ties = ties | alt(alt_ties) | alt(pynini.string_map([("4", "förtio"), ("4", "förti"), ("2", "tju")]))

        graph_tens = teen | ties + (pynutil.delete("0") | graph_digit)
        if not det:
            graph_tens |= alt(pynini.cross("18", "aderton"))
            graph_tens |= alt(ties + pynutil.insert(" ") + graph_digit)
        self.tens = graph_tens.optimize()

        hundreds = digits_no_one + pynutil.insert("hundra") | pynini.cross("1", "hundra")
        if not det:
            hundreds |= alt(pynini.cross("1", "etthundra"))
            hundreds |= alt(digit + pynutil.insert(" hundra"))

        graph_hundreds = hundreds + (pynutil.delete("00") | graph_tens | pynutil.delete("0") + final_digit)
        if not det:
            graph_hundreds |= alt(hundreds + (
                pynutil.insert(" ") + graph_tens | pynini.cross("0", " ") + final_digit))
        component = graph_hundreds | pynutil.delete("0") + graph_tens
        component_nz = component | pynutil.delete("00") + graph_digit
        component_nz_no_one = (component | pynutil.delete("00") + digits_no_one).optimize()
        self.component_nz_no_one = component_nz_no_one

        tusen = pynutil.insert("tusen")
        etttusen = pynini.cross("001", "tusen")
        following = pynutil.insert(" ") + component_nz
        if not det:
            tusen |= alt(pynutil.insert(" tusen"))
            for form in (" tusen", "etttusen", " etttusen", "ett tusen", " ett tusen"):
                etttusen |= alt(pynini.cross("001", form))
            following |= alt(component_nz)
        rest = following | pynutil.delete("000")
        thousands = (
            pynutil.delete("000") + component_nz
            | component_nz_no_one + tusen + rest
            | etttusen + rest
        )

        graph = pynini.accep("")
        for scale in _SCALES:
            graph += self.make_million(scale, component_nz_no_one)
        graph += thousands | pynutil.delete("000000")

        pad = pynini.cdrewrite(pynutil.insert("0").star, "[BOS]", "", digit_any.star)
        number = ((digit_any - "0") + digit_any.star) @ pad @ digit_any**24 @ graph @ squeeze_spaces()
        number = (number | zero).optimize()
        self.graph_unfiltered = number

        three = digit_any**3
        grouped = digit_any.plus | (
            pynini.closure(digit_any, 1, 3) + (pynutil.delete(" ") + three).star
            + pynutil.delete(" ") + three)
        self.graph = (grouped @ number).optimize()
        self.graph_en = rewrite(self.graph, ett_to_en, "", "[EOS]")
        numeral = (digit_any | " ").star
        self.not_one = (numeral - "1").optimize()
        self.not_zero = (numeral - "0").optimize()
        self.graph_no_one = (self.not_one @ self.graph).optimize()
        self.graph_no_one_en = (self.not_one @ self.graph_en).optimize()

        # 1 to 999 without leading zeros.
        upto = ((pynini.closure(digit_any, 2, 3) | (digit_any - "0")) @ self.graph).optimize()
        self.up_to_thousand = upto
        self.up_to_thousand_en = rewrite(upto, ett_to_en, "", "[EOS]")
        self.up_to_thousand_no_one = (self.not_one @ upto).optimize()
        self.up_to_thousand_no_one_en = (self.not_one @ self.up_to_thousand_en).optimize()

        nonzero = digit_any - "0"
        zero_space = zero + pynutil.insert(" ")
        self.zero_space = zero_space
        self.three_digits_read = (
            (nonzero + digit_any**2) @ self.up_to_thousand_no_one
            | zero_space + (digit_any**2 @ graph_tens)
            | zero_space + zero_space + digit
        )
        self.two_digits_read = (nonzero + digit_any) @ graph_tens | zero_space + digit
        self.any_read_digit = (nonzero @ digit) + (pynutil.insert(" ") + digit).star
        # Two or three fractional digits are read as a number unless they
        # start with zero; one digit or four and more are read digit by digit.
        digit_by_digit = (
            digit_any | "0" + digit_any.plus | pynini.closure(digit_any, 4)
        ) @ self.single_digits_graph
        self.fraction_digits_read = (
            (nonzero + digit_any**2) @ self.up_to_thousand_no_one
            | (nonzero + digit_any) @ graph_tens
            | digit_by_digit
        )
        if not det:
            self.fraction_digits_read |= alt(self.single_digits_graph) | alt(self.graph)
        self.fraction_digits_read = self.fraction_digits_read.optimize()
        if not det:
            self.three_digits_read |= alt(digit + pynutil.insert(" ") + digit + pynutil.insert(" ") + digit)
            self.three_digits_read |= alt((nonzero + digit_any) @ graph_tens + pynutil.insert(" ") + digit)
            self.three_digits_read |= alt(digit + pynutil.insert(" ") + (nonzero + digit_any) @ graph_tens)
            self.two_digits_read |= alt(digit + pynutil.insert(" ") + digit)
            self.any_read_digit |= self.two_digits_read | self.three_digits_read
        self.three_digits_read = self.three_digits_read.optimize()
        self.two_digits_read = self.two_digits_read.optimize()
        self.any_read_digit = self.any_read_digit.optimize()

    def make_million(self, number, non_zero_no_one):
        old = number.replace("lj", "lli")
        graph = pynini.cross("001", number) | non_zero_no_one + pynutil.insert(" " + number + "er")
        if not self.deterministic:
            forms = [old]
            for one in ("en", "ett"):
                forms += [one + " " + number, one + " " + old, one + number, one + old]
            graph |= alt(pynini.string_map([("001", form) for form in forms]))
            graph |= alt(non_zero_no_one + pynutil.insert(" " + old + "er"))
            graph |= alt(non_zero_no_one + pynutil.insert(old + "er"))
        return (graph | pynutil.delete("000")) + pynutil.insert(" ")
