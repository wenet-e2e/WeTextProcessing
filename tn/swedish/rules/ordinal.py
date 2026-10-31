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
from tn.swedish.rules.cardinal import _SCALES
from tn.swedish.rules.utils import alt, rewrite, squeeze_spaces, tsv

TOKEN_ORDER = {}


class Ordinal(Processor):
    """Raw ordinal digits plus their written suffix (``.``, ``:a``, ``:e``)."""

    def __init__(self, cardinal, deterministic=True):
        super().__init__("ordinal", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        self.build_graphs(cardinal)

        digit = self.DIGIT
        nonzero = digit - "0"
        cardinal_format = nonzero + pynini.closure(pynini.closure(digit | " ") + digit, 0, 1)
        one_two = pynini.union("1", "2")
        a_format = (
            pynini.closure(cardinal_format + (digit - "1"), 0, 1) + one_two
            | (digit - "1") + one_two
        )
        e_format = pynini.closure(
            (digit - one_two)
            | cardinal_format + "1" + digit
            | cardinal_format + (digit - "1") + (digit - one_two),
            1,
        )
        self.a_words = (a_format @ self.graph).optimize()
        self.e_words = (e_format @ self.graph).optimize()
        self.suffixed_to_words = (
            self.a_words + pynutil.delete(pynini.union(":a", ":A"))
            | self.e_words + pynutil.delete(pynini.union(":e", ":E"))
        ).optimize()

        dot = pynini.accep(".")
        suffix_a = pynini.union(":a", ":A")
        suffix_e = pynini.union(":e", ":E")
        noun = pynini.union("kapitlet", pynini.cross("kap", "kapitlet"))
        tagger = (
            self.tag_field("integer", self.bare_ordinals) + pynutil.insert(" ") + self.tag_field("suffix", dot)
            | self.tag_field("integer", self.a_words) + pynutil.insert(" ") + self.tag_field("suffix", suffix_a)
            | self.tag_field("integer", self.e_words) + pynutil.insert(" ") + self.tag_field("suffix", suffix_e)
            | self.tag_field("integer", self.bare_ordinals) + pynini.accep(" ") + self.tag_field("noun", noun)
        )
        drop = pynutil.delete(dot | suffix_a | suffix_e)
        verbalizer = (
            self.verbalize_field("integer", self.graph) + self.verbalize_field("suffix", drop, leading_space=True)
            | self.verbalize_field("integer", self.bare_ordinals) + pynutil.insert(" ")
            + self.verbalize_field("noun", noun, leading_space=True)
        )
        self.tagger = self.add_tokens(tagger)
        self.verbalizer = self.delete_tokens(verbalizer)

    def build_graphs(self, cardinal):
        det = self.deterministic
        digit_any = self.DIGIT
        digit = tsv("ordinals/digit").invert()
        teens = tsv("ordinals/teen").invert()
        ties = tsv("ordinals/ties").invert()
        zero = tsv("ordinals/zero").invert()
        card_ties = tsv("numbers/ties").invert()
        card_digit = tsv("numbers/digit").invert()
        digits_no_one = (digit_any - "1") @ card_digit
        if not det:
            ties |= alt(pynini.cross("4", "förtionde"))
            teens |= alt(pynini.cross("18", "adertonde"))

        tens = teens | card_ties + digit | ties + pynutil.delete("0")
        self.tens = tens.optimize()
        bare_hundreds = digits_no_one + pynini.cross("00", "hundrade") | pynini.cross("100", "hundrade")
        hundreds = digits_no_one + pynutil.insert("hundra") | pynini.cross("1", "hundra")
        if not det:
            bare_hundreds |= alt(pynini.string_map([("100", "etthundrade"), ("100", "ett hundrade")]))
            bare_hundreds |= alt(card_digit + pynini.cross("00", " hundrade"))
            hundreds |= alt(pynini.string_map([("1", "etthundra"), ("1", "ett hundra")]))
            hundreds |= alt(card_digit + pynutil.insert(" hundra"))
        graph_hundreds = hundreds + (tens | pynutil.delete("0") + digit)
        if not det:
            graph_hundreds |= alt(hundreds + (
                teens | pynutil.insert(" ") + teens | pynini.cross("0", " ") + digit))
        graph_hundreds |= bare_hundreds
        component = graph_hundreds | pynutil.delete("0") + tens
        component_nz = component | pynutil.delete("00") + digit

        tusen = pynini.accep("tusen")
        if not det:
            for form in (" tusen", "ettusen", " ettusen", "ett tusen", " ett tusen"):
                tusen |= alt(pynini.accep(form))
        rest = pynutil.insert(" ") + component_nz | pynutil.delete("000")
        thousands = (
            pynutil.delete("000") + component_nz
            | cardinal.component_nz_no_one + pynutil.insert(tusen) + rest
            | pynini.cross("001", tusen) + rest
        )
        graph = pynini.accep("")
        for scale in _SCALES:
            graph += cardinal.make_million(scale, cardinal.component_nz_no_one)
        graph += thousands | pynutil.delete("000000")

        endings = pynini.string_map([
            ("ljon", "ljonte"), ("ljoner", "ljonte"), ("llion", "llionte"), ("llioner", "llionte"),
            ("ljard", "ljarte"), ("ljarder", "ljarte"), ("lliard", "lliarte"), ("lliarder", "lliarte"),
            ("tusen", "tusende"),
        ])
        pad = pynini.cdrewrite(pynutil.insert("0").star, "[BOS]", "", digit_any.star)
        number = ((digit_any - "0") + digit_any.star) @ pad @ digit_any**24 @ graph @ squeeze_spaces()
        self.bare_ordinals = rewrite(number, endings, "", "[EOS]")
        three = digit_any**3
        grouped = digit_any.plus | (
            pynini.closure(digit_any, 1, 3) + (pynutil.delete(" ") + three).star
            + pynutil.delete(" ") + three)
        self.graph = (grouped @ (self.bare_ordinals | zero)).optimize()
