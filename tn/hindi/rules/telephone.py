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

from tn.hindi.rules.utils import ALL_DIGIT, ALL_ZERO, hindi_path, squeeze_spaces
from tn.processor import Processor

TOKEN_ORDER = {}

MOBILE_START = pynini.union(*"६७८९6789")
LANDLINE_START = pynini.union(*"२३४६2346")


class Telephone(Processor):
    """Phone, card and pin numbers keep their written digits and context words raw.

    ``+९१ ९२१०५१५६०६`` -> ``telephone { country_code: "+९१" number_part: "९२१०५१५६०६" }``;
    the verbalizer reads every digit (``प्लस नौ एक नौ दो ...``).
    """

    def __init__(self):
        super().__init__("telephone", token_orders=TOKEN_ORDER)
        digit_word = pynini.string_file(hindi_path("telephone/number.tsv"))
        digits = pynini.string_file(hindi_path("numbers/digit.tsv"))
        zero = pynini.string_file(hindi_path("numbers/zero.tsv"))
        self.num = pynini.union(digit_word, digits, zero).optimize()
        self.squeeze = squeeze_spaces()
        whitespace = pynini.union(" ", "\t", "\n", "\r", "\u00a0")
        non_digit = pynini.difference(self.VCHAR, pynini.union(ALL_DIGIT, whitespace))
        window = pynini.closure(non_digit.plus + " ", 0, 5)

        def context(name):
            keywords = pynini.string_file(hindi_path("telephone/{}_context.tsv".format(name)))
            return (keywords + " " + window).ques, (" " + window + keywords).ques

        patterns = [
            (self.mobile(*context("mobile")), 0.7),
            (self.landline(*context("landline")), 0.8),
            (self.digits_only(4, *context("credit")), 0.9),
            (self.digits_only(6, *context("pincode")), 1.0),
        ]
        self.tagger = self.add_tokens(pynini.union(*[add_weight(t, w) for (t, _), w in patterns]))
        self.verbalizer = self.delete_tokens(pynini.union(*[add_weight(v, w) for (_, v), w in patterns]))

    def field(self, name, graph, first=False):
        """Returns the raw tagger and the spoken verbalizer of one field."""

        graph = (graph @ self.squeeze).optimize()
        tagger = self.tag_field(name, graph)
        verbalizer = self.verbalize_field(name, graph, leading_space=not first)
        if not first:
            tagger = insert(" ") + tagger
            verbalizer = insert(" ") + verbalizer
        return tagger, verbalizer

    def run(self, low, high=None):
        token = self.num + insert(" ")
        return pynini.closure(token, low, high) if high is not None else pynini.closure(token, low)

    def mobile(self, before, after):
        number = MOBILE_START @ self.num + insert(" ") + self.run(9)
        local_t, local_v = self.field("number_part", before + delete(ALL_ZERO).ques + insert("शून्य ") + number + after,
                                      first=True)
        country_t, country_v = self.field("country_code", before + pynini.cross("+", "प्लस ") + self.run(1, 3),
                                          first=True)
        number_t, number_v = self.field("number_part", number + after)
        extension_t, extension_v = self.field("extension", self.run(1, 3) + after)
        tagger = (local_t | country_t + self.DELETE_SPACE + number_t) + (self.DELETE_SPACE + extension_t).ques
        verbalizer = (local_v | country_v + number_v) + extension_v.ques
        return tagger, verbalizer

    def landline(self, before, after):
        graphs = []
        space = delete(" ").star
        for std_length in range(2, 8):
            std = delete(ALL_ZERO).ques + insert("शून्य ") + self.run(std_length, std_length)
            bracketed = delete(ALL_ZERO).ques + space + delete("(") + space + std + space + delete(")")
            number = LANDLINE_START @ self.num + insert(" ") + self.run(9 - std_length, 9 - std_length)
            separator = delete(pynini.union("-", ".")).ques
            graphs.append(before + (std | bracketed) + separator + space + number + after)
        return self.field("number_part", pynini.union(*graphs), first=True)

    def digits_only(self, low, before, after):
        return self.field("number_part", before + self.run(low) + after, first=True)
