# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from tn.hindi.rules.utils import ALL_DIGIT, hindi_path
from tn.processor import Processor

TOKEN_ORDER = {}


class Serial(Processor):
    """Alphanumeric codes keep their written form in one raw ``value`` field.

    ``कोविड-19`` -> ``serial { value: "कोविड-19" }`` -> ``कोविड-उन्नीस``;
    ``A12`` -> ``ए बारह``; ``2^2`` -> ``दो स्क्वेर्ड``.
    """

    def __init__(self, cardinal):
        super().__init__("serial", token_orders=TOKEN_ORDER)
        number = cardinal.code_num_graph
        symbols = pynini.string_file(hindi_path("serial/special_symbols.tsv"))
        chars = pynini.string_file(hindi_path("serial/chars.tsv")).optimize()
        letters = pynini.string_file(hindi_path("address/letters.tsv"))
        letters = (letters | (self.TO_LOWER @ letters)).optimize()
        latin = letters + (insert(" ") + letters).star
        devanagari = pynini.closure(chars, 2)
        alphas = (latin | devanagari).optimize()
        delimiter = pynini.union("-", "/", " ")
        segment = (alphas | number | symbols).optimize()
        core = segment + (delimiter + segment).plus

        # Letters and digits written together ("5जी", "FY2024") are read as separate
        # segments. NeMo inserts these spaces with two whole-string rewrites; the same
        # segmentation is spelled out here as an alternation of letter and digit runs.
        gap = insert(" ")
        from_alpha = alphas + (gap + number + gap + alphas).star + (gap + number).ques
        from_number = number + (gap + alphas + gap + number).star + (gap + alphas).ques
        chunk = (from_alpha | from_number | symbols).optimize()
        multi = (alphas + gap + number + (gap + alphas + gap + number).star + (gap + alphas).ques
                 | number + gap + alphas + (gap + number + gap + alphas).star + (gap + number).ques)
        glued = chunk + (delimiter + chunk).plus | multi
        serial = (core | glued).optimize()

        all_alphas = (self.ALPHA | chars.copy().project("input")).optimize()
        join = pynini.union(all_alphas, ALL_DIGIT, "-", "/")
        has_letter = join.star + all_alphas + join.star
        has_digit = join.star + ALL_DIGIT + join.star
        mixed = pynini.intersect(pynini.intersect(join.plus, has_letter), has_digit).optimize()
        self.mixed_alphanum_graph = (mixed @ serial).optimize()

        special_power = add_weight(pynini.string_file(hindi_path("serial/power_special.tsv")), -1.0)
        generic_power = add_weight(delete("^") + insert(" टु द पावर ") + number, 1.0)
        serial |= number + (special_power | generic_power)
        serial = pynini.closure(self.NOT_SPACE, 2) @ serial

        word_slash = self.ALPHA.plus + "/" + self.ALPHA.plus
        latin_join = self.ALPHA | "-" | "/"
        latin_word = pynini.intersect(latin_join.plus, latin_join.star + self.ALPHA + latin_join.star)
        dimension = ALL_DIGIT.plus + pynini.union("x", "X") + ALL_DIGIT.plus
        ordinal_suffix = pynini.union(
            pynini.string_file(hindi_path("ordinal/suffixes.tsv")),
            pynini.string_file(hindi_path("ordinal/suffixes_map.tsv"))).project("input")
        ordinal = ALL_DIGIT.plus + ordinal_suffix
        exclusions = (word_slash | latin_word | dimension | ordinal).optimize()
        self.graph = (pynini.difference(self.VSIGMA, exclusions) @ serial).optimize()

        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
