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

from tn.hindi.rules.utils import ALL_DIGIT, hindi_path, squeeze_spaces
from tn.processor import Processor

TOKEN_ORDER = {}


class Address(Processor):
    """Address phrases around a keyword stay one raw ``value`` field.

    ``७०० ओक स्ट्रीट`` -> ``address { value: "७०० ओक स्ट्रीट" }`` -> ``सात सौ ओक स्ट्रीट``.
    NeMo models this as a measure with the pseudo unit "address".
    """

    def __init__(self, cardinal, ordinal, serial):
        super().__init__("address", token_orders=TOKEN_ORDER)
        digit = pynini.string_file(hindi_path("numbers/digit.tsv"))
        zero = pynini.string_file(hindi_path("numbers/zero.tsv"))
        phone_digit = pynini.string_file(hindi_path("telephone/number.tsv"))
        letters = pynini.string_file(hindi_path("address/letters.tsv"))
        letters = (letters | (self.TO_LOWER @ letters)).optimize()
        special = pynini.string_file(hindi_path("address/special_characters.tsv"))
        context = pynini.string_file(hindi_path("address/context.tsv"))
        keywords = (context | ((self.TO_LOWER + self.VSIGMA) @ context)).project("input").optimize()
        places = pynini.union(
            pynini.string_file(hindi_path("address/states.tsv")),
            pynini.string_file(hindi_path("address/cities.tsv"))).optimize()
        whitespace = pynini.union(" ", "\t", "\n", "\r", "\u00a0")
        punctuation = pynini.union(",", ".", "।")

        # State/city followed by a six digit pincode.
        number_word = (digit | zero | phone_digit).optimize()
        pincode = number_word + (insert(" ") + number_word)**5
        word = (self.NOT_SPACE - (ALL_DIGIT | punctuation)).plus
        separator = pynini.accep(",").ques + " "
        text = pynini.closure(word + separator | cardinal.code_num_graph + separator, 0, 5)
        structured = ((cardinal.code_num_graph + separator).ques + text + places
                      + (pynini.accep(" ") + pincode).ques)
        structured = add_weight(structured, 1.0)

        # Free address text: convert digits, letters, "-" and "/" near an address keyword.
        single_letter = letters.copy().project("input")
        special_chars = pynini.union("-", "/")
        convertible = pynini.union(ALL_DIGIT, special_chars, single_letter)
        other_char = pynini.difference(self.VCHAR, pynini.union(whitespace, convertible, ",")).optimize()
        token = pynini.union(
            pynini.accep(" "),
            delete(whitespace.star) + ",",
            insert(" ") + (special_chars @ (digit | zero | special | phone_digit)),
            insert(" ") + serial.mixed_alphanum_graph,
            add_weight(insert(" ") + ordinal.graph, self.MIN_NEG_WEIGHT),
            add_weight(insert(" ") + (ALL_DIGIT.plus @ cardinal.code_num_graph), 0.5),
            insert(" ") + (single_letter @ letters),
            insert(" ") + single_letter + single_letter.plus,
            add_weight(insert(" ") + other_char.plus, 0.1),
        ).optimize()
        processor = token.plus.optimize()

        # Up to five words before and/or after an address keyword.
        boundary_char = pynini.union(whitespace, ",", "।", ".")
        inner_word = pynini.difference(self.VCHAR, boundary_char).plus
        boundary = boundary_char.plus
        window = pynini.accep("") | (pynini.closure(inner_word + boundary, 0, 4) + inner_word + boundary_char.star)
        pattern = pynini.union(
            (keywords + boundary + window).optimize(),
            (window + boundary + keywords + (boundary + window).ques).optimize())

        # The tagger only needs the weighted domain of ``pattern @ processor``; the
        # verbalizer reads any raw value with the processor, whose best path is the
        # same one NeMo picks inside the keyword window. ``tag_field`` optimizes the
        # projected domain once.
        free_domain = pattern @ self.input_projection(processor)
        structured_domain = structured.copy().project("input")
        # Leading/trailing whitespace belongs to the token separators, not to the address.
        trimmed = self.NOT_SPACE | self.NOT_SPACE + self.VSIGMA + self.NOT_SPACE
        domain = trimmed @ (free_domain | structured_domain)
        reading = ((processor | structured) @ squeeze_spaces()).optimize()

        self.tagger = self.add_tokens(self.tag_field("value", domain))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", reading))
