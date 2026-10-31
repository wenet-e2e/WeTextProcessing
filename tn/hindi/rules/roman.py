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

from tn.hindi.rules.utils import hindi_path
from tn.processor import Processor
from tn.utils import load_labels

TOKEN_ORDER = {}


class Roman(Processor):
    """Roman numerals next to a Hindi keyword keep both parts raw.

    ``भास्कर-II`` -> ``roman { key_cardinal: "भास्कर" integer: "II" }`` -> ``भास्कर दो``;
    ``XIIवीं कक्षा`` -> ``roman { ordinal: "XIIवीं" key_cardinal: "कक्षा" }`` -> ``बारहवीं कक्षा``.
    """

    def __init__(self):
        super().__init__("roman", token_orders=TOKEN_ORDER)
        roman = pynini.string_file(hindi_path("roman/roman_to_spoken.tsv")).optimize()
        chars = pynini.string_file(hindi_path("serial/chars.tsv")).project("input")
        word = chars.plus
        phrase = (word + (pynini.union(" ", "-") + word).star).optimize()
        separator = delete(pynini.union("-", " ")) + insert(" ")

        rows = load_labels(hindi_path("roman/roman_to_spoken.tsv"))
        exceptions = load_labels(hindi_path("roman/roman_ordinal_exceptions.tsv"))
        fused_exceptions = {fused for fused, _ in exceptions}
        suffixes = load_labels(hindi_path("ordinal/suffixes.tsv")) + load_labels(
            hindi_path("ordinal/suffixes_map.tsv"))
        exception_graph = pynini.string_map([(fused, spoken) for fused, spoken in exceptions])
        regular = []
        for numeral, spoken in rows:
            for row in suffixes:
                fused = numeral + row[0]
                if fused not in fused_exceptions:
                    regular.append((fused, spoken + (row[1] if len(row) > 1 else row[0])))
        ordinal = pynini.union(add_weight(exception_graph, -0.1), pynini.string_map(regular)).optimize()

        key_t = self.tag_field("key_cardinal", phrase)
        integer_t = self.tag_field("integer", roman)
        ordinal_t = self.tag_field("ordinal", ordinal)
        space_t = delete(" ") + insert(" ")
        self.tagger = self.add_tokens(
            key_t + separator + integer_t
            | integer_t + separator + key_t
            | ordinal_t + (space_t + key_t).ques)

        key_v = self.verbalize_field("key_cardinal", phrase)
        integer_v = self.verbalize_field("integer", roman)
        space = delete(" ") + insert(" ")
        self.verbalizer = self.delete_tokens(
            key_v + space + integer_v
            | integer_v + space + key_v
            | self.verbalize_field("ordinal", ordinal) + (space + key_v).ques)
