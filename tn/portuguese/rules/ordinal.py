# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {}


def ordinal_table(name):
    """Maps cardinal words to ordinal words (the NeMo tables are ordinal -> cardinal)."""
    return {row[1]: row[0] for row in load_labels(get_abs_path("portuguese/data/ordinals/" + name))}


class Ordinal(Processor):
    """Raw ordinal digits and gender marks, e.g. 11ª -> integer "11", suffix "ª"."""

    def __init__(self, cardinal):
        super().__init__("ordinal", token_orders=TOKEN_ORDER)
        units = ordinal_table("digit.tsv")
        ties = ordinal_table("ties.tsv")
        hundreds = ordinal_table("hundreds.tsv")
        fem_rows = load_labels(get_abs_path("portuguese/data/ordinals/feminine.tsv"))
        fem_words = {row[0]: row[1] for row in fem_rows}
        fem_suffix = [(row[0], row[1]) for row in fem_rows if row[0].startswith("é")]

        def feminine(word):
            if word in fem_words:
                return fem_words[word]
            for masc, fem in fem_suffix:
                if word.endswith(masc):
                    return word[:-len(masc)] + fem
            return word

        def words(value):
            hundred, rest = divmod(value, 100)
            ten, unit = divmod(rest, 10)
            parts = []
            if hundred:
                parts.append(hundreds[cardinal.hundred_1 if hundred == 1 else cardinal.hundreds[str(hundred)]])
            if ten == 1:
                parts.append(units[cardinal.teens["10"]])
            elif ten:
                parts.append(ties[cardinal.tens[str(ten)]])
            if unit:
                parts.append(units[cardinal.digits[str(unit)]])
            return parts

        self.masculine = pynini.string_map([(str(n), " ".join(words(n))) for n in range(1, 1000)]).optimize()
        self.feminine = pynini.string_map([
            (str(n), " ".join(feminine(word) for word in words(n))) for n in range(1, 1000)
        ]).optimize()
        dot = pynini.accep(".").ques
        masc_suffix = dot + pynini.union("º", "°")
        fem_suffix = dot + pynini.accep("ª")

        number = self.input_projection(self.masculine)
        self.tagger = self.add_tokens(
            self.tag_field("integer", number) + insert(" ")
            + self.tag_field("suffix", masc_suffix | fem_suffix)
        )
        masc = self.verbalize_field("integer", self.masculine) + self.verbalize_field(
            "suffix", delete(masc_suffix), leading_space=True)
        fem = self.verbalize_field("integer", self.feminine) + self.verbalize_field(
            "suffix", delete(fem_suffix), leading_space=True)
        self.verbalizer = self.delete_tokens(masc | fem)
