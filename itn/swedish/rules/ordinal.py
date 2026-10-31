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

from itn.swedish.rules.utils import DIGIT, NONZERO, SCALES, SPACE, old_spelling, rows, strip_zeros
from tn.processor import Processor

TOKEN_ORDER = {}


def ordinal_words():
    """Final ordinal morphemes: (digits, teens, ties) as (word, value) rows."""

    digits = [tuple(row) for row in rows("ordinals/digit")]
    teens = [tuple(row) for row in rows("ordinals/teen")] + [("adertonde", "18")]
    ties = [tuple(row) for row in rows("ordinals/ties")] + [("förtionde", "4")]
    return digits, teens, ties


def scale_ordinal(scale):
    """miljon -> miljonte, miljard -> miljarte."""

    return scale + "te" if scale.endswith("n") else scale[:-1] + "te"


def number_graph(cardinal, digits, teens, ties, hundrade, tusende, scale_word):
    """Spoken numbers whose last morpheme comes from the given tables, to digits."""

    def mapping(pairs):
        return pynini.string_map(pairs).optimize()

    digit = mapping(digits)
    teen = mapping(teens)
    tie = mapping(ties)
    two = cardinal.ties + SPACE + digit | tie + pynutil.insert("0") | teen
    below_hundred = two | pynutil.insert("0") + digit
    three = (
        cardinal.hundred_prefix + SPACE + below_hundred
        | (cardinal.digit + SPACE | pynutil.insert("1")) + pynini.cross(hundrade, "00")
        | pynutil.insert("0") + below_hundred
    )
    six = (
        cardinal.thousand_prefix + SPACE + three
        | (cardinal.three + SPACE | pynutil.insert("001")) + pynini.cross(tusende, "000")
        | pynutil.insert("000") + three
    )
    graphs = [cardinal.groups + six]
    for index, scale in enumerate(SCALES):
        prefix = pynini.accep("")
        for group in cardinal.scale_groups[:index]:
            prefix += group
        words = pynini.union(scale_word(scale), scale_word(old_spelling(scale)))
        tail = "000" * (len(SCALES) - index - 1) + "000000"
        graphs.append(
            prefix + (cardinal.three + SPACE | pynutil.insert("001")) + pynutil.delete(words) + pynutil.insert(tail))
    positive = strip_zeros() @ (NONZERO + DIGIT.star)
    return (pynini.union(*graphs) @ positive).optimize()


class Ordinal(Processor):
    """Raw spoken ordinals (except första/andra) to ``21.``."""

    def __init__(self, cardinal):
        super().__init__(name="ordinal", ordertype="itn", token_orders=TOKEN_ORDER)
        digits, teens, ties = ordinal_words()
        self.graph = number_graph(cardinal, digits, teens, ties, "hundrade", "tusende", scale_ordinal)
        self.bare = self.graph
        excluded = pynini.union("första", "andra")
        graph = (pynini.difference(self.input_projection(self.graph), excluded) @ self.graph).optimize()
        self.tagger = self.add_tokens(self.tag_field("integer", graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("integer", self.graph + pynutil.insert(".")))
