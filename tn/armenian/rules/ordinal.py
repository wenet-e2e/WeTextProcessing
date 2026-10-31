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
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor


class Ordinal(Processor):
    """Ordinals, e.g. ``5-րդ`` -> ``հինգերորդ`` and ``1-ին`` -> ``առաջին``.

    The tagger keeps the written number and its suffix (``րդ`` or ``ին``)
    without the hyphen; the verbalizer chooses the ordinal form.
    """

    def __init__(self, cardinal):
        super().__init__("ordinal", token_orders={})
        self.cardinal = cardinal
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        # Words ending in ``ը`` (ինը, տասը) take ``ն`` before ``երորդ``.
        self.suffix_rule = pynini.cdrewrite(pynini.cross("ըերորդ", "ներորդ"), "", "[EOS]", self.VSIGMA)
        regular = add_weight(self.cardinal.graph + insert("երորդ"), 1.5)
        special = pynini.string_map([("2", "երկրորդ"), ("3", "երրորդ"), ("4", "չորրորդ")])
        self.first = pynini.cross("1", "առաջին")
        self.graph = ((add_weight(special, 1) | regular) @ self.suffix_rule).optimize()
        # A fraction denominator has no ``րդ``; only ``1-ին`` keeps its suffix.
        first = add_weight(pynini.cross("1-ին", "առաջին"), 1)
        self.denominator = (first | self.graph).optimize()

    def build_tagger(self):
        first = self.tag_field("integer", add_weight(self.first, 1)) + delete("-") + insert(" ")
        first += self.tag_field("suffix", pynini.accep("ին"))
        other = self.tag_field("integer", self.graph) + delete("-") + insert(" ")
        other += self.tag_field("suffix", pynini.accep("րդ"))
        self.tagger = self.add_tokens(first | other)

    def build_verbalizer(self):
        first = self.verbalize_field("integer", self.first)
        first += self.verbalize_field("suffix", delete("ին"), leading_space=True)
        other = self.verbalize_field("integer", self.graph)
        other += self.verbalize_field("suffix", delete("րդ"), leading_space=True)
        self.verbalizer = self.delete_tokens(first | other)
