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
from tn.swedish.rules.utils import rewrite, tsv

TOKEN_ORDER = {}


class Decimal(Processor):
    """Raw comma decimals with optional quantity words (``1,5 mn``)."""

    def __init__(self, cardinal, deterministic=True):
        super().__init__("decimal", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        self.cardinal = cardinal
        self.fractional = cardinal.fraction_digits_read

        quantities = tsv("numbers/millions")
        abbr = tsv("numbers/millions_abbr")
        plural = (
            quantities + "er"
            | rewrite(quantities, pynutil.insert("er"), "", "[EOS]")
            | abbr + pynutil.insert("er")
        )
        self.quantity_singular = (quantities | abbr).optimize()
        self.quantity_plural = plural.optimize()
        self.tusen = pynini.accep("tusen")
        self._quantity_fields = None

        sign_t = (self.tag_field("negative", cardinal.sign) + pynutil.insert(" ")).ques
        sign_v = (self.verbalize_field("negative", cardinal.sign) + pynutil.delete(" ") + pynutil.insert(" ")).ques
        tagger, verbalizer = self.fields(en=False)
        self.tagger = self.add_tokens(sign_t + tagger)
        self.verbalizer = self.delete_tokens(sign_v + verbalizer)

    def _integer(self, graph):
        return self.tag_field("integer_part", graph), self.verbalize_field("integer_part", graph)

    def _fractional(self):
        tagger = pynutil.delete(",") + pynutil.insert(" ") + self.tag_field("fractional_part", self.fractional)
        verbalizer = pynutil.insert(" komma ") + self.verbalize_field(
            "fractional_part", self.fractional, leading_space=True)
        return tagger, verbalizer

    def _quantity(self, graph):
        tagger = (pynini.accep(" ") | pynutil.insert(" ")) + self.tag_field("quantity", graph)
        verbalizer = pynutil.insert(" ") + self.verbalize_field("quantity", graph, leading_space=True)
        return tagger, verbalizer

    @staticmethod
    def _join(*parts):
        tagger = pynini.accep("")
        verbalizer = pynini.accep("")
        for part_tagger, part_verbalizer in parts:
            tagger += part_tagger
            verbalizer += part_verbalizer
        return tagger, verbalizer

    def quantity_fields(self):
        """Numbers followed by a quantity word, independent of the noun gender."""

        if self._quantity_fields is None:
            self._quantity_fields = self._build_quantity_fields()
        return self._quantity_fields

    def _build_quantity_fields(self):
        cardinal = self.cardinal
        frac = self._fractional()
        plural = self._quantity(self.quantity_plural)
        tusen = self._quantity(self.tusen)
        paths = [
            self._join(self._integer(cardinal.up_to_thousand_no_one_en), plural),
            self._join(self._integer(cardinal.up_to_thousand_no_one), tusen),
            self._join(self._integer(pynini.cross("1", "ett")), tusen),
            self._join(self._integer(pynini.cross("1", "en")), self._quantity(self.quantity_singular)),
            self._join(self._integer(cardinal.graph_en), frac, plural),
            self._join(self._integer(cardinal.graph), frac, tusen),
        ]
        return self._union(paths)

    def plain_fields(self, en=False):
        integer = self.cardinal.graph_en if en else self.cardinal.graph
        return self._join(self._integer(integer), self._fractional())

    def fields(self, en=False):
        """Unsigned decimal fields: (tagger, verbalizer)."""

        return self._union([self.plain_fields(en), self.quantity_fields()])

    @staticmethod
    def _union(paths):
        tagger = pynini.union(*[path[0] for path in paths]).optimize()
        verbalizer = pynini.union(*[path[1] for path in paths]).optimize()
        return tagger, verbalizer
