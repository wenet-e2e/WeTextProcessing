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

from itn.hungarian.rules.utils import data_map, suffix_rewrite
from tn.processor import Processor

TOKEN_ORDER = {}


class Ordinal(Processor):
    """Ordinals, e.g. ``harmincegyedik`` -> ``31.``.

    The spoken word stays raw; the verbalizer writes plain digits and the
    ordinal dot. A bare ``első`` ("first") is not tagged on its own.
    """

    def __init__(self, cardinal):
        super().__init__("ordinal", ordertype="itn", token_orders=TOKEN_ORDER)
        to_ordinal = suffix_rewrite(data_map("ordinals/endings.tsv"), data_map("ordinals/exceptional.tsv"))
        spoken = (cardinal.spoken @ to_ordinal).optimize()
        self.graph = (pynini.invert(spoken) + insert(".")).optimize()
        # Denominators: "harmad" -> "3", "fél" -> "2".
        exceptions = pynini.string_map([("első", "egyed"), ("második", "fél")])
        fractional = spoken @ suffix_rewrite(delete("ik"), exceptions)
        self.fractional = pynini.invert(fractional).optimize()
        standalone = pynini.difference(pynini.project(self.graph, "input"), "első") @ self.graph
        self.tagger = self.add_tokens(self.tag_field("integer", standalone))
        self.verbalizer = self.delete_tokens(self.verbalize_field("integer", self.graph))
