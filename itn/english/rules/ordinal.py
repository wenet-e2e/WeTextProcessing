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
from pynini import closure, cross, string_file
from pynini.lib.pynutil import delete, insert

from itn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Ordinal(Processor):

    def __init__(self, cardinal=None):
        super().__init__(name="ordinal", ordertype="itn")
        self.cardinal = cardinal or Cardinal()
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        graph_digit = string_file(get_abs_path("../itn/english/data/ordinals/digit.tsv"))
        graph_teen = string_file(get_abs_path("../itn/english/data/ordinals/teen.tsv"))
        # first => one => 1, twelfth => twelve => 12, twentieth => twenty => 2(0)
        suffix = graph_digit | graph_teen | cross("tieth", "ty") | cross("th", "")
        graph = closure(self.VCHAR) + suffix
        self.graph = pynini.compose(graph, self.cardinal.graph)

        final_graph = insert('integer: "') + self.graph + insert('"')
        self.tagger = self.add_tokens(final_graph)

    def build_verbalizer(self):
        integer = delete("integer:") + self.DELETE_SPACE + delete('"') + self.NOT_QUOTE.plus + delete('"')
        # 1 => 1st, 2 => 2nd, 3 => 3rd, 11 => 11th, 12 => 12th, 13 => 13th, rest => Xth
        ordinal_suffix = (
            cross("11", "11th") | cross("12", "12th") | cross("13", "13th")
            | cross("1", "1st") | cross("2", "2nd") | cross("3", "3rd")
            | insert("th", weight=0.01)
        )
        graph = integer @ pynini.cdrewrite(ordinal_suffix, "", "[EOS]", self.VSIGMA)
        self.verbalizer = self.delete_tokens(graph)
