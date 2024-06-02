# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2024, WENET COMMUNITY.  Xingchen Song (sxc19@tsinghua.org.cn).
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
from tn.utils import get_abs_path
from tn.english.rules.cardinal import Cardinal


class Ordinal(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__("ordinal", ordertype="en_tn")
        self.deterministic = deterministic
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying ordinal, e.g.
            13th -> ordinal { integer: "thirteen" }
        """
        cardinal = Cardinal(self.deterministic)
        cardinal_graph = cardinal.graph
        cardinal_format = pynini.closure(self.DIGIT | pynini.accep(","))
        st_format = (pynini.closure(cardinal_format +
                                    (self.DIGIT - "1"), 0, 1) +
                     pynini.accep("1") +
                     pynutil.delete(pynini.union("st", "ST", "ˢᵗ")))
        nd_format = (pynini.closure(cardinal_format +
                                    (self.DIGIT - "1"), 0, 1) +
                     pynini.accep("2") +
                     pynutil.delete(pynini.union("nd", "ND", "ⁿᵈ")))
        rd_format = (pynini.closure(cardinal_format +
                                    (self.DIGIT - "1"), 0, 1) +
                     pynini.accep("3") +
                     pynutil.delete(pynini.union("rd", "RD", "ʳᵈ")))
        th_format = pynini.closure(
            (self.DIGIT - "1" - "2" - "3")
            | (cardinal_format + "1" + self.DIGIT)
            | (cardinal_format + (self.DIGIT - "1") +
               (self.DIGIT - "1" - "2" - "3")),
            1,
        ) + pynutil.delete(pynini.union("th", "TH", "ᵗʰ"))
        self.graph = (st_format | nd_format | rd_format
                      | th_format) @ cardinal_graph
        final_graph = pynutil.insert(
            "integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing ordinal, e.g.
            ordinal { integer: "thirteen" } } -> thirteenth
        """
        graph_digit = pynini.string_file(
            get_abs_path("english/data/ordinal/digit.tsv")).invert()
        graph_teens = pynini.string_file(
            get_abs_path("english/data/ordinal/teen.tsv")).invert()

        graph = (pynutil.delete("integer:") + self.DELETE_SPACE +
                 pynutil.delete("\"") + pynini.closure(self.NOT_QUOTE, 1) +
                 pynutil.delete("\""))
        convert_rest = pynutil.insert("th")

        suffix = pynini.cdrewrite(
            graph_digit | graph_teens | pynini.cross("ty", "tieth")
            | convert_rest,
            "",
            "[EOS]",
            pynini.closure(self.VCHAR),
        ).optimize()
        self.graph_v = pynini.compose(graph, suffix)
        self.suffix = suffix
        delete_tokens = self.delete_tokens(self.graph_v)
        self.verbalizer = delete_tokens.optimize()
