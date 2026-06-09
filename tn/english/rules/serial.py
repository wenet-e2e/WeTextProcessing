# Copyright (c) 2024 Zhendong Peng (pzd17@tsinghua.org.cn)
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

from tn.english.rules.cardinal import Cardinal
from tn.english.rules.ordinal import Ordinal
from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


class Serial(Processor):

    def __init__(self, deterministic: bool = False, cardinal=None, ordinal=None):
        super().__init__("serial", ordertype="en_tn")
        self.deterministic = deterministic
        self.cardinal = cardinal or Cardinal(deterministic)
        self.ordinal = ordinal or Ordinal(deterministic, cardinal=self.cardinal)
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        cardinal = self.cardinal

        num_graph = pynini.compose(self.DIGIT ** (6, ...), cardinal.single_digits_graph).optimize()
        num_graph |= pynini.compose(self.DIGIT ** (1, 5), cardinal.graph).optimize()
        num_graph |= pynini.compose(
            pynini.accep("0") + pynini.closure(self.DIGIT), cardinal.single_digits_graph
        ).optimize()

        symbols_graph = pynini.string_file(get_abs_path("english/data/whitelist/symbol.tsv")).optimize()
        symbols_graph |= pynini.cross("#", "hash")
        num_graph |= symbols_graph

        symbols = [x[0] for x in load_labels(get_abs_path("english/data/whitelist/symbol.tsv"))]
        symbols = pynini.union(*symbols)
        digit_symbol = self.DIGIT | symbols

        graph_with_space = pynini.compose(
            pynini.cdrewrite(pynutil.insert(" "), self.ALPHA | symbols, digit_symbol, self.VSIGMA),
            pynini.cdrewrite(pynutil.insert(" "), digit_symbol, self.ALPHA | symbols, self.VSIGMA),
        )

        delimiter = pynini.accep("-") | pynini.accep("/") | pynini.accep(" ")

        alphas = pynini.closure(self.ALPHA, 1)
        letter_num = alphas + delimiter + num_graph
        num_letter = pynini.closure(num_graph + delimiter, 1) + alphas
        next_alpha_or_num = pynini.closure(delimiter + (alphas | num_graph))

        serial_graph = letter_num + next_alpha_or_num
        serial_graph |= num_letter + next_alpha_or_num
        serial_graph |= (
            num_graph + delimiter + num_graph + delimiter + num_graph + pynini.closure(delimiter + num_graph)
        )

        serial_graph = pynini.compose(
            pynini.difference(self.VSIGMA, pynini.project(self.ordinal.graph, "input")), serial_graph
        ).optimize()

        serial_graph |= (
            pynini.closure(self.NOT_SPACE, 1)
            + (pynini.cross("^2", " squared") | pynini.cross("^3", " cubed")).optimize()
        )

        serial_graph = (
            pynini.closure((serial_graph | num_graph | alphas) + delimiter)
            + serial_graph
            + pynini.closure(delimiter + (serial_graph | num_graph | alphas))
        )

        serial_graph |= pynini.compose(graph_with_space, serial_graph.optimize()).optimize()
        serial_graph = pynini.compose(pynini.closure(self.NOT_SPACE, 2), serial_graph).optimize()

        serial_graph = pynini.compose(
            pynini.difference(
                self.VSIGMA, pynini.closure(self.ALPHA, 1) + pynini.accep("/") + pynini.closure(self.ALPHA, 1)
            ),
            serial_graph,
        )

        self.graph = serial_graph.optimize()
        graph = pynutil.insert('name: "') + self.graph + pynutil.insert('"')
        self.tagger = self.add_tokens(graph)

    def build_verbalizer(self):
        graph = (
            pynutil.delete("name:")
            + self.DELETE_SPACE
            + pynutil.delete('"')
            + self.NOT_QUOTE.plus
            + pynutil.delete('"')
        )
        self.verbalizer = self.delete_tokens(graph)
