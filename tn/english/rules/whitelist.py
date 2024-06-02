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

INPUT_CASED = "cased"
INPUT_LOWER_CASED = "lower_cased"

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels, augment_labels_with_punct_at_end
from tn.english.rules.measure import SINGULAR_TO_PLURAL
from tn.english.rules.roman import get_names


class WhiteList(Processor):

    def __init__(self,
                 deterministic: bool = False,
                 input_case: str = INPUT_CASED):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__('whitelist', ordertype="en_tn")
        self.deterministic = deterministic
        self.input_case = input_case
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying whitelist, e.g.
            misses -> tokens { name: "mrs" }
            for non-deterministic case: "Dr. Abc" ->
                whitelist { name: "drive" } word { value: "Abc" }
                whitelist { name: "doctor" } word { value: "Abc" }
                whitelist { name: "Dr." } word { vale: "Abc" }
        This class has highest priority among all classifier grammars. Whitelisted tokens are defined and loaded from "data/whitelist.tsv".
        """

        def _get_whitelist_graph(input_case,
                                 file,
                                 keep_punct_add_end: bool = False):
            whitelist = load_labels(file)
            if input_case == INPUT_LOWER_CASED:
                whitelist = [[x.lower(), y] for x, y in whitelist]
            else:
                whitelist = [[x, y] for x, y in whitelist]

            if keep_punct_add_end:
                whitelist.extend(augment_labels_with_punct_at_end(whitelist))

            graph = pynini.string_map(whitelist)
            return graph

        graph = _get_whitelist_graph(
            self.input_case, get_abs_path("english/data/whitelist/tts.tsv"))
        graph |= pynini.compose(
            pynini.difference(pynini.closure(self.VCHAR),
                              pynini.accep("/")).optimize(),
            _get_whitelist_graph(
                self.input_case,
                get_abs_path("english/data/whitelist/symbol.tsv")),
        ).optimize()

        if self.deterministic:
            names = get_names()
            graph |= (pynini.cross(pynini.union("st", "St", "ST"), "Saint") +
                      pynini.closure(pynutil.delete(".")) + pynini.accep(" ") +
                      names)
        else:
            graph |= _get_whitelist_graph(
                self.input_case,
                get_abs_path("english/data/whitelist/alternatives.tsv"),
                keep_punct_add_end=True)

        for x in [".", ". "]:
            graph |= (self.UPPER +
                      pynini.closure(pynutil.delete(x) + self.UPPER, 2) +
                      pynini.closure(pynutil.delete("."), 0, 1))

        if not self.deterministic:
            multiple_forms_whitelist_graph = get_formats(
                get_abs_path(
                    "english/data/whitelist/alternatives_all_format.tsv"))
            graph |= multiple_forms_whitelist_graph

            graph_unit = pynini.string_file(
                get_abs_path("english/data/measure/unit.tsv")
            ) | pynini.string_file(
                get_abs_path("english/data/measure/unit_alternatives.tsv"))
            graph_unit_plural = graph_unit @ SINGULAR_TO_PLURAL
            units_graph = pynini.compose(self.VCHAR**(3, ...),
                                         graph_unit | graph_unit_plural)
            graph |= units_graph

        # convert to states only if comma is present before the abbreviation to avoid converting all caps words,
        # e.g. "IN", "OH", "OK"
        # TODO or only exclude above?
        states = load_labels(get_abs_path("english/data/address/state.tsv"))
        additional_options = []
        for x, y in states:
            if self.input_case == INPUT_LOWER_CASED:
                x = x.lower()
            additional_options.append((x, f"{y[0]}.{y[1:]}"))
            if not self.deterministic:
                additional_options.append((x, f"{y[0]}.{y[1:]}."))

        states.extend(additional_options)
        state_graph = pynini.string_map(states)
        graph |= pynini.closure(self.NOT_SPACE, 1) + pynini.union(
            ", ", ",") + pynini.invert(state_graph).optimize()

        self.graph = graph.optimize()

        fianl_graph = (pynutil.insert("name: \"") + self.graph +
                       pynutil.insert("\"")).optimize()
        self.tagger = self.add_tokens(fianl_graph)

    def build_verbalizer(self):
        graph = (pynutil.delete("name:") + self.DELETE_SPACE +
                 pynutil.delete("\"") + pynini.closure(self.VCHAR - " ", 1) +
                 pynutil.delete("\""))
        final_graph = graph.optimize()
        self.verbalizer = self.delete_tokens(final_graph)


def get_formats(input_f, input_case=INPUT_CASED, is_default=True):
    """
    Adds various abbreviation format options to the list of acceptable input forms
    """
    multiple_formats = load_labels(input_f)
    additional_options = []
    for x, y in multiple_formats:
        if input_case == INPUT_LOWER_CASED:
            x = x.lower()
        additional_options.append((
            f"{x}.",
            y))  # default "dr" -> doctor, this includes period "dr." -> doctor
        additional_options.append(
            (f"{x[0].upper() + x[1:]}",
             f"{y[0].upper() + y[1:]}"))  # "Dr" -> Doctor
        additional_options.append(
            (f"{x[0].upper() + x[1:]}.",
             f"{y[0].upper() + y[1:]}"))  # "Dr." -> Doctor
    multiple_formats.extend(additional_options)

    if not is_default:
        multiple_formats = [
            (x, f"|raw_start|{x}|raw_end||norm_start|{y}|norm_end|")
            for (x, y) in multiple_formats
        ]

    multiple_formats = pynini.string_map(multiple_formats)
    return multiple_formats
