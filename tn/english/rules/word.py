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
from pynini.examples import plurals
from pynini.lib import pynutil
import sys
from unicodedata import category

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


class Word(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__("word", ordertype="en_tn")
        self.deterministic = deterministic
        s = "!#%&\'()*+,-./:;<=>?@^_`{|}~\""
        punct_symbols_to_exclude = ["[", "]"]
        punct_unicode = [
            chr(i) for i in range(sys.maxunicode)
            if category(chr(i)).startswith("P")
            and chr(i) not in punct_symbols_to_exclude
        ]
        whitelist_symbols = load_labels(
            get_abs_path("english/data/whitelist/symbol.tsv"))
        whitelist_symbols = [x[0] for x in whitelist_symbols]
        self.punct_marks = [
            p for p in punct_unicode + list(s) if p not in whitelist_symbols
        ]

        punct = pynini.union(*self.punct_marks)
        punct = pynini.closure(punct, 1)
        emphasis = (
            pynini.accep("<") +
            ((pynini.closure(self.NOT_SPACE - pynini.union("<", ">"), 1) +
              pynini.closure(pynini.accep("/"), 0, 1))
             | (pynini.accep("/") +
                pynini.closure(self.NOT_SPACE - pynini.union("<", ">"), 1))) +
            pynini.accep(">"))  # noqa
        punct = plurals._priority_union(emphasis, punct,
                                        pynini.closure(self.VCHAR))
        self.punct_graph = punct

        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying word. Considers sentence boundary exceptions.
            e.g. sleep -> word { value: "sleep" }
        """
        default_graph = pynini.closure(
            pynini.difference(self.NOT_SPACE,
                              self.punct_graph.project("input")), 1)
        symbols_to_exclude = (pynini.union("$", "€", "₩", "£", "¥", "#", "%")
                              | self.DIGIT).optimize()
        graph = pynini.closure(
            pynini.difference(self.NOT_SPACE, symbols_to_exclude), 1)
        graph = pynutil.add_weight(graph, self.MIN_NEG_WEIGHT) | default_graph

        # leave phones of format [HH AH0 L OW1] untouched
        phoneme_unit = pynini.closure(self.ALPHA, 1) + pynini.closure(
            self.DIGIT)
        phoneme = (pynini.accep(pynini.escape("[")) +
                   pynini.closure(phoneme_unit + pynini.accep(" ")) +
                   phoneme_unit + pynini.accep(pynini.escape("]")))

        # leave IPA phones of format [ˈdoʊv] untouched, single words and sentences with punctuation marks allowed
        punct_marks = pynini.union(*self.punct_marks).optimize()
        stress = pynini.union("ˈ", "'", "ˌ")
        ipa_phoneme_unit = pynini.string_file(
            get_abs_path("english/data/whitelist/ipa_symbols.tsv"))
        # word in ipa form
        ipa_phonemes = (pynini.closure(stress, 0, 1) +
                        pynini.closure(ipa_phoneme_unit, 1) +
                        pynini.closure(stress | ipa_phoneme_unit))
        # allow sentences of words in IPA format separated with spaces or punct marks
        delim = (punct_marks | pynini.accep(" "))**(1, ...)
        ipa_phonemes = ipa_phonemes + pynini.closure(
            delim + ipa_phonemes) + pynini.closure(delim, 0, 1)
        ipa_phonemes = (pynini.accep(pynini.escape("[")) + ipa_phonemes +
                        pynini.accep(pynini.escape("]"))).optimize()

        if not self.deterministic:
            phoneme = (pynini.accep(pynini.escape("[")) +
                       pynini.closure(pynini.accep(" "), 0, 1) +
                       pynini.closure(phoneme_unit + pynini.accep(" ")) +
                       phoneme_unit + pynini.closure(pynini.accep(" "), 0, 1) +
                       pynini.accep(pynini.escape("]"))).optimize()
            ipa_phonemes = (pynini.accep(pynini.escape("[")) + ipa_phonemes +
                            pynini.accep(pynini.escape("]"))).optimize()

        phoneme |= ipa_phonemes
        self.graph = plurals._priority_union(phoneme.optimize(), graph,
                                             pynini.closure(self.VCHAR))
        final_graph = (pynutil.insert("value: \"") + self.graph +
                       pynutil.insert("\"")).optimize()
        final_graph = self.add_tokens(final_graph)
        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing word
            e.g. word { value: "sleep" } -> sleep
        """
        chars = pynini.closure(self.VCHAR - " ", 1)
        char = pynutil.delete("value:") + self.DELETE_SPACE + pynutil.delete(
            "\"") + chars + pynutil.delete("\"")
        graph = char @ pynini.cdrewrite(pynini.cross(u"\u00A0", " "), "", "",
                                        pynini.closure(self.VCHAR))

        delete_tokens = self.delete_tokens(graph)
        self.verbalizer = delete_tokens.optimize()
