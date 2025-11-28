# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import sys
from unicodedata import category

from pynini import accep, closure, cross, union
from pynini.examples import plurals
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


class Punctuation(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__("p", ordertype="en_tn")
        self.deterministic = deterministic
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying punctuation
            e.g. a, -> w { v: "a" } p { v: "," }
        """
        s = "!#%&'()*+,-./:;<=>?@^_`{|}~"

        punct_symbols_to_exclude = ["[", "]", '"', "\\"]
        punct_unicode = [
            chr(i)
            for i in range(sys.maxunicode)
            if category(chr(i)).startswith("P")
            and chr(i) not in punct_symbols_to_exclude
        ]

        whitelist_symbols = load_labels(
            get_abs_path("english/data/whitelist/symbol.tsv")
        )
        whitelist_symbols = [x[0] for x in whitelist_symbols]
        self.punct_marks = [
            p for p in punct_unicode + list(s) if p not in whitelist_symbols
        ]

        self.punct = union(*self.punct_marks)
        punct = (self.punct | cross("\\", "\\\\\\") | cross('"', '\\"')).plus

        self.emphasis = (
            accep("<")
            + (
                (
                    (self.NOT_SPACE - union("<", ">")).plus + closure(accep("/"), 0, 1)
                )  # noqa
                | (accep("/") + (self.NOT_SPACE - union("<", ">")).plus)
            )
            + accep(">")
        )  # noqa
        punct = plurals._priority_union(self.emphasis, punct, self.VCHAR.star)

        self.graph = punct
        final_graph = (
            insert('v: "')
            + add_weight(accep(" "), -1.0).star
            + punct
            + add_weight(accep(" "), -1.0).star
            + insert('"')
        )
        self.tagger = self.add_tokens(final_graph)

    def build_verbalizer(self):
        punct = closure(
            self.punct | self.emphasis | cross("\\\\\\", "\\") | cross('\\"', '"'), 1
        )
        verbalizer = (
            delete('v: "')
            + add_weight(accep(" "), -1.0).star
            + punct
            + add_weight(accep(" "), -1.0).star
            + delete('"')
        )
        self.verbalizer = self.delete_tokens(verbalizer)
