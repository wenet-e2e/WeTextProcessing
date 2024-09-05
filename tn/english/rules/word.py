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

from pynini import difference, union
from pynini.lib.pynutil import delete, insert

from tn.english.rules.punctuation import Punctuation
from tn.processor import Processor


class Word(Processor):

    def __init__(self):
        super().__init__("w", ordertype="en_tn")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying word. Considers sentence boundary exceptions.
            e.g. sleep -> w { v: "sleep" }
        """
        punct = Punctuation().graph
        default_graph = difference(self.NOT_SPACE, punct.project("input"))
        symbols_to_exclude = union("$", "€", "₩", "£", "¥", "#", "%") | self.DIGIT
        self.char = difference(default_graph, symbols_to_exclude)
        graph = (insert('v: "') + self.char.plus + insert('"')).optimize()
        final_graph = self.add_tokens(graph)
        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing word
            e.g. w { v: "sleep" } -> sleep
        """
        graph = delete("v: ") + delete('"') + self.char.plus + delete('"')
        final_graph = self.delete_tokens(graph)
        self.verbalizer = final_graph.optimize()
