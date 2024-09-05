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

from pynini import accep, compose, closure, difference, string_map
from pynini.lib.pynutil import delete, insert

from tn.english.rules.ordinal import Ordinal
from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


class Roman(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__("roman", ordertype="en_tn")
        self.deterministic = deterministic
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying roman numbers:
            e.g. "IV" -> roman { integer: "four" }
        """
        roman_dict = load_labels(get_abs_path("english/data/roman/roman_to_spoken.tsv"))
        default_graph = string_map(roman_dict).optimize()
        default_graph = insert('integer: "') + default_graph + insert('"')
        ordinal_limit = 19

        if self.deterministic:
            # exclude "I"
            start_idx = 1
        else:
            start_idx = 0

        graph_teens = string_map(
            [x[0] for x in roman_dict[start_idx:ordinal_limit]]
        ).optimize()

        # roman numerals up to ordinal_limit with a preceding name are converted to ordinal form
        names = get_names()
        graph = (
            insert('key_the_ordinal: "')
            + names
            + insert('"')
            + accep(" ")
            + graph_teens @ default_graph
        ).optimize()

        # single symbol roman numerals with preceding key words (multiple formats) are converted to cardinal form
        key_words = []
        for k_word in load_labels(get_abs_path("english/data/roman/key_word.tsv")):
            key_words.append(k_word)
            key_words.append([k_word[0][0].upper() + k_word[0][1:]])
            key_words.append([k_word[0].upper()])

        key_words = string_map(key_words).optimize()
        graph |= (
            insert('key_cardinal: "')
            + key_words
            + insert('"')
            + accep(" ")
            + default_graph
        ).optimize()

        if self.deterministic:
            # two digit roman numerals up to 49
            roman_to_cardinal = compose(
                closure(self.ALPHA, 2),
                (
                    insert('default_cardinal: "default" ')
                    + (string_map([x[0] for x in roman_dict[:50]]).optimize())
                    @ default_graph
                ),
            )
            graph |= roman_to_cardinal
        else:
            # two or more digit roman numerals
            roman_to_cardinal = compose(
                difference(self.VSIGMA, "I"),
                (
                    insert('default_cardinal: "default" integer: "')
                    + string_map(roman_dict).optimize()
                    + insert('"')
                ),
            ).optimize()
            graph |= roman_to_cardinal

        # convert three digit roman or up with suffix to ordinal
        roman_to_ordinal = compose(
            closure(self.ALPHA, 3),
            (
                insert('default_ordinal: "default" ')
                + graph_teens @ default_graph
                + delete("th")
            ),
        )

        graph |= roman_to_ordinal
        graph = self.add_tokens(graph.optimize())

        self.tagger = graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing roman numerals
            e.g. tokens { roman { integer: "one" } } -> one
        """
        suffix = Ordinal(self.deterministic).suffix

        cardinal = self.NOT_QUOTE.star
        ordinal = compose(cardinal, suffix)

        graph = (
            delete('key_cardinal: "')
            + self.NOT_QUOTE.plus
            + delete('"')
            + accep(" ")
            + delete('integer: "')
            + cardinal
            + delete('"')
        ).optimize()

        graph |= (
            delete('default_cardinal: "default" integer: "') + cardinal + delete('"')
        ).optimize()

        graph |= (
            delete('default_ordinal: "default" integer: "') + ordinal + delete('"')
        ).optimize()

        graph |= (
            delete('key_the_ordinal: "')
            + self.NOT_QUOTE.plus
            + delete('"')
            + accep(" ")
            + delete('integer: "')
            + insert("the ").ques
            + ordinal
            + delete('"')
        ).optimize()

        delete_tokens = self.delete_tokens(graph)
        self.verbalizer = delete_tokens.optimize()


def get_names():
    """
    Returns the graph that matched common male and female names.
    """
    male_labels = load_labels(get_abs_path("english/data/roman/male.tsv"))
    female_labels = load_labels(get_abs_path("english/data/roman/female.tsv"))
    male_labels.extend([[x[0].upper()] for x in male_labels])
    female_labels.extend([[x[0].upper()] for x in female_labels])
    names = string_map(male_labels).optimize()
    names |= string_map(female_labels).optimize()
    return names
