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
from tn.spanish.rules.utils import (
    CHAR,
    DELETE_SPACE,
    EOS,
    data_file,
    data_labels,
    rewrite,
    roman_to_int,
    shift_number_gender,
    strip_accent,
)

TOKEN_ORDER = {}


def one_to_one_thousand(cardinal):
    """Spoken masculine cardinals for 1..999 (an acceptor)."""
    numbers = pynini.string_map([str(number) for number in range(1, 1000)]) @ cardinal.graph
    return pynini.project(numbers, "output").optimize()


class Ordinal(Processor):
    """Raw ordinal digits (or Roman numerals) and their written gender suffix."""

    def __init__(self, cardinal, deterministic=True):
        super().__init__("ordinal", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic

        def table(name):
            return pynini.invert(data_file("ordinals/" + name + ".tsv"))

        digit, teens, twenties = table("digit"), table("teen"), table("twenties")
        ties, hundreds = table("ties"), table("hundreds")
        if not deterministic:
            ties |= pynini.cross("sesenta", "setuagésimo")
            teens |= pynini.cross("once", "decimoprimero") | pynini.cross("doce", "decimosegundo")
            digit |= pynini.cross("nueve", "nono") | pynini.cross("siete", "sétimo")

        tens_component = teens | ties + pynini.closure(pynini.cross(" y ", " ") + digit, 0, 1) | twenties
        hundred_component = pynini.union(
            hundreds + pynini.closure(" " + pynini.union(tens_component, digit), 0, 1), tens_component, digit,
        )
        self.one_to_one_thousand = one_to_one_thousand(cardinal)
        thousands = pynini.cross("mil", "milésimo")
        graph_thousands = strip_accent(self.one_to_one_thousand) + " " + thousands
        graph_thousands = rewrite(graph_thousands, DELETE_SPACE, right="milésimo")
        graph_thousands |= thousands
        self.multiples_of_thousand = (cardinal.graph @ graph_thousands).optimize()
        if not deterministic:
            graph_thousands |= (self.one_to_one_thousand @ hundred_component) + " " + thousands
        graph_thousands += pynini.closure(" " + hundred_component, 0, 1)

        graph = cardinal.graph @ (graph_thousands | hundred_component)
        if not deterministic:
            split = pynini.cross("decimo", "décimo ") | pynini.cross("vigesimo", "vigésimo ")
            graph |= rewrite(graph, split, right=CHAR)
        # Written digits to masculine singular ordinal words.
        self.graph = rewrite(graph, pynutil.delete("o"), right="octavo")

        exceptions = pynini.project(data_file("ordinals/roman_exceptions.tsv"), "input")
        roman = roman_to_int(self.graph)
        self.roman_graph = ((pynini.project(roman, "input") - exceptions.arcsort()) @ roman).optimize()

        end = pynini.union(" ", EOS)
        self.fem = shift_number_gender(rewrite(self.graph, pynini.cross("o", "a"), right=end))
        apocope = pynini.cross("tercero", "tercer") | pynini.cross("primero", "primer")
        apocope |= pynini.cross("undécimo", "decimoprimer")
        self.apocope = rewrite(self.graph, apocope)
        roman_fem = shift_number_gender(rewrite(self.roman_graph, pynini.cross("o", "a"), right=end))

        suffixes = {}
        for suffix, feature in data_labels("ordinals/gender_suffix.tsv"):
            suffixes.setdefault(feature, []).append(suffix)
        period = pynini.closure(pynini.accep("."), 0, 1)
        suffix = {feature: (period + pynini.union(*items)).optimize() for feature, items in suffixes.items()}
        self.suffix_graph = pynini.union(*suffix.values()).optimize()

        def plural(graph):
            return rewrite(graph, pynutil.insert("s"), pynini.union("o", "a"), end)

        forms = {"gender_masc": self.graph, "gender_fem": self.fem, "apocope": self.apocope}
        if not deterministic:
            forms = {feature: graph | pynutil.add_weight(plural(graph), 0.001) for feature, graph in forms.items()}
        roman_forms = self.roman_graph | pynutil.add_weight(roman_fem, 0.001)
        if not deterministic:
            # Roman numerals carry no written suffix: offer every morphology.
            roman_forms |= pynutil.add_weight(rewrite(self.roman_graph, apocope), 0.001)
            roman_forms |= pynutil.add_weight(plural(roman_forms), 0.001)

        self.tagger = self.add_tokens(
            self.tag_field("integer", self.graph) + pynutil.insert(" ") + self.tag_field("suffix", self.suffix_graph)
            | self.tag_field("integer", self.roman_graph)
        )
        verbalizers = [
            self.verbalize_field("integer", forms[feature])
            + self.verbalize_field("suffix", pynutil.delete(suffix[feature]), leading_space=True)
            for feature in forms
        ]
        verbalizers.append(self.verbalize_field("integer", roman_forms))
        self.verbalizer = self.delete_tokens(pynini.union(*verbalizers))
