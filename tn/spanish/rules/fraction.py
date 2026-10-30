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
    ACCENTS,
    CHAR,
    DIGIT,
    NOT_SPACE,
    WHITE_SPACE,
    data_file,
    data_labels,
    restrict,
    rewrite,
    shift_cardinal_gender,
    strip_cardinal_apocope,
)

TOKEN_ORDER = {}


def remove_marks(graph):
    """No accents before an -avo/-ésimo suffix and no doubled vowels."""
    graph = rewrite(
        graph, ACCENTS, pynini.union(" ", "[BOS]") + pynini.closure(NOT_SPACE),
        pynini.closure(NOT_SPACE) + pynini.union("avo", "ava", "ésimo", "ésima"),
    )
    return rewrite(graph, pynini.string_map([("aa", "a"), ("oo", "o")]))


def merge(graph):
    """A denominator is one orthographic word, with "y" spelled "i"."""
    graph = rewrite(graph, pynini.cross(" y ", "i"))
    graph = rewrite(graph, pynutil.delete(WHITE_SPACE), right=CHAR)
    return remove_marks(graph)


def merge_stem(graph):
    """Alternative spelling that only joins the -avo suffix to its stem."""
    graph = rewrite(graph, pynutil.delete(WHITE_SPACE), right=pynini.union("avo", "ava", "avos", "avas"))
    return remove_marks(graph)


class Fraction(Processor):
    """Raw numerator/denominator digits; V picks the ordinal, -avo or sobre reading."""

    def __init__(self, cardinal, ordinal, deterministic=True):
        super().__init__("fraction", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        cardinal_graph = cardinal.graph

        # Denominators read as ordinals: 2-10, hundreds and powers of ten.
        small = [str(number) for number in range(2, 11)]
        if not deterministic:
            small += [str(number) for number in range(20, 100, 10)]
        small_graph = pynini.string_map(small) @ ordinal.graph
        small_graph = rewrite(small_graph, data_file("fractions/ordinal_exceptions.tsv"))
        hundreds = pynini.string_map([str(number) for number in range(100, 1000, 100)]) @ ordinal.graph
        powers = data_file("fractions/powers_of_ten.tsv")
        one_to_one_thousand = ordinal.one_to_one_thousand + " "
        higher = (
            pynini.closure(one_to_one_thousand, 0, 1) + pynini.closure("mil ", 0, 1)
            + pynini.closure(one_to_one_thousand, 0, 1) + powers
        )
        higher = cardinal_graph @ higher
        higher = rewrite(higher, pynutil.delete("un "), "[BOS]", pynini.project(powers, "output"))
        higher = (ordinal.multiples_of_thousand | hundreds | higher).optimize()
        self.ordinal_denominator = (higher | small_graph).optimize()

        # Other denominators take the -avo suffix ("doceavo").
        blocked = pynini.project(higher, "input") | pynini.string_map([str(number) for number in range(2, 11)]) | "0"
        root_graph = cardinal_graph
        if deterministic:
            root_graph = pynini.closure(DIGIT, 1, 2) @ root_graph
        self.root_denominator = restrict(root_graph, blocked)
        self.denominator_graph = (
            self.ordinal_denominator | self.root_denominator | pynutil.add_weight(cardinal_graph, 0.001)
        ).optimize()
        self.symbols = data_file("fractions/fraction_symbols.tsv")

        sign_t = cardinal.sign_tagger
        integer_t = pynini.closure(self.tag_field("integer_part", cardinal_graph) + pynini.accep(" "), 0, 1)
        fraction_t = (
            self.tag_field("numerator", cardinal_graph) + pynutil.delete(pynini.union("/", " / "))
            + pynutil.insert(" ") + self.tag_field("denominator", self.denominator_graph)
            | self.tag_field("value", pynini.project(self.symbols, "input"))
        )
        self.fields_tagger = (sign_t + integer_t + fraction_t).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)

        words = self._word_forms(cardinal)
        masc_fields, fem_fields = self._fractions(
            words,
            lambda graph: self.verbalize_field("numerator", graph),
            lambda graph: self.verbalize_field("denominator", graph, leading_space=True),
        )
        # A vulgar fraction symbol spells its numerator and denominator.
        values = [row[1].split("/") for row in data_labels("fractions/fraction_symbols.tsv")]
        numerators = pynini.union(*[value[0] for value in values])
        denominators = pynini.union(*[value[1] for value in values])
        masc_plain, fem_plain = self._fractions(
            words,
            lambda graph: numerators @ graph,
            lambda graph: pynutil.delete("/") + (denominators @ graph),
        )
        masc_symbol = self.verbalize_field("value", self.symbols @ masc_plain)
        fem_symbol = self.verbalize_field("value", self.symbols @ fem_plain)

        integer = strip_cardinal_apocope(cardinal_graph)
        conjunction = pynutil.delete(" ") + pynutil.insert(" y ")
        masc_integer = pynini.closure(self.verbalize_field("integer_part", integer) + conjunction, 0, 1)
        fem_integer = shift_cardinal_gender(integer)
        fem_integer = pynini.closure(self.verbalize_field("integer_part", fem_integer) + conjunction, 0, 1)
        sign_v = cardinal.sign_verbalizer
        self.masc_verbalizer = (sign_v + masc_integer + (masc_fields | masc_symbol)).optimize()
        self.fem_verbalizer = (sign_v + fem_integer + (fem_fields | fem_symbol)).optimize()
        self.fields_verbalizer = (self.masc_verbalizer | pynutil.add_weight(self.fem_verbalizer, 0.001)).optimize()
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)

    def _word_forms(self, cardinal):
        """Raw digits to every spoken numerator and denominator form."""
        cardinal_graph = cardinal.graph
        forms = {"one": pynini.accep("1") @ cardinal_graph, "other": restrict(cardinal_graph, pynini.accep("1"))}
        forms["over"] = strip_cardinal_apocope(cardinal_graph)
        forms["over_denominator"] = pynutil.add_weight(forms["over"], 0.001)
        stem = self.root_denominator + pynutil.insert(" avo")
        singular = stem | self.ordinal_denominator
        if not self.deterministic:
            exceptions = pynini.string_map([("once avo", "undécimo"), ("doce avo", "duodécimo")])
            singular |= stem @ exceptions
        singular = singular.optimize()
        plural = (singular + pynutil.insert("s")).optimize()
        forms["singular"] = merge(singular)
        forms["plural"] = merge(plural)
        forms["other_fem"] = shift_cardinal_gender(forms["other"])
        forms["media"] = (singular @ pynini.cross("medio", "media")).optimize()
        forms["medias"] = (plural @ pynini.cross("medios", "medias")).optimize()
        if not self.deterministic:
            singular_fem = rewrite(shift_cardinal_gender(singular), pynini.cross("o", "a"), right="[EOS]")
            singular_fem = rewrite(singular_fem, pynini.cross("tercia", "tercera"))
            plural_fem = (singular_fem + pynutil.insert("s")).optimize()
            forms["one_fem"] = shift_cardinal_gender(forms["one"])
            forms["over_fem"] = forms["one_fem"] | forms["other_fem"]
            forms["over_denominator_fem"] = pynutil.add_weight(shift_cardinal_gender(cardinal_graph), 0.001)
            forms["singular_stem"] = merge_stem(singular)
            forms["plural_stem"] = merge_stem(plural)
            forms["singular_parte"] = (merge_stem(singular_fem) | merge(singular_fem)) + pynutil.insert(" parte")
            forms["plural_partes"] = (merge_stem(plural_fem) | merge(plural_fem)) + pynutil.insert(" partes")
        return forms

    def _fractions(self, words, numerator, denominator):
        """Masculine and feminine numerator/denominator readings."""
        space = pynutil.insert(" ")
        over = pynutil.insert(" sobre ")
        default = numerator(words["other"]) + space + denominator(words["plural"])
        with_one = numerator(words["one"]) + space + denominator(words["singular"])
        with_cardinal = numerator(words["over"]) + over + denominator(words["over_denominator"])
        if not self.deterministic:
            with_cardinal |= numerator(words["over_fem"]) + over + denominator(words["over_denominator_fem"])
            with_one_fem = numerator(words["one_fem"]) + space + denominator(words["singular_parte"])
            with_one_fem = rewrite(with_one_fem, pynini.cross("una media", "media"))
            default_fem = numerator(words["other_fem"]) + space + denominator(words["plural_partes"])
            default |= pynutil.add_weight(numerator(words["other"]) + space + denominator(words["plural_stem"]), 0.001)
            default |= pynutil.add_weight(default_fem, 0.001)
            with_one |= pynutil.add_weight(numerator(words["one"]) + space + denominator(words["singular_stem"]), 0.001)
            with_one |= pynutil.add_weight(with_one_fem, 0.001)
        # "medio", not "un medio".
        with_one = rewrite(with_one, pynini.cross("un medio", "medio"))
        masc = (with_one | default | with_cardinal).optimize()

        default |= numerator(words["other_fem"]) + space + denominator(words["medias"])
        with_one |= numerator(pynutil.delete(pynini.project(words["one"], "input"))) + denominator(words["media"])
        fem = (with_one | default | with_cardinal).optimize()
        return masc, fem
