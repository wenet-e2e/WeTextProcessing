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

"""Spanish word-level morphology shared by the TN verbalizers.

Every helper here maps a *bounded* spoken-word graph to its inflected
variants. Rewrite rules are cached and composed one after another onto the
bounded graph, never pre-composed with each other over the full alphabet.
"""

from functools import lru_cache

import pynini
from pynini.lib import byte, pynutil, utf8

from tn.utils import get_abs_path, load_labels

SIGMA = utf8.VALID_UTF8_CHAR.star
CHAR = utf8.VALID_UTF8_CHAR
DIGIT = byte.DIGIT
ALPHA = byte.ALPHA
WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", "\u00a0").optimize()
NOT_SPACE = pynini.difference(CHAR, WHITE_SPACE).optimize()
DELETE_SPACE = pynutil.delete(WHITE_SPACE).star
EOS = pynini.accep("[EOS]")
CARDINAL_SEPARATOR = pynini.union(".", " ")
DECIMAL_SEPARATOR = pynini.accep(",")


def data_path(name):
    return get_abs_path("spanish/data/" + name)


def data_file(name):
    return pynini.string_file(data_path(name))


def data_labels(name):
    return load_labels(data_path(name))


@lru_cache(maxsize=None)
def number_words(name):
    return pynini.project(data_file("numbers/" + name + ".tsv"), "input").optimize()


ACCENTS = pynini.string_map([("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u")])
ONES = pynini.union("un", "ún")


@lru_cache(maxsize=None)
def _rules():
    digits = number_words("digit")
    hundreds = number_words("hundreds")
    one_to_one_hundred = pynini.union(
        digits, "uno", number_words("ties"), number_words("teen"), number_words("twenties"),
        number_words("ties") + pynini.accep(" y ") + digits,
    )
    fem_hundreds = (hundreds @ pynini.cdrewrite(pynini.cross("ientos", "ientas"), "", "", SIGMA)).optimize()
    fem_ones = pynini.union(pynini.cross("un", "una"), pynini.cross("ún", "una"), pynini.cross("uno", "una"))
    end = pynini.union(EOS, pynini.accep('"'), DECIMAL_SEPARATOR)
    before_mil = (
        " " + pynini.union("mil", "milésimo")
        + pynini.closure(" " + hundreds, 0, 1)
        + pynini.closure(" " + one_to_one_hundred, 0, 1)
        + end
    )
    before_double_digits = pynini.closure(" " + one_to_one_hundred, 0, 1) + pynini.union(EOS, pynini.accep('"'))
    space_or_end = pynini.union(" ", EOS, pynini.accep('"'))
    end_or_quote = pynini.union(EOS, pynini.accep('"'))
    return {
        "cardinal_gender": (
            pynini.cdrewrite(fem_hundreds, "", before_mil, SIGMA).optimize(),
            pynini.cdrewrite(fem_hundreds, "", before_double_digits, SIGMA).optimize(),
            pynini.cdrewrite(fem_ones, "", end, SIGMA).optimize(),
        ),
        "number_gender": (
            pynini.cdrewrite(fem_hundreds, "", "", SIGMA).optimize(),
            pynini.cdrewrite(fem_ones, "", space_or_end, SIGMA).optimize(),
        ),
        "strip_apocope": (
            pynini.cdrewrite(pynini.cross("un", "uno") | pynini.cross("ún", "uno"), "", end_or_quote, SIGMA).optimize(),
        ),
        "add_apocope_fem": (
            pynini.cdrewrite(
                pynini.cross("una", "un") | pynini.cross("veintiuna", "veintiún"), "", end_or_quote, SIGMA,
            ).optimize(),
        ),
        "strip_accent": (pynini.cdrewrite(ACCENTS, "", "", SIGMA).optimize(),),
    }


def _apply(fst, name):
    for rule in _rules()[name]:
        fst = (fst @ rule).optimize()
    return fst


def shift_cardinal_gender(fst):
    """Feminine cardinal agreement below 1000 and on multiples of 1000."""
    return _apply(fst, "cardinal_gender")


def shift_number_gender(fst):
    """Feminine agreement on every hundreds word and every final one."""
    return _apply(fst, "number_gender")


def strip_cardinal_apocope(fst):
    """Restores a final "un"/"ún" to "uno"."""
    return _apply(fst, "strip_apocope")


def add_cardinal_apocope_fem(fst):
    """Drops a final feminine "una" to "un" before a stressed vowel."""
    return _apply(fst, "add_apocope_fem")


def strip_accent(fst):
    return _apply(fst, "strip_accent")


def rewrite(fst, rule, left="", right=""):
    """Composes one context-dependent rewrite onto a bounded graph."""
    return (fst @ pynini.cdrewrite(rule, left, right, SIGMA)).optimize()


def inputs(fst):
    """The unweighted, deterministic acceptor of a graph's source strings."""
    projected = pynini.project(fst, "input").optimize()
    return pynini.arcmap(projected, map_type="rmweight").optimize()


def restrict(fst, excluded):
    """Restricts ``fst`` to source strings outside the acceptor ``excluded``."""
    allowed = pynini.difference(inputs(fst), inputs(excluded)).optimize()
    return (allowed @ fst).optimize()


def roman_to_int(fst):
    """Upper- or lower-cased Roman numerals up to 3999 composed onto ``fst``."""

    def load(name, upper):
        labels = data_labels("roman/" + name + ".tsv")
        return pynini.string_map([(x.upper() if upper else x, y) for x, y in labels])

    zero = pynutil.add_weight(pynutil.insert("0"), 0.01)
    graphs = []
    for upper in (True, False):
        digit, ties = load("digit", upper), load("ties", upper)
        hundreds, thousands = load("hundreds", upper), load("thousands", upper)
        graphs.append(
            digit
            | ties + (digit | zero)
            | hundreds + (ties | zero) + (digit | zero)
            | thousands + (hundreds | zero) + (ties | zero) + (digit | zero)
        )
    return (pynini.union(*graphs).optimize() @ fst).optimize()
