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

"""Shared helpers of the Spanish ITN rules.

Spoken-word graphs here map *bounded* sets of words to written forms. The one
rewrite rule (initial-capital folding for cased input) is composed onto those
bounded graphs, never onto another rule over the full alphabet.
"""

from functools import lru_cache

import pynini
from pynini.lib import byte, pynutil, utf8

from itn.utils import get_abs_path, load_labels

SIGMA = utf8.VALID_UTF8_CHAR.star
CHAR = utf8.VALID_UTF8_CHAR
DIGIT = byte.DIGIT
ALPHA = byte.ALPHA
WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", "\u00a0").optimize()
DELETE_SPACE = pynutil.delete(WHITE_SPACE).star
DELETE_EXTRA_SPACE = pynini.cross(pynini.closure(WHITE_SPACE, 1), " ")
MINUS = pynini.union("menos", "Menos", "MENOS")
PLUS = pynini.union("más", "Más", "MÁS")

_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÑÜ"
_LOWER = "abcdefghijklmnopqrstuvwxyzáéíóúñü"
TO_LOWER = pynini.string_map(list(zip(_UPPER, _LOWER))).optimize()


def data_path(name):
    return get_abs_path("spanish/data/" + name)


def data_file(name):
    return pynini.string_file(data_path(name))


def data_labels(name):
    return load_labels(data_path(name))


@lru_cache(maxsize=None)
def _fold_initials():
    """Optionally lower-cases the first letter of every word."""
    return pynini.cdrewrite(TO_LOWER, pynini.union("[BOS]", " "), "", SIGMA, mode="opt").optimize()


def cased(graph, enabled=True):
    """Also accepts ``graph`` inputs whose words start with a capital letter."""
    if not enabled:
        return graph
    folded = (_fold_initials() @ graph).optimize()
    return (graph | pynutil.add_weight(folded, 0.001)).optimize()


def case_insensitive(graph):
    """Accepts any letter of ``graph`` inputs in upper case."""
    return (pynini.closure(TO_LOWER | CHAR) @ graph).optimize()


def inputs(fst):
    """The unweighted, deterministic acceptor of a graph's source strings."""
    projected = pynini.project(fst, "input").optimize()
    return pynini.arcmap(projected, map_type="rmweight").optimize()


def restrict(fst, excluded):
    """Restricts ``fst`` to source strings outside the acceptor ``excluded``."""
    allowed = pynini.difference(inputs(fst), inputs(excluded)).optimize()
    return (allowed @ fst).optimize()


def int_to_roman(fst):
    """Composes lower-cased Roman numerals up to 3999 onto digit outputs."""

    def load(name):
        return pynini.string_map([(y, x.lower()) for x, y in data_labels("roman/" + name + ".tsv")])

    zero = pynutil.add_weight(pynutil.delete("0"), 0.01)
    digit, ties, hundreds, thousands = load("digit"), load("ties"), load("hundreds"), load("thousands")
    graph = (
        digit
        | ties + (digit | zero)
        | hundreds + (ties | zero) + (digit | zero)
        | thousands + (hundreds | zero) + (ties | zero) + (digit | zero)
    )
    return (fst @ graph.optimize()).optimize()
