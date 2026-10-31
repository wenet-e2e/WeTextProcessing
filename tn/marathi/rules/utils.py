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
from pynini.lib import utf8

from tn.utils import get_abs_path

# Devanagari digits are the canonical written form; ASCII digits are accepted
# as input and mapped onto them inside the verbalizer graphs.
DEVANAGARI_DIGITS = "०१२३४५६७८९"
MR_DIGIT = pynini.union(*DEVANAGARI_DIGITS).optimize()
MR_NON_ZERO = pynini.union(*DEVANAGARI_DIGITS[1:]).optimize()
ALL_DIGIT = pynini.union(MR_DIGIT, *"0123456789").optimize()
TO_DEVANAGARI = pynini.union(MR_DIGIT, *[pynini.cross(str(index), char) for index, char in enumerate(DEVANAGARI_DIGITS)])
TO_DEVANAGARI = TO_DEVANAGARI.optimize()


def to_devanagari(number, width=1):
    """Formats an integer with Devanagari digits, zero-padded to ``width``."""

    return "".join(DEVANAGARI_DIGITS[int(char)] for char in str(number).zfill(width))


def canonical(graph):
    """Lets a graph over Devanagari digits also read ASCII (or mixed) digits.

    Non-digit characters pass through unchanged.
    """

    char = pynini.union(TO_DEVANAGARI, pynini.difference(utf8.VALID_UTF8_CHAR, ALL_DIGIT))
    return (char.optimize().star @ graph).optimize()


def load_pairs(rel_path):
    """Loads ``written<TAB>spoken`` rows of a Marathi data table in file order.

    A single-column row is read the same way on both sides.
    """

    pairs = []
    with open(get_abs_path("marathi/data/" + rel_path), encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\r\n")
            if line:
                columns = line.split("\t")
                pairs.append((columns[0], columns[-1]))
    return pairs


def number_words():
    """Maps 0-99 to their Marathi spoken words."""

    words = {}
    for name in ("zero", "digit", "tens"):
        for written, spoken in load_pairs("numbers/{}.tsv".format(name)):
            words[int(written.translate(str.maketrans(DEVANAGARI_DIGITS, "0123456789")))] = spoken
    return words


def written_map(pairs):
    """Builds a ``written -> spoken`` transducer from ``(written, spoken)`` pairs."""

    return pynini.union(*[pynini.cross(pynini.escape(written), pynini.escape(spoken))
                          for written, spoken in pairs]).optimize()
