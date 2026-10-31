# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from itn.utils import get_abs_path

# Devanagari digits, the written form produced by Marathi ITN.
DIGITS = "०१२३४५६७८९"
DIGIT = pynini.union(*DIGITS).optimize()
NON_ZERO = pynini.union(*DIGITS[1:]).optimize()


def to_devanagari(number, width=1):
    """Formats an integer with Devanagari digits, zero-padded to ``width``."""

    return "".join(DIGITS[int(char)] for char in str(number).zfill(width))


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
    """Maps 1-99 to their Marathi spoken words."""

    words = {}
    for name in ("digit", "tens"):
        for written, spoken in load_pairs("numbers/{}.tsv".format(name)):
            words[int(written.translate(str.maketrans(DIGITS, "0123456789")))] = spoken
    return words


def spoken_map(pairs):
    """Builds a ``spoken -> written`` transducer from ``(written, spoken)`` pairs."""

    return pynini.union(*[pynini.cross(pynini.escape(spoken), pynini.escape(written))
                          for written, spoken in pairs]).optimize()
