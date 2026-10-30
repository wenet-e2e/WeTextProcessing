# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.lib.pynutil import delete

from itn.utils import get_abs_path

CHAR = utf8.VALID_UTF8_CHAR
SIGMA = CHAR.star
HI_DIGIT = pynini.union(*"०१२३४५६७८९").optimize()
WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", "\u00a0").optimize()
NOT_SPACE = pynini.difference(CHAR, WHITE_SPACE).optimize()
# Optional whitespace between spoken words (deleted).
DELETE_SPACE = delete(WHITE_SPACE.star)
# Mandatory whitespace between spoken words (deleted).
DELETE_EXTRA_SPACE = delete(WHITE_SPACE.plus)
MINUS = pynini.union("ऋणात्मक", "नकारात्मक").optimize()
# "डेढ़" (one and a half) appears both with a combining nukta and with the precomposed letter.
DEDH = pynini.union("\u0921\u0947\u0922\u093c", "\u0921\u0947\u095d").optimize()


def hindi_path(rel_path):
    return get_abs_path("hindi/data/" + rel_path)


def string_file(rel_path):
    """Loads ``written<TAB>spoken`` (or single-column) data as written -> spoken."""

    return pynini.string_file(hindi_path(rel_path))


def spoken_to_written(rel_path):
    return string_file(rel_path).invert().optimize()


def devanagari_numbers(start, stop):
    digits = str.maketrans("0123456789", "०१२३४५६७८९")
    return pynini.union(*[str(i).translate(digits) for i in range(start, stop)]).optimize()
