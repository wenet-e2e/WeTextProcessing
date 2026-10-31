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
from pynini.lib import byte, pynutil

from itn.utils import get_abs_path

DIGIT = byte.DIGIT
NONZERO = pynini.difference(DIGIT, "0").optimize()
# Optional word boundary inside spoken numbers ("tjugo en", "tjugoen").
SPACE = pynutil.delete(" ").ques
SCALES = ("triljard", "triljon", "biljard", "biljon", "miljard", "miljon")


def rows(name):
    """Reads TSV rows verbatim."""

    with open(get_abs_path("swedish/data/" + name + ".tsv"), encoding="utf-8") as stream:
        return [line.rstrip("\r\n").split("\t") for line in stream if line.rstrip("\r\n")]


def table(name):
    """Spoken-to-written map for a ``written<TAB>spoken`` resource."""

    return pynini.string_map([(row[-1], row[0]) for row in rows(name)]).optimize()


def spoken_table(name):
    """Map for a ``spoken<TAB>written`` resource."""

    return pynini.string_map([(row[0], row[-1]) for row in rows(name)]).optimize()


def strip_zeros():
    """Removes leading zeros from a fixed-width digit string, keeping one digit."""

    return (pynutil.delete(pynini.accep("0").star) + NONZERO + DIGIT.star | pynutil.delete(
        pynini.accep("0").star) + "0").optimize()


def old_spelling(word):
    return word.replace("lj", "lli")
