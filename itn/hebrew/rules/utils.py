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
from pynini.lib import byte, utf8
from pynini.lib.pynutil import delete

from itn.utils import get_abs_path

ALPHA_HE = pynini.union(*"אבגדהוזחטיכלמםנןסעפףצץקרשת").optimize()
SPACE = (byte.SPACE | "\u00a0").optimize()
NOT_SPACE = pynini.difference(utf8.VALID_UTF8_CHAR, SPACE).optimize()
VSIGMA = utf8.VALID_UTF8_CHAR.star.optimize()
DIGIT = byte.DIGIT
# Zero or more spaces between the words of one field, as in NeMo.
DELETE_SPACE = delete(SPACE).star
# Field values never start or end with a space: spaces around a field
# belong to the token, not to the field.
TRIMMED = (NOT_SPACE | NOT_SPACE + VSIGMA + NOT_SPACE).optimize()


def load_labels(rel_path):
    """Loads the rows of a Hebrew data table in file order."""

    rows = []
    with open(get_abs_path("hebrew/data/" + rel_path), encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\r\n")
            if line:
                rows.append(line.split("\t"))
    return rows


def string_file(rel_path):
    return pynini.string_map(load_labels(rel_path)).optimize()


def accep_file(rel_path):
    """Accepts the first column of a table."""

    return pynini.union(*[row[0] for row in load_labels(rel_path)]).optimize()


def inverse_file(rel_path):
    """Maps the second column of a ``written<TAB>spoken`` table to the first."""

    return pynini.string_map([(row[1], row[0]) for row in load_labels(rel_path)]).optimize()


PREFIX = accep_file("prefix.tsv")


def trim(graph):
    return (TRIMMED @ graph).optimize()


def first_field(processor, name, graph):
    """Verbalizes a field that is followed by another field."""

    return processor.verbalize_field(name, graph) + delete(" ")


def next_field(processor, name, graph):
    """Verbalizes a field that follows another field."""

    return processor.verbalize_field(name, graph, leading_space=True)
