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

from tn.utils import get_abs_path

HI_DIGIT = pynini.union(*"०१२३४५६७८९").optimize()
HI_NON_ZERO = pynini.union(*"१२३४५६७८९").optimize()
ALL_DIGIT = pynini.union(HI_DIGIT, *"0123456789").optimize()
ALL_ZERO = pynini.union("०", "0").optimize()
ALL_NON_ZERO = pynini.union(HI_NON_ZERO, *"123456789").optimize()

HI_DEDH = "डेढ़"  # 1.5
HI_DHAI = "ढाई"  # 2.5
HI_SAVVA = "सवा"  # x.25
HI_SADHE = "साढ़े"  # x.5
HI_PAUNE = "पौने"  # x.75
MINUS = "ऋणात्मक "


def hindi_path(rel_path):
    return get_abs_path("hindi/data/" + rel_path)


def read_lines(rel_path):
    with open(hindi_path(rel_path), encoding="utf-8") as stream:
        return [line.rstrip("\n") for line in stream if line.strip()]


def squeeze_spaces():
    """Drops edge spaces and collapses inner runs, like NeMo's final ``SPACE_DUP``.

    NeMo's semantic graphs insert a space after every unit they read; applying this
    small acceptor-shaped transducer to one field's output keeps the spoken text clean.
    """

    char = utf8.VALID_UTF8_CHAR - " "
    words = char.plus + (pynini.cross(pynini.accep(" ").plus, " ") + char.plus).star
    return (delete(" ").star + words.ques + delete(" ").star).optimize()
