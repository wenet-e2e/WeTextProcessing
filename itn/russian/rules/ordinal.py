# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from itn.russian.rules.cardinal import marked_numbers
from itn.russian.rules.utils import DIGIT, number_names
from tn.processor import Processor


class Ordinal(Processor):
    """Spoken ordinals of two or more digits, e.g. "тысяча девятьсот девяносто шестого" -> "1996".

    Written endings ("1950-е", "8-го") and one-digit ordinals are lower-ranked alternatives.
    """

    def __init__(self):
        super().__init__(name="ordinal", ordertype="itn")
        self.numbers = number_names().ordinal.copy().invert().optimize()
        multiple = (self.numbers @ pynini.closure(DIGIT, 2)).optimize()
        single = (self.numbers @ DIGIT).optimize()
        self.graph = (
            multiple | single | pynutil.add_weight(marked_numbers(self.numbers, "numbers/ordinal_endings.tsv"), 1)
        ).optimize()
        self.tagger = self.add_tokens(
            self.tag_field("integer", multiple) | pynutil.add_weight(self.tag_field("integer", single), 100))
        self.verbalizer = self.delete_tokens(self.verbalize_field("integer", self.graph))
