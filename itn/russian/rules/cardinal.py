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

from itn.russian.rules.utils import DIGIT, RU_ALPHA_OR_SPACE, data_path, number_names
from itn.utils import load_labels
from tn.processor import Processor


def marked_numbers(numbers, endings_file):
    """Digits plus a written case ending agreeing with the spoken word ("шестью" -> "6-ю")."""
    marked = []
    for (row,) in load_labels(data_path(endings_file)):
        spoken, written = row.split("-", 1)
        marked.append(((pynini.closure(RU_ALPHA_OR_SPACE) + spoken) @ numbers) + pynutil.insert("-" + written))
    return pynini.union(*marked).optimize()


class Cardinal(Processor):
    """Spoken integers of two or more digits, e.g. "двести пятьдесят четыре" -> "254".

    One-digit numbers stay words in the best path ("девять"); their digits and
    written case endings ("6-ю") remain lower-ranked alternatives.
    """

    def __init__(self):
        super().__init__(name="cardinal", ordertype="itn")
        self.numbers = number_names().cardinal.copy().invert().optimize()
        self.nominative = number_names().nominative.copy().invert().optimize()
        multiple = (self.numbers @ pynini.closure(DIGIT, 2)).optimize()
        single = (self.numbers @ DIGIT).optimize()
        self.graph = (
            multiple | single | pynutil.add_weight(marked_numbers(self.numbers, "numbers/cardinal_endings.tsv"), 1)
        ).optimize()
        self.sign = pynini.cross("минус", "-")
        self.sign_tagger = self.tag_field("negative", self.sign) + pynutil.delete(" ") + pynutil.insert(" ")
        self.sign_verbalizer = self.verbalize_field("negative", self.sign) + pynutil.delete(" ")

        self.tagger = self.add_tokens(
            self.sign_tagger.ques + self.tag_field("integer", multiple)
            | pynutil.add_weight(self.tag_field("integer", single), 100)
        )
        self.verbalizer = self.delete_tokens(self.sign_verbalizer.ques + self.verbalize_field("integer", self.graph))
