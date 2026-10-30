# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from pynini import cross, difference, string_file, union
from pynini.lib.pynutil import delete, insert

from itn.korean.rules.utils import josa_graph
from itn.utils import get_abs_path
from tn.processor import Processor


class Ordinal(Processor):
    def __init__(self, cardinal):
        super().__init__(name="ordinal", ordertype="itn")
        digit = string_file(get_abs_path("korean/data/ordinals/digit.tsv"))
        cardinal_digit = string_file(get_abs_path("korean/data/numbers/digit.tsv"))
        tens = cross("열", "10") | cross("열", "1") + digit
        twenties = cross("스무", "20") | cross("스물", "2") + digit
        thirties = cross("서른", "30") | cross("서른", "3") + digit
        two_to_nine = difference(self.input_projection(digit), "한") @ digit
        native = (cross("첫", "1") | two_to_nine | tens | twenties | thirties).optimize()
        below_40 = cardinal_digit
        for word in ("십", "이십", "삼십"):
            below_40 |= union(word) + self.input_projection(cardinal_digit).ques
        over_40 = difference(self.input_projection(cardinal.just_cardinals), self.input_projection(below_40))
        over_40 = (over_40 @ cardinal.just_cardinals).optimize()
        ordinal = (native | over_40) + self.DELETE_SPACE + union("번째")
        counting = (digit | tens | twenties | thirties | over_40).optimize()
        counter = string_file(get_abs_path("korean/data/ordinals/counter_suffix.tsv"))
        counted = self.tag_field("integer", counting) + self.DELETE_SPACE + insert(" ")
        counted += self.tag_field("counter", counter)
        counted += (self.DELETE_SPACE + insert(" ") + self.tag_field("suffix", josa_graph())).ques
        self.tagger = self.add_tokens(self.tag_field("integer", ordinal) | counted)
        counted_out = self.verbalize_field("integer", counting) + delete(" ")
        counted_out += self.verbalize_field("counter", counter)
        counted_out += (delete(" ") + self.verbalize_field("suffix", josa_graph())).ques
        self.verbalizer = self.delete_tokens(self.verbalize_field("integer", ordinal) | counted_out)
