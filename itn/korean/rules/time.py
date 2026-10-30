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

from pynini import cross, string_file, union
from pynini.lib.pynutil import delete, insert

from itn.utils import get_abs_path
from tn.processor import Processor


class Time(Processor):
    def __init__(self):
        super().__init__(name="time", ordertype="itn")
        digit = string_file(get_abs_path("korean/data/numbers/digit.tsv"))
        zero = string_file(get_abs_path("korean/data/numbers/zero.tsv"))
        tens_prefix = string_file(get_abs_path("korean/data/time/ten_prefix.tsv"))
        tens = tens_prefix + (cross("십", "0") | delete("십") + digit)
        number = (digit | zero | cross("십", "10") | cross("십", "1") + digit | tens).optimize()
        hours = string_file(get_abs_path("korean/data/time/time_hours.tsv"))
        half = cross("반", "30")
        minute = number | half
        prefix = union("오전", "오후", "새벽", "아침")
        suffix = union("전", "후")
        space = self.DELETE_SPACE
        h = self.tag_field("hours", hours) + delete(" ").ques + delete("시")
        m = self.tag_field("minutes", number) + delete(" ").ques + delete("분")
        m |= self.tag_field("minutes", half)
        s = self.tag_field("seconds", number) + delete(" ").ques + delete("초")
        boundary = space + insert(" ")
        hm = h + (boundary + m).ques
        regular = hm + (boundary + s).ques
        regular |= (self.tag_field("minutes", number) + delete(" ").ques + delete("분")) + (boundary + s).ques
        regular |= s
        prefix_tag = (self.tag_field("prefix", prefix) + boundary).ques
        suffix_tag = (boundary + self.tag_field("suffix", suffix)).ques
        self.tagger = self.add_tokens(prefix_tag + regular + suffix_tag)
        pad = insert("0") + self.DIGIT | self.DIGIT ** 2
        h_out = self.verbalize_field("hours", hours)
        m_out = self.verbalize_field("minutes", minute @ pad)
        s_out = self.verbalize_field("seconds", number @ pad)
        colon = delete(" ") + insert(":")
        regular_out = h_out + insert(":00")
        regular_out |= h_out + colon + m_out
        regular_out |= h_out + delete(" ") + insert(":00:") + s_out
        regular_out |= h_out + colon + m_out + colon + s_out
        regular_out |= m_out | m_out + colon + s_out | s_out
        prefix_out = (self.verbalize_field("prefix", prefix) + delete(" ") + insert(" ")).ques
        suffix_out = (delete(" ") + insert(" ") + self.verbalize_field("suffix", suffix)).ques
        self.verbalizer = self.delete_tokens(prefix_out + regular_out + suffix_out)
