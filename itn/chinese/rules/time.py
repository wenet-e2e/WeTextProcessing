# Copyright (c) 2022 Xingchen Song (sxc19@tsinghua.org.cn)
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

from pynini import string_file
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor
from itn.utils import get_abs_path


class Time(Processor):

    def __init__(self):
        super().__init__(name="time")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        h = string_file(get_abs_path("chinese/data/time/hour.tsv"))
        m = string_file(get_abs_path("chinese/data/time/minute.tsv"))
        m_zero = string_file(get_abs_path("chinese/data/time/minute_zero.tsv"))
        s = string_file(get_abs_path("chinese/data/time/second.tsv"))
        noon = string_file(get_abs_path("chinese/data/time/noon.tsv"))

        self.hour = h
        self.minute_tagger = m + delete("分").ques | m_zero + delete("分")
        self.minute = self.minute_tagger | m_zero
        self.minute_zero_no_fen = m_zero
        self.second = s
        self.noon = noon
        hour = self.tag_field("hour", self.hour)
        minute = insert(" ") + self.tag_field("minute", self.minute_tagger)
        minute_zero_no_fen = insert(" ") + self.tag_field("minute", self.minute_zero_no_fen)
        second = (insert(" ") + self.tag_field("second", self.second)).ques

        tagger = ((self.tag_field("noon", self.noon) + insert(" ")).ques + hour + minute + second)
        # "X点零Y" without "分" requires noon prefix to disambiguate from decimal
        tagger |= (self.tag_field("noon", self.noon) + insert(" ") + hour + minute_zero_no_fen + second)
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        addcolon = insert(":")
        hour = self.verbalize_field("hour", self.hour)
        minute = self.verbalize_field("minute", self.minute, leading_space=True)
        second = self.verbalize_field("second", self.second, leading_space=True)
        noon = self.verbalize_field("noon", self.noon, leading_space=True)
        verbalizer = hour + addcolon + minute + (addcolon + second).ques + noon.ques
        self.verbalizer = self.delete_tokens(verbalizer)
