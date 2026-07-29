# Copyright (c) 2024 Logan Liu (2319277867@qq.com)
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

    def __init__(self, input_processor=None):
        super().__init__(name="time")
        self.input_processor = input_processor
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        h = string_file(get_abs_path("japanese/data/time/hour.tsv"))
        m = string_file(get_abs_path("japanese/data/time/minute.tsv"))
        s = string_file(get_abs_path("japanese/data/time/second.tsv"))

        # 一時三十分三秒 一時三十分 三十分三秒 一時 三十分 三秒
        self.hour = self.apply_input_processor(h, self.input_processor)
        self.minute = self.apply_input_processor(m, self.input_processor)
        self.second = self.apply_input_processor(s, self.input_processor)
        tagger = (((self.tag_field("hour", self.hour) + insert(" ")).ques + self.tag_field("minute", self.minute) +
                   (insert(" ") + self.tag_field("second", self.second)).ques)
                  | self.tag_field("hour", self.hour) + insert(" ")
                  | insert(" ") + self.tag_field("second", self.second))
        tagger = self.add_tokens(tagger)
        self.tagger = tagger

    def build_verbalizer(self):
        hour = (self.verbalize_field("hour", self.hour) + delete(" ").ques + insert("時"))
        minute = self.verbalize_field("minute", self.minute) + insert("分")
        second = (delete(" ").ques + self.verbalize_field("second", self.second) + insert("秒"))

        verbalizer = hour.ques + minute + second.ques | second | hour
        self.verbalizer = self.delete_tokens(verbalizer)
