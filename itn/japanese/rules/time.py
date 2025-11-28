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
from tn.utils import get_abs_path


class Time(Processor):

    def __init__(self):
        super().__init__(name="time")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        h = string_file(get_abs_path("../itn/japanese/data/time/hour.tsv"))
        m = string_file(get_abs_path("../itn/japanese/data/time/minute.tsv"))
        s = string_file(get_abs_path("../itn/japanese/data/time/second.tsv"))

        # 一時三十分三秒 一時三十分 三十分三秒 一時 三十分 三秒
        tagger = (
            (
                (insert('hour: "') + h + insert('" ')).ques
                + (insert('minute: "') + m + insert('"'))
                + (insert(' second: "') + s + insert('"')).ques
            )
            | insert('hour: "') + h + insert('" ')
            | insert(' second: "') + s + insert('"')
        )
        tagger = self.add_tokens(tagger)
        self.tagger = tagger

    def build_verbalizer(self):
        hour = delete('hour: "') + self.SIGMA + delete('"') + delete(" ").ques + insert("時")
        minute = delete('minute: "') + self.SIGMA + delete('"') + insert("分")
        second = delete(" ").ques + delete('second: "') + self.SIGMA + delete('"') + insert("秒")

        verbalizer = hour.ques + minute + second.ques | second | hour
        self.verbalizer = self.delete_tokens(verbalizer)
