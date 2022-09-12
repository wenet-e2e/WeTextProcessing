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

from tn.processor import Processor

from pynini import string_file
from pynini.lib.pynutil import delete, insert


class Time(Processor):

    def __init__(self):
        super().__init__(name='time')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        h = string_file('itn/chinese/data/time/hour.tsv')
        m = string_file('itn/chinese/data/time/minute.tsv')
        s = string_file('itn/chinese/data/time/second.tsv')
        noon = string_file('itn/chinese/data/time/noon.tsv')

        tagger = (
            (insert('noon: "') + noon + insert('" ')).ques +
            insert('hour: "') + h + insert('"') +
            insert(' minute: "') + m + delete('分').ques + insert('"') +
            (insert(' second: "') + s + insert('"')).ques)
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        addcolon = insert(':')
        hour = delete('hour: "') + self.SIGMA + delete('"')
        minute = delete(' minute: "') + self.SIGMA + delete('"')
        second = delete(' second: "') + self.SIGMA + delete('"')
        noon = delete(' noon: "') + self.SIGMA + delete('"')
        verbalizer = (hour + addcolon + minute +
                      (addcolon + second).ques + noon.ques)
        self.verbalizer = self.delete_tokens(verbalizer)
