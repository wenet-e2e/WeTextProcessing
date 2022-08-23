# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
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

from processors.processor import Processor

from pynini import string_file
from pynini.lib.pynutil import delete, insert


class Time(Processor):

    def __init__(self):
        super().__init__(name='time')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        h = string_file('data/time/hour.tsv')
        m = string_file('data/time/min_sec.tsv')
        s = string_file('data/time/min_sec.tsv')
        noon = string_file('data/time/noon.tsv')

        tagger = (
            insert('hours: "') + h + insert('" ') + delete(':') +
            insert('minutes: "') + m + insert('"') +
            (delete(':') + insert(' seconds: "') + s + insert('"')).ques +
            delete(' ').ques + (insert(' noon: "') + noon + insert('"')).ques)
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        noon = delete('noon: "') + self.SIGMA + delete('" ')
        hours = delete('hours: "') + self.SIGMA + insert('点') + delete('"')
        minutes = delete('minutes: "') + self.SIGMA + insert('分') + delete('"')
        seconds = delete('seconds: "') + self.SIGMA + insert('秒') + delete('"')
        verbalizer = (noon.ques + hours + delete(' ') + minutes +
                      (delete(' ') + seconds).ques)
        self.verbalizer = self.delete_tokens(verbalizer)
