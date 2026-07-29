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

    def __init__(self, input_normalizer=None, range_tagger=None):
        super().__init__(name="time")
        self.input_normalizer = input_normalizer
        self.range_tagger = range_tagger
        self.hour = None
        self.minute = None
        self.second = None
        self.noon = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        h = string_file(get_abs_path("japanese/data/time/hour.tsv"))
        m = string_file(get_abs_path("japanese/data/time/minute.tsv"))
        s = string_file(get_abs_path("japanese/data/time/second.tsv"))
        noon = string_file(get_abs_path("japanese/data/time/noon.tsv"))

        colon = self.apply_input_processor(delete(":") | delete("："), self.input_normalizer)
        self.hour = self.apply_input_processor(h, self.input_normalizer)
        self.minute = self.apply_input_processor(m, self.input_normalizer)
        self.second = self.apply_input_processor(s, self.input_normalizer)
        self.noon = self.apply_input_processor(noon, self.input_normalizer)

        h_noon = self.tag_field("hour", self.hour) + insert(" ") + self.tag_field("noon", self.noon)
        tagger = (self.tag_field("hour", self.hour) + insert(" ") + colon + self.tag_field("minute", self.minute) +
                  (colon + insert(" ") + self.tag_field("second", self.second)).ques +
                  (delete(" ").ques + insert(" ") + self.tag_field("noon", self.noon)).ques
                  | h_noon)
        tagger = self.add_tokens(tagger)

        if self.range_tagger is None:
            self.tagger = tagger
        else:
            self.tagger = tagger + (self.range_tagger + tagger).ques

    def build_verbalizer(self):
        noon = delete('noon: "') + self.noon + delete('" ')
        hour = delete('hour: "') + self.hour + delete('"')
        minute = delete(' minute: "') + self.minute + delete('"')
        second = delete(' second: "') + self.second + delete('"')
        verbalizer = noon.ques + hour + minute + second.ques | noon + hour
        self.verbalizer = self.delete_tokens(verbalizer)
