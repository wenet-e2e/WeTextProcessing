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

from tn.japanese.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path

from pynini import string_file, cross
from pynini.lib.pynutil import delete, insert


class Sport(Processor):

    def __init__(self):
        super().__init__(name='sport')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        country = string_file(get_abs_path('japanese/data/sport/country.tsv'))
        club = string_file(get_abs_path('japanese/data/sport/club.tsv'))
        rmsign = delete('/') | delete('-') | delete(':')
        rmspace = delete(' ').ques

        number = Cardinal().positive_integer
        score = rmspace + number + rmsign + insert('対') + number + rmspace
        only_score = rmspace + number + cross(':', '対') + number + rmspace
        tagger = ((insert('team: "') + (country | club) +
                   insert('" score: "') + score + insert('"'))
                  | (insert('score: "') + only_score + insert('"')))
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        super().build_verbalizer()
        team = delete('team: "') + self.SIGMA + delete('" ')
        score = delete('score: "') + self.SIGMA + delete('"')
        verbalizer = team.ques + score
        self.verbalizer = self.delete_tokens(verbalizer)
