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

from pynini import cross, string_file
from pynini.lib.pynutil import delete, insert

from tn.japanese.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Sport(Processor):

    def __init__(self, cardinal=None, input_normalizer=None):
        super().__init__(name="sport")
        self.input_normalizer = input_normalizer
        self.cardinal = cardinal or Cardinal(input_normalizer=input_normalizer)
        self.team = None
        self.score = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        country = string_file(get_abs_path("japanese/data/sport/country.tsv"))
        club = string_file(get_abs_path("japanese/data/sport/club.tsv"))
        rmsign = delete("/") | delete("-") | delete(":")

        number = self.cardinal.positive_integer
        self.team = self.apply_input_processor(country | club, self.input_normalizer)
        score = self.apply_input_processor(
            number + rmsign + insert("対") + number,
            self.input_normalizer,
        )
        only_score = self.apply_input_processor(
            number + cross(":", "対") + number,
            self.input_normalizer,
        )
        self.score = (score | only_score).optimize()
        tagger = (self.tag_field("team", self.team) + delete(" ").ques + insert(" ") + self.tag_field("score", score)
                  | self.tag_field("score", only_score))
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        team = delete('team: "') + self.team + delete('" ')
        score = delete('score: "') + self.score + delete('"')
        verbalizer = team.ques + score
        self.verbalizer = self.delete_tokens(verbalizer)
