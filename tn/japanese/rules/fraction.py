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

from pynini.lib.pynutil import delete, insert

from tn.japanese.rules.cardinal import Cardinal
from tn.processor import Processor


class Fraction(Processor):

    def __init__(self, cardinal=None, input_normalizer=None):
        super().__init__(name="fraction")
        self.input_normalizer = input_normalizer
        self.cardinal = cardinal or Cardinal(input_normalizer=input_normalizer)
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        rmspace = delete(" ").ques
        number = self.cardinal.number
        slash = self.apply_input_processor(delete("/"), self.input_normalizer)

        tagger = (self.tag_field("numerator", number) + rmspace + slash + rmspace + insert(" ") +
                  self.tag_field("denominator", number)).optimize()
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        denominator = delete('denominator: "') + self.cardinal.number + delete('" ')
        numerator = delete('numerator: "') + self.cardinal.number + delete('"')
        verbalizer = denominator + insert("分の") + numerator
        self.verbalizer = self.delete_tokens(verbalizer)
