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

from pynini import accep, cross, string_file
from pynini.lib.pynutil import delete, insert


class Measure(Processor):

    def __init__(self):
        super().__init__(name='measure')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        units_en = string_file(
            get_abs_path('japanese/data/measure/units_en.tsv'))
        units_ja = string_file(
            get_abs_path('japanese/data/measure/units_ja.tsv'))

        # taking '-' '~' as 'から' if the follwing word in units
        units = (units_en | units_ja)
        rmspace = delete(' ').ques
        to = cross('-', 'から') | cross('~', 'から') | accep('から')

        number = Cardinal().number
        # 1-11月，1月-11月
        prefix = number + (rmspace + units).ques + to
        measure = prefix.ques + number + rmspace + units
        measure |= (measure + rmspace + delete('/') + insert('毎') + rmspace +
                    units)
        tagger = insert('value: "') + measure + insert('"')

        # 10km/h, 2m/s
        self.tagger = self.add_tokens(tagger)
