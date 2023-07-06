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

from tn.processor import Processor

from pynini import string_file
from pynini.lib.pynutil import delete


class PreProcessor(Processor):

    def __init__(self, remove_interjections=True, full_to_half=True,
                 traditional_to_simple=True):
        super().__init__(name='preprocessor')
        blacklist = string_file('tn/chinese/data/default/blacklist.tsv')
        traditional2simple = string_file(
            'tn/chinese/data/char/traditional_to_simple.tsv')
        full2half = string_file(
            'tn/chinese/data/char/fullwidth_to_halfwidth.tsv')

        processor = self.build_rule('')
        if traditional_to_simple:
            processor @= self.build_rule(traditional2simple)

        if full_to_half:
            processor @= self.build_rule(full2half)

        if remove_interjections:
            processor @= self.build_rule(delete(blacklist))

        self.processor = processor.optimize()
