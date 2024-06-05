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
from tn.utils import get_abs_path

from pynini import string_file


class PreProcessor(Processor):

    def __init__(self, traditional_to_simple=True):
        super().__init__(name='preprocessor')
        traditional2simple = string_file(
            get_abs_path('chinese/data/char/traditional_to_simple.tsv'))

        processor = self.build_rule('')
        if traditional_to_simple:
            processor @= self.build_rule(traditional2simple)

        self.processor = processor.optimize()
