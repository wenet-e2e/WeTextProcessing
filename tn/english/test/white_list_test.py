# Copyright (c) 2024 Di Wu (1176705630@qq.com)
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

import pytest

from tn.english.rules.whitelist import Whitelist 
from tn.english.test.utils import parse_test_case


class TestWord:

    word = Whitelist(asr=True, tts=True, symbol=True)
    word_cases = parse_test_case('data/white_list.txt')

    @pytest.mark.parametrize("written, spoken", word_cases)
    def test_char(self, written, spoken):
        print(self.word.normalize(written))
        assert self.word.normalize(written) == spoken
