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

import pytest

from processors.postprocessor import PostProcessor
from test.utils import parse_test_case


class TestPostProcessor:

    processor = PostProcessor(remove_puncts=True, tag_oov=True).processor
    processor_cases = parse_test_case('data/postprocessor.txt')

    @pytest.mark.parametrize("written, xxx", processor_cases)
    def test_processor(self, written, xxx):
        print((written @ self.processor).string())
        assert (written @ self.processor).string() == xxx
