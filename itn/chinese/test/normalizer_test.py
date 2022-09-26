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

from itertools import chain

from itn.chinese.inverse_normalizer import InverseNormalizer
from itn.chinese.test.utils import parse_test_case


class TestNormalizer:

    normalizer = InverseNormalizer(overwrite_cache=True)

    normalizer_cases = chain(
        parse_test_case('data/cardinal.txt'),
        parse_test_case('data/char.txt'),
        parse_test_case('data/date.txt'),
        parse_test_case('data/fraction.txt'),
        parse_test_case('data/math.txt'),
        parse_test_case('data/measure.txt'),
        parse_test_case('data/money.txt'),
        parse_test_case('data/time.txt'),
        parse_test_case('data/whitelist.txt'),
        parse_test_case('data/number.txt'),
        parse_test_case('data/normalizer.txt'))

    @pytest.mark.parametrize("spoken, written", normalizer_cases)
    def test_normalizer(self, spoken, written):
        assert self.normalizer.normalize(spoken) == written
