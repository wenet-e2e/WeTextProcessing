# Copyright (c) 2024 Xingchen Song (sxc19@tsinghua.org.cn)
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

from tn.english.rules.electronic import Electronic
from tn.english.test.utils import parse_test_case


class TestElectronic:

    electronic = Electronic(deterministic=False)
    electronic_cases = parse_test_case("data/electronic.txt")

    @pytest.mark.parametrize("written, spoken", electronic_cases)
    def test_electronic(self, written, spoken):
        assert self.electronic.normalize(written) == spoken

    @pytest.mark.parametrize(
        "written, tagged, spoken",
        [
            (
                "www.abc.com",
                'electronic { protocol: "www." domain: "abc.com" }',
                "WWW dot abc dot com",
            ),
            (
                "http://www.abc.com",
                'electronic { protocol: "http://www." domain: "abc.com" }',
                "HTTP colon slash slash WWW dot abc dot com",
            ),
        ],
    )
    def test_protocol_structure_weight_survives_raw_fields(self, written, tagged, spoken):
        assert self.electronic.tag(written).strip() == tagged
        assert self.electronic.normalize(written) == spoken

    @pytest.mark.parametrize("written", ["cdf1@abc.edu", "http://www.abc.com"])
    def test_nbest_has_canonical_whitespace(self, written):
        outputs = self.electronic.normalize(written, nbest=32)

        assert outputs
        assert all(output == output.strip() for output in outputs)
        assert all("  " not in output for output in outputs)
