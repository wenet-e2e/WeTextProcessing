# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026 Zhendong Peng (pzd17@tsinghua.org.cn)
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

from pynini import cross, string_file, union
from pynini.lib.pynutil import delete, insert

from itn.utils import get_abs_path
from tn.processor import Processor


class Cardinal(Processor):
    """Sino-Korean integers in four-digit magnitude groups, not digit strings."""

    def __init__(self):
        super().__init__(name="cardinal", ordertype="itn")
        digit = string_file(get_abs_path("korean/data/numbers/digit.tsv"))
        zero = string_file(get_abs_path("korean/data/numbers/zero.tsv"))
        group = digit | insert("0")
        for unit in ("십", "백", "천"):
            group = ((digit + delete(unit)) | cross(unit, "1") | insert("0")) + group
        group = group.optimize()
        number = group
        for unit in ("만", "억", "조", "경"):
            prefix = (group + delete(unit)) | (insert("000") + cross(unit, "1")) | insert("0000")
            number = (prefix + number).optimize()
        strip_zeros = delete(union("0").star) + (self.DIGIT - "0") + self.DIGIT.star
        self.just_cardinals = ((number @ strip_zeros) | zero).optimize()
        self.negative = cross("마이너스", "-") | union("-")
        sign = (self.tag_field("negative", self.negative) + self.DELETE_SPACE + insert(" ")).ques
        self.tagger = self.add_tokens(sign + self.tag_field("integer", self.just_cardinals))
        sign_out = (self.verbalize_field("negative", self.negative) + delete(" ")).ques
        self.verbalizer = self.delete_tokens(sign_out + self.verbalize_field("integer", self.just_cardinals))
