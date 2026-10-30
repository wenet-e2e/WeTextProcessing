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

from itn.utils import get_abs_path
from tn.processor import Processor


class Telephone(Processor):
    def __init__(self):
        super().__init__(name="telephone", ordertype="itn")
        digit = string_file(get_abs_path("korean/data/numbers/digit.tsv"))
        digit |= cross(union("영", "공"), "0")
        separator = cross(union(" ", "에"), "-").ques
        number = (digit ** 2 | digit ** 3) + separator + (digit ** 3 | digit ** 4) + separator + digit ** 4
        self.tagger = self.add_tokens(self.tag_field("number_part", number))
        self.verbalizer = self.delete_tokens(self.verbalize_field("number_part", number))
