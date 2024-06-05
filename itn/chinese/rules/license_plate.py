# Copyright (c) 2023 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
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
from pynini.lib.pynutil import insert


class LicensePlate(Processor):

    def __init__(self):
        super().__init__(name='licenseplate')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        digit = string_file(
            get_abs_path('../itn/chinese/data/number/digit.tsv'))  # 1 ~ 9
        zero = string_file(
            get_abs_path('../itn/chinese/data/number/zero.tsv'))  # 0
        digits = zero | digit
        province = string_file(
            get_abs_path(
                '../itn/chinese/data/license_plate/province.tsv'))  # 皖
        license_plate = province + self.ALPHA + (self.ALPHA | digits)**5
        license_plate |= province + self.ALPHA + (self.ALPHA | digits)**6
        tagger = insert('value: "') + license_plate + insert('"')
        self.tagger = self.add_tokens(tagger)
