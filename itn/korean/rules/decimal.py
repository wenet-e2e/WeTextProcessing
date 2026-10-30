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


class Decimal(Processor):
    def __init__(self, cardinal):
        super().__init__(name="decimal", ordertype="itn")
        number = cardinal.just_cardinals
        digit = string_file(get_abs_path("korean/data/numbers/digit.tsv"))
        zero = string_file(get_abs_path("korean/data/numbers/zero.tsv"))
        fractional = (digit | zero).plus
        regular = number + cross("점", ".") + fractional
        self.just_decimal = (number | regular).optimize()
        quantity = union("만", "억", "조", "경")
        self.negative = cardinal.negative
        sign = (self.tag_field("negative", self.negative) + insert(" ")).ques
        fields = self.tag_field("integer_part", number) + delete("점") + insert(" ")
        fields += self.tag_field("fractional_part", fractional)
        fields += (insert(" ") + self.tag_field("quantity", quantity)).ques
        self.tagger = self.add_tokens(sign + fields)
        sign_out = (self.verbalize_field("negative", self.negative) + delete(" ")).ques
        fields_out = self.verbalize_field("integer_part", number) + delete(" ") + insert(".")
        fields_out += self.verbalize_field("fractional_part", fractional)
        fields_out += (delete(" ") + self.verbalize_field("quantity", quantity)).ques
        self.verbalizer = self.delete_tokens(sign_out + fields_out)
