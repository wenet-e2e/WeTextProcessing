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


class Measure(Processor):
    def __init__(self, cardinal):
        super().__init__(
            name="measure", ordertype="itn",
            token_orders={"measure": ["negative", "numerator", "denominator", "value", "units"]},
        )
        number = cardinal.just_cardinals
        digit = string_file(get_abs_path("korean/data/numbers/digit.tsv"))
        fractional = (digit | cross(union("영", "공"), "0")).plus
        space = delete(" ").star
        decimal = number + space + cross("점", ".") + space + fractional
        value = (number | decimal).optimize()
        negative = cross(union("마이너스", "영하"), "-")
        unit = string_file(get_abs_path("korean/data/measure_units.tsv"))
        units = (unit | unit + space + cross(union("퍼", "당"), "/") + space + unit).optimize()
        root = union("√") | cross("루트", "√")
        operand = (number | root + number).optimize()
        sign = (self.tag_field("negative", negative) + space + insert(" ")).ques
        regular = sign + self.tag_field("value", value)
        fraction = self.tag_field("denominator", operand) + space + delete("분의") + space
        fraction += insert(" ") + self.tag_field("numerator", operand)
        fields = space + (regular | fraction) + space + insert(" ") + self.tag_field("units", units)
        self.tagger = self.add_tokens(fields)
        sign_out = (self.verbalize_field("negative", negative) + delete(" ")).ques
        regular_out = sign_out + self.verbalize_field("value", value)
        fraction_out = self.verbalize_field("numerator", operand) + delete(" ") + insert("/")
        fraction_out += self.verbalize_field("denominator", operand)
        fields_out = (regular_out | fraction_out) + delete(" ") + insert(" ")
        fields_out += self.verbalize_field("units", units)
        self.verbalizer = self.delete_tokens(fields_out)
