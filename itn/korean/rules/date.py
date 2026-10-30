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

from pynini import string_file
from pynini.lib.pynutil import delete, insert

from itn.utils import get_abs_path
from tn.processor import Processor


class Date(Processor):
    def __init__(self, cardinal):
        super().__init__(name="date", ordertype="itn")
        number = cardinal.just_cardinals
        month = string_file(get_abs_path("korean/data/months.tsv"))
        year_tag = self.tag_field("year", number) + delete("년")
        month_tag = self.tag_field("month", month) + delete("월")
        day_tag = self.tag_field("day", number) + delete("일")
        boundary = delete(" ").ques + insert(" ")
        year_month = year_tag + boundary + month_tag
        fields = year_tag | month_tag | year_month
        fields |= (month_tag | year_month) + boundary + day_tag
        self.tagger = self.add_tokens(fields)
        year_out = self.verbalize_field("year", number) + insert("년")
        month_out = self.verbalize_field("month", month) + insert("월")
        day_out = self.verbalize_field("day", number) + insert("일")
        year_month_out = year_out + delete(" ") + month_out
        fields_out = year_out | month_out | year_month_out
        fields_out |= (month_out | year_month_out) + delete(" ") + day_out
        self.verbalizer = self.delete_tokens(fields_out)
