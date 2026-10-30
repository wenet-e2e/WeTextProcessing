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


class Money(Processor):
    def __init__(self, decimal):
        super().__init__(name="money", ordertype="itn")
        value = decimal.just_decimal
        currency = string_file(get_abs_path("korean/data/currency.tsv"))
        fields = self.tag_field("value", value) + delete(" ").ques + insert(" ")
        fields += self.tag_field("currency", currency)
        self.tagger = self.add_tokens(fields)
        # The pipeline owns the currency-before-value field permutation.
        fields_out = self.verbalize_field("currency", currency) + delete(" ")
        fields_out += self.verbalize_field("value", value)
        self.verbalizer = self.delete_tokens(fields_out)
