# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pynini
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Ordinal(Processor):
    """``thứ 4`` keeps the written number and reads as "thứ tư"."""

    def __init__(self, cardinal):
        super().__init__("ordinal", token_orders={})
        exceptions = pynini.string_file(get_abs_path("vietnamese/data/ordinal/ordinal_exceptions.tsv"))
        number = self.DIGIT.plus
        # Exceptions ("1" -> "nhất", "4" -> "tư") replace the plain cardinal reading.
        regular = pynini.difference(number, self.input_projection(exceptions))
        self.graph = (exceptions | regular @ cardinal.graph).optimize()
        self.tagger = self.add_tokens(delete("thứ ") + self.tag_field("integer", number @ cardinal.raw))
        self.verbalizer = self.delete_tokens(insert("thứ ") + self.verbalize_field("integer", self.graph))
