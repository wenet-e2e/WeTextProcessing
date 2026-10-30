# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from itn.utils import get_abs_path
from tn.processor import Processor


class Whitelist(Processor):
    def __init__(self):
        super().__init__(name="whitelist", ordertype="itn")
        graph = string_file(get_abs_path("korean/data/whitelist.tsv"))
        self.tagger = self.add_tokens(self.tag_field("value", graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", graph))
