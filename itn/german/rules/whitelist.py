# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini

from tn.processor import Processor


TOKEN_ORDER = {}


class WhiteList(Processor):
    def __init__(self, tn_whitelist):
        super().__init__("whitelist", ordertype="itn", token_orders=TOKEN_ORDER)
        self.graph = (
            tn_whitelist.graph.copy().invert()
            @ pynini.cdrewrite(pynini.cross("\u00a0", " "), "", "", self.VSIGMA)
        ).optimize()
        self.fields_tagger = self.tag_field("value", self.graph)
        self.fields_verbalizer = self.verbalize_field("value", self.graph)
        self.tagger = self.add_tokens(self.fields_tagger)
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
