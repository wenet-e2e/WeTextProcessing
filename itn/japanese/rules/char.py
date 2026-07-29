# Copyright (c) 2024 Logan Liu (2319277867@qq.com)
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


class Char(Processor):

    def __init__(self, input_processor=None):
        super().__init__(name="char")
        self.input_processor = input_processor
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        self.graph = self.apply_input_processor(self.VCHAR, self.input_processor)
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))

    def build_verbalizer(self):
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
