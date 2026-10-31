# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from itn.utils import get_abs_path, load_labels
from tn.processor import Processor


class Whitelist(Processor):
    """Raw whitelisted phrases ("usted"); V writes the listed form ("Ud.").

    The code-mixed grammar carries its own Spanish and English lists: the
    English one drops "DOS -> dos" so the Spanish number "dos" survives.
    """

    def __init__(self, name, tsv):
        super().__init__(name, ordertype="itn")
        # Spoken form -> written form; the first written form of a phrase wins.
        # An optional third column is a path weight, as in a pynini string file.
        mapping = {}
        for written, spoken, *weight in load_labels(get_abs_path("spanish_english/data/" + tsv)):
            mapping.setdefault(spoken, (spoken, written, *weight))
        self.graph = pynini.string_map([mapping[spoken] for spoken in sorted(mapping)]).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
