# Copyright (c) 2022 Xingchen Song (sxc19@tsinghua.org.cn)
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

from importlib_resources import files
from pynini.lib.pynutil import add_weight, delete

from itn.english.rules.cardinal import Cardinal
from itn.english.rules.word import Word
from tn.processor import Processor


class InverseNormalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="en_inverse_normalizer", ordertype="itn")
        if cache_dir is None:
            cache_dir = files("itn")
        self.build_fst("en_itn", cache_dir, overwrite_cache)

    def build_tagger(self):
        tagger = (add_weight(Cardinal().tagger, 1.0) | add_weight(Word().tagger, 100)).optimize()

        tagger = tagger.star
        # remove the last space
        self.tagger = tagger @ self.build_rule(delete(" "), "", "[EOS]")

    def build_verbalizer(self):
        verbalizer = (Cardinal().verbalizer | Word().verbalizer).optimize()
        self.verbalizer = verbalizer.star
