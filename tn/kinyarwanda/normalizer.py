# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, DIGITAL UMUGANDA
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

from pynini import cdrewrite
from pynini.lib.pynutil import add_weight, delete

from tn.kinyarwanda.rules.cardinal import Cardinal
from tn.kinyarwanda.rules.punctuation import Punctuation
from tn.kinyarwanda.rules.time import Time
from tn.kinyarwanda.rules.whitelist import WhiteList
from tn.kinyarwanda.rules.word import Word
from tn.processor import Processor, RuleSpec

# Every Kinyarwanda rule emits its fields in spoken order, so no token needs
# reordering before verbalization.
TOKEN_ORDERS = {}


class Normalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="rw_normalizer", token_orders=TOKEN_ORDERS)
        self.build_fst("rw_tn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        punctuation = Punctuation()
        word = Word()
        rules = (
            RuleSpec(WhiteList(), 1.01),
            RuleSpec(Time(), 1.05),
            RuleSpec(Cardinal(), 1.1),
            RuleSpec(word, 100, verbalize=False),
            RuleSpec(punctuation, verbalize=False),
        )
        classify = self.tagger_union(rules)
        punct = add_weight(punctuation.tagger, 1.1)
        token = punct.star + classify + punct.star
        self.tagger = self.DELETE_SPACE + token + (self.DELETE_EXTRA_SPACE + token).star + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        verbalize = self.join_tokens(self.verbalizer_union(rules), (word.verbalizer, punctuation.verbalizer))
        # Whole hours read with a trailing space (fixture "1:00~saa saba "); drop the joining space after it.
        squeeze = cdrewrite(delete(" "), " ", self.NOT_SPACE, self.VSIGMA)
        self.verbalizer = (verbalize @ squeeze).optimize()
