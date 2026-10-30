# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0

import pynini

from tn.processor import Processor


class Punctuation(Processor):

    def __init__(self):
        super().__init__("punctuation", token_orders={})
        graph = pynini.cross("〜", "부터") | pynini.accep("부터")
        self.tagger = self.add_tokens(self.tag_field("value", graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", graph))
