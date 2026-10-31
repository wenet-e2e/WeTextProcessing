# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0

from pynini.lib.pynutil import add_weight, delete

from tn.processor import Processor


class Word(Processor):

    def __init__(self):
        super().__init__("word", token_orders={})
        word = (self.NOT_SPACE - self.DIGIT).plus
        space = add_weight(self.SPACE, -0.0001).star
        word |= space + self.PUNCT.plus + space
        self.tagger = self.add_tokens(self.tag_field("value", word))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", word | delete(self.SPACE.plus)))
