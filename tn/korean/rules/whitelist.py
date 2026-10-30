# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0

import pynini

from tn.processor import Processor
from tn.utils import get_abs_path


class WhiteList(Processor):

    def __init__(self):
        super().__init__("whitelist", token_orders={})
        graph = pynini.string_file(get_abs_path("korean/data/whitelist.tsv"))
        self.tagger = self.add_tokens(self.tag_field("value", graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", graph))
