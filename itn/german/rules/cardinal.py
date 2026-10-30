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
from pynini.lib import pynutil

from tn.processor import Processor


TOKEN_ORDER = {}


class Cardinal(Processor):
    """German spoken cardinals; retain source words until verbalization."""

    def __init__(self, tn_cardinal):
        super().__init__("cardinal", ordertype="itn", token_orders=TOKEN_ORDER)
        # Only spaces present in TN's numeral output may be omitted. This is
        # not a rewrite allowing arbitrary spaces between spoken characters.
        omit_spaces = pynini.closure(self.VSIGMA | pynutil.delete(" "))
        self.graph_no_exception = pynutil.add_weight(
            (tn_cardinal.graph @ omit_spaces).invert().optimize(), 0.001,
        )
        self.digits = pynini.arcmap(tn_cardinal.digits, map_type="rmweight").invert().optimize()
        exceptions = self.input_projection(self.digits)
        self.graph = (
            (self.input_projection(self.graph_no_exception) - exceptions) @ self.graph_no_exception
        ).optimize()
        self.hundreds = (tn_cardinal.hundreds @ omit_spaces).invert().optimize()
        self.two_digits = (tn_cardinal.two_digits @ omit_spaces).invert().optimize()
        self.sign = pynini.cross("minus", "-")

        sign_tagger = pynini.closure(
            self.tag_field("negative", self.sign) + pynutil.delete(" ") + pynutil.insert(" "), 0, 1,
        )
        sign_verbalizer = pynini.closure(
            self.verbalize_field("negative", self.sign) + pynutil.delete(" "), 0, 1,
        )
        self.tagger = self.add_tokens(sign_tagger + self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(sign_verbalizer + self.verbalize_field("integer", self.graph))
