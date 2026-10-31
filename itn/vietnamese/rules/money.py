# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import pynini
from pynini.lib.pynutil import add_weight, delete, insert

from itn.utils import get_abs_path
from itn.vietnamese.rules.cardinal import DELETE_EXTRA_SPACE
from tn.processor import Processor


# The currency is spoken after the amount but written after its minor part.
TOKEN_ORDER = {"money": ["integer_part", "fractional_part", "minor_part", "quantity", "currency"]}


class Money(Processor):
    """``hai đô rưỡi`` keeps amount and currency words raw and becomes ``2.5$``."""

    def __init__(self, cardinal, decimal):
        super().__init__(name="money", ordertype="itn", token_orders=TOKEN_ORDER)
        sep = DELETE_EXTRA_SPACE + insert(" ")
        self.currency_graph = pynini.string_file(get_abs_path("vietnamese/data/currency.tsv")).invert().optimize()
        pad = self.DIGIT**2 | insert("0") + self.DIGIT
        # "hai đô năm mươi" -> 2.50$; "hai đô rưỡi" -> 2.5$.
        self.minor_graph = (add_weight(cardinal.graph_no_exception @ pad, -0.7) | cardinal.half).optimize()

        currency = self.tag_field("currency", self.currency_graph)
        minor = sep + self.tag_field("minor_part", self.minor_graph)
        integer = self.tag_field("integer_part", cardinal.graph_no_exception) + sep + currency + minor.ques
        # As in NeMo, a minor amount may also follow a bare scale: "một triệu đô năm mươi" -> 1.50 triệu$.
        scaled = self.tag_field("integer_part", decimal.scale_integer_graph) + sep
        scaled += self.tag_field("quantity", decimal.magnitude_graph) + sep + currency + minor
        self.tagger = self.add_tokens(integer | scaled | decimal.fields_tagger + sep + currency)

        space = delete(" ")
        minor_v = insert(".") + space + self.verbalize_field("minor_part", self.minor_graph)
        integer_v = self.verbalize_field("integer_part", cardinal.graph_no_exception) + minor_v.ques
        scaled_v = self.verbalize_field("integer_part", decimal.scale_integer_graph) + minor_v
        scaled_v += insert(" ") + space + self.verbalize_field("quantity", decimal.magnitude_graph)
        currency_v = space + self.verbalize_field("currency", self.currency_graph)
        self.verbalizer = self.delete_tokens((integer_v | scaled_v | decimal.fields_verbalizer) + currency_v)
