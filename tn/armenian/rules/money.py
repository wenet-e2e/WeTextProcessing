# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pynini.lib.pynutil import delete, insert

from tn.armenian.rules.utils import preferred_map
from tn.processor import Processor


class Money(Processor):
    """Amounts followed by a currency, e.g. ``15 $`` -> ``տասնհինգ դոլար``.

    The tagger keeps the written amount (cardinal or decimal fields) and the
    currency symbol; the verbalizer reads the amount and names the currency.
    """

    def __init__(self, cardinal, decimal):
        super().__init__("money", token_orders={})
        currency = preferred_map("armenian/data/currency.tsv")
        amount_tag = self.tag_field("integer_part", cardinal.graph) | decimal.fields_tagger
        currency_tag = delete(" ").ques + insert(" ") + self.tag_field("currency", currency)
        self.tagger = self.add_tokens(amount_tag + currency_tag)

        amount = self.verbalize_field("integer_part", cardinal.graph) | decimal.fields_verbalizer
        currency_out = insert(" ") + self.verbalize_field("currency", currency, leading_space=True)
        self.verbalizer = self.delete_tokens(amount + currency_out)
