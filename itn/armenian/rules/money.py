# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from itn.armenian.rules.utils import preferred_map
from tn.processor import Processor


class Money(Processor):
    """Spoken amounts with a currency, e.g. ``քսան հազար դրամ`` -> ``20000 ֏``.

    The tagger keeps the spoken amount and currency name; the verbalizer
    writes the digits followed by the currency symbol.
    """

    def __init__(self, cardinal, decimal):
        super().__init__(name="money", ordertype="itn", token_orders={})
        currency = preferred_map("armenian/data/currency.tsv", invert=True)
        amount_tag = self.tag_field("integer_part", cardinal.graph) | decimal.fields_tagger
        currency_tag = delete(self.SPACE.plus) + insert(" ") + self.tag_field("currency", currency)
        self.tagger = self.add_tokens(amount_tag + currency_tag)

        amount = self.verbalize_field("integer_part", cardinal.graph) | decimal.fields_verbalizer
        currency_out = insert(" ") + self.verbalize_field("currency", currency, leading_space=True)
        self.verbalizer = self.delete_tokens(amount + currency_out)
