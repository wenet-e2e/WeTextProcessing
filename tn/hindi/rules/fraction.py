# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from tn.hindi.rules.utils import HI_DEDH, HI_DHAI, HI_PAUNE, HI_SADHE, HI_SAVVA, MINUS, hindi_path
from tn.processor import Processor

TOKEN_ORDER = {"fraction": ["negative", "quarter", "integer_part", "numerator", "denominator", "value"]}


class Fraction(Processor):
    """``२३ ४/६`` keeps ``integer_part``, ``numerator`` and ``denominator`` raw.

    ``५ १/४`` (सवा पाँच) keeps ``integer_part`` plus a raw ``quarter`` field, which is
    ordered first so the verbalizer shares one integer reading; the other quarter
    idioms such as ``२ १/२`` (ढाई) stay one raw ``value`` field.
    """

    def __init__(self, cardinal):
        super().__init__("fraction", token_orders=TOKEN_ORDER)
        number = cardinal.final_graph
        sign_tagger = self.tag_field("negative", pynini.accep("-")) + insert(" ")
        integer_tagger = self.tag_field("integer_part", number)
        regular_tagger = (sign_tagger.ques + (integer_tagger + delete(" ") + insert(" ")).ques
                          + self.tag_field("numerator", number) + delete(pynini.union("/", " / ")) + insert(" ")
                          + self.tag_field("denominator", number))
        quarters = pynini.string_map([("१/४", HI_SAVVA), ("१/२", HI_SADHE)])
        quarter_tagger = add_weight(integer_tagger + delete(" ") + insert(" ") + self.tag_field("quarter", quarters),
                                    -0.1)

        paune = pynini.string_file(hindi_path("whitelist/paune_mappings.tsv"))
        idioms = pynini.union(
            add_weight(pynini.string_file(hindi_path("fraction/common_fractions.tsv")), -0.3),
            add_weight(pynini.string_map([("१ १/२", HI_DEDH), ("२ १/२", HI_DHAI)]), -0.2),
            add_weight(insert(HI_PAUNE + " ") + paune + delete(" ३/४"), -0.2),
        ).optimize()
        self.tagger = self.add_tokens(regular_tagger | quarter_tagger | self.tag_field("value", idioms))

        sign = self.verbalize_field("negative", pynini.cross("-", MINUS)) + delete(" ")
        quarter = self.verbalize_field("quarter", quarters) + delete(" ") + insert(" ")
        integer = self.verbalize_field("integer_part", number)
        conjunction = delete(" ") + insert(" और ")
        ratio = (self.verbalize_field("numerator", number) + delete(" ") + insert(" बटा ")
                 + self.verbalize_field("denominator", number))
        # One shared copy of every number reading; the tagger only emits the valid field
        # combinations (sign? integer? ratio, or quarter integer).
        fields = sign.ques + quarter.ques + (integer + conjunction.ques).ques + ratio.ques
        self.verbalizer = self.delete_tokens(fields | self.verbalize_field("value", idioms))
