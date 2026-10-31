# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.lib import pynutil

from itn.russian.rules.utils import DIGIT, data_path, insertion_labels, number_names
from tn.processor import Processor


class Decimal(Processor):
    """Spoken decimals, e.g. "ноль целых и одна десятая" -> "0,1".

    decimal { integer_part: "ноль целых и" fractional_part: "одна десятая" }
    """

    def __init__(self, cardinal):
        super().__init__(name="decimal", ordertype="itn")
        names = number_names()
        delimiter = insertion_labels(data_path("numbers/decimal_delimiter.tsv"))["@@decimal_delimiter@@"]
        integer = names.cardinal + delimiter + pynini.closure(pynutil.add_weight(pynutil.insert(" и"), 0.5), 0, 1)
        self.integer_graph = integer.invert().optimize()
        endings = insertion_labels(data_path("numbers/decimal_endings.tsv"))
        with_zeros = pynini.closure(pynutil.delete("0")) + names.cardinal
        fractional = pynini.union(*[
            (DIGIT ** length) @ with_zeros + endings[str(10 ** length)] for length in range(1, 5)])
        self.fractional_graph = fractional.invert().optimize()
        self.quantity_graph = pynini.string_file(data_path("numbers/quantity.tsv")).invert().optimize()

        space = pynutil.delete(" ") + pynutil.insert(" ")
        quantity_tagger = space + self.tag_field("quantity", self.quantity_graph)
        self.fields_tagger = (
            cardinal.sign_tagger.ques + self.tag_field("integer_part", self.integer_graph)
            + space + self.tag_field("fractional_part", self.fractional_graph) + quantity_tagger.ques
        ).optimize()
        quantity_verbalizer = pynini.accep(" ") + self.verbalize_field("quantity", self.quantity_graph)
        self.fields_verbalizer = (
            cardinal.sign_verbalizer.ques + self.verbalize_field("integer_part", self.integer_graph)
            + pynutil.delete(" ") + pynutil.insert(",")
            + self.verbalize_field("fractional_part", self.fractional_graph) + quantity_verbalizer.ques
        ).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
