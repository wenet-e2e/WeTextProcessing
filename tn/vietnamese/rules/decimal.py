# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {"decimal": ["negative", "integer_part", "fractional_part", "quantity", "minor_part"]}

# ``1tr2`` reads "một triệu hai trăm nghìn": the trailing digit counts hundreds of the next scale.
COMPOUND_SCALES = {"tr": ("triệu", "trăm nghìn"), "t": ("tỷ", "trăm triệu")}


class Decimal(Processor):
    """Comma decimals, scale words and scale abbreviations (``1,5 triệu``, ``10k``, ``1tr2``)."""

    def __init__(self, cardinal):
        super().__init__("decimal", token_orders=TOKEN_ORDER)
        abbr_labels = load_labels(get_abs_path("vietnamese/data/numbers/quantity_abbr.tsv"))
        magnitudes = load_labels(get_abs_path("vietnamese/data/numbers/magnitudes.tsv"))
        prefixes = load_labels(get_abs_path("vietnamese/data/measure/prefixes.tsv"))
        measure_prefixes = {row[0].lower() for row in prefixes}

        quantity_words = pynini.union(*[row[1] for row in magnitudes]).optimize()
        abbreviations = pynini.string_map(abbr_labels).optimize()
        # Abbreviations that are also measure prefixes (k, t) never follow a fraction.
        safe_abbreviations = pynini.union(*[row[0] for row in abbr_labels if row[0].lower() not in measure_prefixes])
        self.quantity_graph = (quantity_words | abbreviations).optimize()
        self.fractional_graph = (cardinal.digits + (insert(" ") + cardinal.digits).star).optimize()
        dotted = self.DIGIT + pynini.closure(self.DIGIT, 0, 2) + (delete(".") + self.DIGIT**3).plus
        self.integer_graph = (cardinal.graph | dotted @ cardinal.graph).optimize()

        sep = insert(" ")
        integer = self.tag_field("integer_part", cardinal.raw)
        dotted_integer = self.tag_field("integer_part", dotted @ cardinal.graph)
        short_integer = self.tag_field("integer_part", pynini.closure(self.DIGIT, 1, 3) @ cardinal.raw)
        fraction = delete(",") + self.tag_field("fractional_part", self.DIGIT.plus)
        word_quantity = delete(" ").ques + sep + self.tag_field("quantity", quantity_words)
        safe_quantity = sep + self.tag_field("quantity", safe_abbreviations)
        compound = sep + self.tag_field("quantity", pynini.union(*COMPOUND_SCALES))
        compound += sep + self.tag_field("minor_part", self.DIGIT)

        with_integer = pynini.union(
            (integer | dotted_integer) + sep + fraction + word_quantity.ques,
            integer + word_quantity,
            short_integer + sep + self.tag_field("quantity", abbreviations),
            integer + sep + fraction + safe_quantity,
            short_integer + compound,
        )
        self.with_integer_tagger = with_integer.optimize()
        self.fields_tagger = (with_integer | fraction + (word_quantity | safe_quantity).ques).optimize()
        self.tagger = self.add_tokens(cardinal.negative_tagger.ques + self.fields_tagger)

        integer_v = self.verbalize_field("integer_part", self.integer_graph)
        fraction_v = self.verbalize_field("fractional_part", self.fractional_graph, leading_space=True)
        quantity_v = insert(" ") + self.verbalize_field("quantity", self.quantity_graph, leading_space=True)
        compound_v = pynini.union(*[
            insert(" ") + self.verbalize_field("quantity", pynini.cross(abbr, scale), leading_space=True)
            + insert(" ") + self.verbalize_field("minor_part", cardinal.digits, leading_space=True)
            + insert(" " + hundreds)
            for abbr, (scale, hundreds) in COMPOUND_SCALES.items()
        ])
        self.with_integer_verbalizer = (
            integer_v + insert(" phẩy ") + fraction_v + quantity_v.ques
            | integer_v + quantity_v
            | integer_v + compound_v
        ).optimize()
        fraction_only_v = insert("phẩy ") + self.verbalize_field("fractional_part", self.fractional_graph)
        self.fields_verbalizer = (self.with_integer_verbalizer | fraction_only_v + quantity_v.ques).optimize()
        self.verbalizer = self.delete_tokens(cardinal.negative_verbalizer.ques + self.fields_verbalizer)
