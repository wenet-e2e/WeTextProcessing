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
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


class Cardinal(Processor):
    """Written integers such as ``-1.234`` are kept raw and spoken by the verbalizer."""

    def __init__(self):
        super().__init__("cardinal", token_orders={})
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        space = insert(" ")
        self.zero = pynini.string_file(get_abs_path("vietnamese/data/numbers/zero.tsv"))
        self.digit = pynini.string_file(get_abs_path("vietnamese/data/numbers/digit.tsv"))
        self.teen = pynini.string_file(get_abs_path("vietnamese/data/numbers/teen.tsv"))
        self.ties = pynini.string_file(get_abs_path("vietnamese/data/numbers/ties.tsv"))
        labels = load_labels(get_abs_path("vietnamese/data/numbers/magnitudes.tsv"))
        self.magnitudes = {row[0]: row[1] for row in labels if len(row) == 2}
        labels = load_labels(get_abs_path("vietnamese/data/numbers/digit_special.tsv"))
        special = {row[0]: {"std": row[1], "alt": row[2]} for row in labels if len(row) >= 3}

        special_digits = pynini.union(*[pynini.cross(k, special[k]["alt"]) for k in ("1", "4", "5")])
        self.linh_digits = pynini.union(*[pynini.cross(k, special[k]["std"]) for k in ("1", "4", "5")], self.digit)
        self.two_digit = pynini.union(
            self.teen,
            self.ties + delete("0"),
            self.ties + space + (special_digits | pynini.union("2", "3", "6", "7", "8", "9") @ self.digit),
        ).optimize()

        hundred = insert(self.magnitudes["hundred"])
        linh = insert(self.magnitudes["linh"])
        self.hundreds_pattern = pynini.union(
            self.digit + space + hundred + delete("00"),
            self.digit + space + hundred + delete("0") + space + linh + space + self.linh_digits,
            self.digit + space + hundred + space + self.two_digit,
            delete("0") + self.two_digit,
            delete("00") + self.digit,
        ).optimize()
        self.hundreds = (self.DIGIT**3 @ self.hundreds_pattern).optimize()

        self.magnitude_patterns = {}
        previous = None
        for name, low, high, zeros in (
            ("thousand", 4, 6, 3),
            ("million", 7, 9, 6),
            ("billion", 10, 12, 9),
            ("trillion", 13, 15, 12),
            ("quadrillion", 16, 18, 15),
            ("quintillion", 19, 21, 18),
        ):
            previous = self._magnitude_pattern(name, low, high, zeros, previous)
            self.magnitude_patterns[name] = previous

        self.graph = pynini.union(
            *self._split_patterns(),
            *self.magnitude_patterns.values(),
            self.hundreds,
            self.two_digit,
            self.digit,
            self.zero,
        ).optimize()
        self.digits = (self.digit | self.zero).optimize()
        # The written side of ``graph``; rules tag ``X @ graph`` fields with the cheaper ``X @ raw``.
        self.raw = self.input_projection(self.graph)
        self.negative = pynini.cross("-", "âm ")

    def _magnitude_pattern(self, name, low, high, zeros, previous):
        space = insert(" ")
        patterns = []
        for digits in range(low, high + 1):
            leading = digits - zeros
            leading_fst = {1: self.digit, 2: self.two_digit}.get(leading, self.hundreds_pattern)
            prefix = leading_fst + space + insert(self.magnitudes[name])
            options = [prefix + delete("0" * zeros)]
            if previous is not None and name not in ("quadrillion", "quintillion"):
                options.append(prefix + space + previous)
            for trailing in range(zeros):
                remaining = zeros - trailing
                trailing_prefix = prefix + delete("0" * trailing)
                if remaining == 1:
                    linh = trailing_prefix + space + insert(self.magnitudes["linh"]) + space + self.linh_digits
                    options.append(add_weight(linh, -0.1))
                elif remaining == 2:
                    options.append(trailing_prefix + space + self.two_digit)
                elif remaining == 3:
                    options.append(trailing_prefix + space + self.hundreds_pattern)
            patterns.append(self.DIGIT**digits @ pynini.union(*options))
        return pynini.union(*patterns).optimize()

    def _zero_or_million(self, digits):
        # NeMo leaves a dangling space for an all-zero tail and squeezes it later.
        million = self.DIGIT**digits @ self.magnitude_patterns["million"]
        return delete("0" * digits) | insert(" ") + million

    def _pattern_for_digits(self, count):
        if count == 1:
            return self.digit
        if count == 2:
            return self.two_digit
        if count == 3:
            return self.hundreds_pattern
        if count <= 6:
            return self.magnitude_patterns["thousand"]
        if count <= 9:
            return self.magnitude_patterns["million"]
        return self.magnitude_patterns["billion"]

    def _split_patterns(self):
        """Large numbers are read as ``front tỷ back``; dotted groups are also accepted."""

        space = insert(" ")
        billion = insert(self.magnitudes["billion"])
        back = 9
        patterns = []
        for total in range(13, 22):
            split = self._pattern_for_digits(total - back) + space + billion + self._zero_or_million(back)
            patterns.append(self.DIGIT**total @ add_weight(split, -0.5))

        delete_dot = delete(".")
        ranges = {4: (13, 14, 15), 5: (16, 17, 18), 6: (19, 20, 21)}
        for dots, magnitude in ((6, None), (5, None), (4, None), (3, "billion"), (2, "million"), (1, "thousand")):
            pattern = (self.DIGIT - "0") + pynini.closure(self.DIGIT, 0, 2)
            for _ in range(dots):
                pattern += delete_dot + self.DIGIT**3
            if magnitude is not None:
                patterns.append(add_weight(pattern, -0.3) @ self.magnitude_patterns[magnitude])
                continue
            for total in ranges[dots]:
                front = total - back
                split = (self.DIGIT**front @ self._pattern_for_digits(front)) + space + billion
                split += self._zero_or_million(back)
                patterns.append(pattern @ (self.DIGIT**total @ add_weight(split, -1.0)))
        return patterns

    def build_tagger(self):
        # Shared by every rule whose leading ``-`` reads as "âm".
        self.negative_tagger = (self.tag_field("negative", self.negative) + insert(" ")).optimize()
        self.tagger = self.add_tokens(self.negative_tagger.ques + self.tag_field("integer", self.raw))

    def build_verbalizer(self):
        self.negative_verbalizer = (self.verbalize_field("negative", self.negative) + delete(" ")).optimize()
        self.verbalizer = self.delete_tokens(self.negative_verbalizer.ques + self.verbalize_field("integer", self.graph))
