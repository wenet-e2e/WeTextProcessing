# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Hungarian alphabet, data loading and the naive suffix inflector."""

from functools import lru_cache

import pynini
from pynini.lib import byte, utf8

from itn.utils import get_abs_path
from tn.utils import load_labels

SIGMA = utf8.VALID_UTF8_CHAR.star
DIGIT = byte.DIGIT
NONZERO = pynini.difference(DIGIT, "0").optimize()
_ALPHA_UPPER = "AÁBCDEÉFGHIÍJKLMNOÓÖŐPQRSTUÚÜŰVWXYZ"
_ALPHA_LOWER = "aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz"
TO_LOWER = pynini.union(*[pynini.cross(x, y) for x, y in zip(_ALPHA_UPPER, _ALPHA_LOWER)]).optimize()


def data_path(name):
    return get_abs_path("hungarian/data/" + name)


def data_labels(name):
    return [row for row in load_labels(data_path(name)) if row]


def data_map(name, invert=False):
    """Maps column 0 to column 1 (or the reverse); one-column rows map to themselves."""

    pairs = []
    for row in data_labels(name):
        key, value = row[0], row[1] if len(row) > 1 else row[0]
        pairs.append((value, key) if invert else (key, value))
    return pynini.string_map(pairs).optimize()


def data_keys(name):
    return pynini.union(*[pynini.escape(row[0]) for row in data_labels(name)]).optimize()


def suffix_rewrite(mapping, exceptions=None):
    """Rewrites the end of a bounded word with ``mapping``.

    ``exceptions`` (whole-word replacements) take precedence; the result is
    meant to be composed onto a bounded spoken graph.
    """

    graph = SIGMA + mapping
    if exceptions is None:
        return graph.optimize()
    keys = pynini.project(exceptions, "input")
    return (exceptions | pynini.difference(SIGMA, keys) @ graph).optimize()


@lru_cache(maxsize=None)
def _inflection_tables():
    def load(name):
        return {row[0]: row[1].split(" ") for row in data_labels("inflection/" + name)}

    return load("endings.tsv"), load("plural_endings.tsv"), load("word_endings.tsv")


def _modify_ending(outword, word, form):
    """Adjusts endings that are written differently after abbreviations."""

    if outword == word:
        return form
    undouble = {"nny": "ny", "lly": "ly"}
    for ending in ["ny", "nny", "ly", "lly", "ev", "év", "út", "ut", "a", "á", "e", "é"]:
        if form.startswith(ending):
            return undouble.get(ending, "") + form[len(ending):]
    return form


def naive_inflector(abbr, word, singular_only=False):
    """Inflects an abbreviation together with its expansion.

    Returns ``(inflected abbreviation, inflected expansion)`` pairs, e.g.
    ``("tv.-t", "törvényt")``; possessive forms are omitted.
    """

    singular, plural, lexical = _inflection_tables()
    keys_sorted = sorted(singular, key=len, reverse=True)
    if word in lexical:
        key, ends = word, lexical[word]
    else:
        key = next((key for key in keys_sorted if word.endswith(key)), None)
        if key is None:
            raise KeyError("no inflection class for {}".format(word))
        ends = singular[key]
    outword = word
    for wordend in ["ny", "ly", "év", "út", "a", "e"]:
        if outword.endswith(wordend):
            outword = outword[:-len(wordend)]
    if "-" in abbr:
        abbr = abbr.split("-")[0]
    forms = [("{}-{}".format(abbr, _modify_ending(outword, word, form)), outword + form) for form in ends]
    if not singular_only:
        for plural_form in plural[key]:
            plural_key = key + "k" if plural_form == "k" else plural_form
            tweaked = _modify_ending(outword, word, plural_form)
            forms.append(("{}-{}".format(abbr, tweaked), outword + plural_form))
            for form in singular[plural_key]:
                forms.append(("{}-{}{}".format(abbr, tweaked, form), outword + plural_form + form))
    return forms


def inflect_abbreviation(abbr, word, singular_only=False):
    """Inflects an abbreviation by the name of its last letter.

    Returns ``(inflected abbreviation, inflected expansion)`` pairs, e.g.
    ``("GBP-t", "angol fontot")``.
    """

    last = abbr[-1].lower()
    if last in "bcdgjptvz":
        ending = "é"
    elif last in "aáeéiíoóöőuúüű":
        ending = last
    elif last in "flmnrs":
        ending = "e" + last
    elif last in "hk":
        ending = "á"
    else:
        return []
    word_part = [form[1] for form in naive_inflector(".", word, singular_only)]
    abbr_part = [form[0] for form in naive_inflector(abbr, ending, singular_only)]
    return list(zip(abbr_part, word_part))
