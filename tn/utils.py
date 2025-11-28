# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2024, WENET COMMUNITY.  Xingchen Song (sxc19@tsinghua.org.cn).
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

import csv
import os

import pynini


def get_abs_path(rel_path):
    """
    Get absolute path

    Args:
        rel_path: relative path to this file

    Returns absolute path
    """
    return os.path.dirname(os.path.abspath(__file__)) + "/" + rel_path


def load_labels(abs_path):
    """
    loads relative path file as dictionary

    Args:
        abs_path: absolute path

    Returns dictionary of mappings
    """
    with open(abs_path, encoding="utf-8") as label_tsv:
        labels = list(csv.reader(label_tsv, delimiter="\t"))
    return labels


def augment_labels_with_punct_at_end(labels):
    """
    augments labels: if key ends on a punctuation that value does not have, add a new label
    where the value maintains the punctuation

    Args:
        labels : input labels
    Returns:
        additional labels
    """
    res = []
    for label in labels:
        if len(label) > 1:
            if label[0][-1] == "." and label[1][-1] != ".":
                res.append([label[0], label[1] + "."] + label[2:])
    return res


def get_formats(input_f, input_case="cased", is_default=True):
    """
    Adds various abbreviation format options to the list of acceptable input forms
    """
    multiple_formats = load_labels(input_f)
    additional_options = []
    for x, y in multiple_formats:
        if input_case == "lower_cased":
            x = x.lower()
        additional_options.append((f"{x}.", y))  # default "dr" -> doctor, this includes period "dr." -> doctor
        additional_options.append((f"{x[0].upper() + x[1:]}", f"{y[0].upper() + y[1:]}"))  # "Dr" -> Doctor
        additional_options.append((f"{x[0].upper() + x[1:]}.", f"{y[0].upper() + y[1:]}"))  # "Dr." -> Doctor
    multiple_formats.extend(additional_options)

    if not is_default:
        multiple_formats = [(x, f"|raw_start|{x}|raw_end||norm_start|{y}|norm_end|") for (x, y) in multiple_formats]

    multiple_formats = pynini.string_map(multiple_formats)
    return multiple_formats
