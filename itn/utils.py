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
"""Utilities owned by the inverse text normalization package."""

from pathlib import Path

from tn.utils import augment_labels_with_punct_at_end, get_formats, load_labels, str2bool

_PACKAGE_ROOT = Path(__file__).resolve().parent


def get_abs_path(rel_path):
    """Returns an ITN package resource path without escaping the package root."""

    path = (_PACKAGE_ROOT / rel_path).resolve()
    try:
        path.relative_to(_PACKAGE_ROOT)
    except ValueError as error:
        raise ValueError("ITN resource path escapes the package: {!r}".format(rel_path)) from error
    return str(path)


__all__ = [
    "augment_labels_with_punct_at_end",
    "get_abs_path",
    "get_formats",
    "load_labels",
    "str2bool",
]
