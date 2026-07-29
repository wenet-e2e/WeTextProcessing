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
"""Shared command-line interface for text and inverse text normalization."""

import argparse
import sys

LANGUAGES = ("zh", "en", "ja")


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected True or False")


def _add_boolean(parser, name, default, help):
    """Adds a Python 3.7-compatible boolean optional argument.

    ``--flag [True|False]`` keeps the historical command line working, while
    ``--flag`` and ``--no-flag`` provide conventional boolean switches.
    """

    hyphenated = name.replace("_", "-")
    parser.add_argument(
        "--" + name,
        "--" + hyphenated,
        dest=name,
        nargs="?",
        const=True,
        default=default,
        type=_parse_bool,
        metavar="BOOL",
        help=help,
    )
    parser.add_argument(
        "--no-" + name,
        "--no-" + hyphenated,
        dest=name,
        action="store_false",
        help=argparse.SUPPRESS,
    )


def _add_common_arguments(parser):
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--text", help="input string; read stdin when omitted")
    inputs.add_argument("--file", help="input file path; read stdin when omitted")
    parser.add_argument("--cache_dir", "--cache-dir", default=None, help="FST cache root")
    parser.add_argument(
        "--overwrite_cache",
        "--overwrite-cache",
        action="store_true",
        help="rebuild the selected FST cache bundle",
    )
    parser.add_argument("--language", choices=LANGUAGES, default="zh", help="input language (default: zh)")


def _add_tn_arguments(parser, language):
    if language == "zh":
        _add_boolean(parser, "remove_interjections", True, 'remove interjections such as "啊"')
        _add_boolean(parser, "remove_erhua", True, 'remove "儿" where the whitelist permits')
        _add_boolean(parser, "traditional_to_simple", True, 'convert traditional characters, e.g. "喆" to "哲"')
        _add_boolean(parser, "remove_puncts", False, "remove punctuation")
        _add_boolean(parser, "full_to_half", True, 'convert full-width characters, e.g. "Ａ" to "A"')
        _add_boolean(parser, "tag_oov", False, 'tag out-of-vocabulary characters with "OOV"')
    elif language == "ja":
        _add_boolean(parser, "transliterate", False, "enable Japanese transliteration")
        _add_boolean(parser, "remove_interjections", False, "remove interjections")
        _add_boolean(parser, "remove_puncts", False, "remove punctuation")
        _add_boolean(parser, "full_to_half", True, "convert full-width characters to half-width")
        _add_boolean(parser, "tag_oov", False, 'tag out-of-vocabulary characters with "OOV"')


def _add_itn_arguments(parser, language):
    if language in ("zh", "ja"):
        _add_boolean(parser, "enable_standalone_number", True, "convert standalone numbers")
        _add_boolean(parser, "enable_0_to_9", False, "convert standalone digits from zero to nine")
        _add_boolean(parser, "enable_million", False, "write million-scale values entirely in digits")
    if language == "zh":
        _add_boolean(parser, "remove_interjections", True, "remove interjections")
    elif language == "ja":
        _add_boolean(parser, "full_to_half", False, "convert full-width characters to half-width")


def create_parser(direction, language="zh"):
    """Builds the parser for one direction/language capability set."""

    if direction not in ("tn", "itn"):
        raise ValueError("direction must be 'tn' or 'itn'")
    if language not in LANGUAGES:
        raise ValueError("unsupported language: {}".format(language))
    label = "text normalization" if direction == "tn" else "inverse text normalization"
    parser = argparse.ArgumentParser(description="WeTextProcessing {}".format(label))
    _add_common_arguments(parser)
    if direction == "tn":
        _add_tn_arguments(parser, language)
    else:
        _add_itn_arguments(parser, language)
    return parser


def _selected_language(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--language", choices=LANGUAGES, default="zh")
    args, _ = parser.parse_known_args(argv)
    return args.language


def parse_args(direction, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    language = _selected_language(argv)
    return create_parser(direction, language).parse_args(argv)


def create_processor(direction, args):
    common = {
        "cache_dir": args.cache_dir,
        "overwrite_cache": args.overwrite_cache,
    }
    if direction == "tn":
        if args.language == "zh":
            from tn.chinese.normalizer import Normalizer

            return Normalizer(remove_interjections=args.remove_interjections,
                              remove_erhua=args.remove_erhua,
                              traditional_to_simple=args.traditional_to_simple,
                              remove_puncts=args.remove_puncts,
                              full_to_half=args.full_to_half,
                              tag_oov=args.tag_oov,
                              **common)
        if args.language == "ja":
            from tn.japanese.normalizer import Normalizer

            return Normalizer(transliterate=args.transliterate,
                              remove_interjections=args.remove_interjections,
                              remove_puncts=args.remove_puncts,
                              full_to_half=args.full_to_half,
                              tag_oov=args.tag_oov,
                              **common)
        from tn.english.normalizer import Normalizer

        return Normalizer(**common)

    if args.language == "zh":
        from itn.chinese.inverse_normalizer import InverseNormalizer

        return InverseNormalizer(remove_interjections=args.remove_interjections,
                                 enable_standalone_number=args.enable_standalone_number,
                                 enable_0_to_9=args.enable_0_to_9,
                                 enable_million=args.enable_million,
                                 **common)
    if args.language == "ja":
        from itn.japanese.inverse_normalizer import InverseNormalizer

        return InverseNormalizer(full_to_half=args.full_to_half,
                                 enable_standalone_number=args.enable_standalone_number,
                                 enable_0_to_9=args.enable_0_to_9,
                                 enable_million=args.enable_million,
                                 **common)
    from itn.english.inverse_normalizer import InverseNormalizer

    return InverseNormalizer(**common)


def _without_line_ending(line):
    if line.endswith("\n"):
        line = line[:-1]
        if line.endswith("\r"):
            line = line[:-1]
    elif line.endswith("\r"):
        line = line[:-1]
    return line


def _process_line(processor, text, stdout):
    tagged = processor.tag(text)
    print(tagged, file=stdout)
    print(processor.verbalize(tagged), file=stdout)


def run(direction, argv=None, stdin=None, stdout=None, processor_factory=create_processor):
    """Runs a CLI entry point and returns the constructed processor."""

    args = parse_args(direction, argv)
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    processor = processor_factory(direction, args)

    if args.text is not None:
        _process_line(processor, args.text, stdout)
        return processor

    if args.file is not None:
        try:
            with open(args.file, encoding="utf-8") as input_file:
                for line in input_file:
                    _process_line(processor, _without_line_ending(line), stdout)
        except OSError as error:
            create_parser(direction, args.language).error(str(error))
        return processor

    for line in stdin:
        _process_line(processor, _without_line_ending(line), stdout)
    return processor
