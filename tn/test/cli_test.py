import io

import pytest

from tn.cli import create_processor, parse_args, run


class FakeProcessor:

    def __init__(self):
        self.tag_calls = []
        self.verbalize_calls = []

    def tag(self, text):
        self.tag_calls.append(text)
        return "<{}>".format(text)

    def verbalize(self, tagged):
        self.verbalize_calls.append(tagged)
        return "[{}]".format(tagged)


@pytest.mark.parametrize(
    "direction,language,module_name,class_name,extra_args,expected",
    [
        (
            "tn",
            "zh",
            "tn.chinese.normalizer",
            "Normalizer",
            ["--no-remove-erhua", "--tag-oov"],
            {
                "remove_erhua": False,
                "tag_oov": True
            },
        ),
        (
            "tn",
            "ja",
            "tn.japanese.normalizer",
            "Normalizer",
            ["--transliterate", "--full-to-half", "False"],
            {
                "transliterate": True,
                "full_to_half": False
            },
        ),
        ("tn", "en", "tn.english.normalizer", "Normalizer", [], {}),
        (
            "itn",
            "zh",
            "itn.chinese.inverse_normalizer",
            "InverseNormalizer",
            ["--enable-0-to-9", "--no-enable-million"],
            {
                "enable_0_to_9": True,
                "enable_million": False
            },
        ),
        (
            "itn",
            "ja",
            "itn.japanese.inverse_normalizer",
            "InverseNormalizer",
            ["--full-to-half", "True", "--enable-standalone-number", "False"],
            {
                "full_to_half": True,
                "enable_standalone_number": False
            },
        ),
        ("itn", "en", "itn.english.inverse_normalizer", "InverseNormalizer", [], {}),
    ],
)
def test_create_processor_uses_language_capabilities(
    monkeypatch,
    direction,
    language,
    module_name,
    class_name,
    extra_args,
    expected,
):
    captured = {}

    class FakeNormalizer:

        def __init__(self, **kwargs):
            captured.update(kwargs)

    module = __import__(module_name, fromlist=[class_name])
    monkeypatch.setattr(module, class_name, FakeNormalizer)
    args = parse_args(direction, ["--language", language, "--cache-dir", "/tmp/cache"] + extra_args)

    create_processor(direction, args)

    assert captured["cache_dir"] == "/tmp/cache"
    assert captured["overwrite_cache"] is False
    for name, value in expected.items():
        assert captured[name] is value


def test_boolean_options_support_switches_and_legacy_values():
    switched = parse_args("tn", ["--remove-erhua", "--tag-oov", "--no-full-to-half"])
    legacy = parse_args("tn", ["--remove_erhua", "False", "--tag_oov", "True"])

    assert switched.remove_erhua is True
    assert switched.tag_oov is True
    assert switched.full_to_half is False
    assert legacy.remove_erhua is False
    assert legacy.tag_oov is True


def test_invalid_boolean_is_rejected():
    with pytest.raises(SystemExit):
        parse_args("tn", ["--remove-erhua", "yes"])


@pytest.mark.parametrize(
    "direction,language,option",
    [
        ("tn", "en", "--remove-puncts"),
        ("tn", "zh", "--transliterate"),
        ("itn", "en", "--enable-million"),
        ("itn", "ja", "--remove-interjections"),
    ],
)
def test_unsupported_language_option_is_rejected(direction, language, option):
    with pytest.raises(SystemExit):
        parse_args(direction, ["--language", language, option])


def test_text_is_tagged_once_and_verbalized_from_that_tag():
    processor = FakeProcessor()
    stdout = io.StringIO()

    run("tn", ["--text", "  12  "], stdout=stdout, processor_factory=lambda _direction, _args: processor)

    assert processor.tag_calls == ["  12  "]
    assert processor.verbalize_calls == ["<  12  >"]
    assert stdout.getvalue() == "<  12  >\n[<  12  >]\n"


def test_stdin_removes_only_line_endings():
    processor = FakeProcessor()
    stdout = io.StringIO()

    run(
        "itn",
        [],
        stdin=io.StringIO("  one  \n two \r\n"),
        stdout=stdout,
        processor_factory=lambda _direction, _args: processor,
    )

    assert processor.tag_calls == ["  one  ", " two "]


def test_file_input_preserves_spaces(tmp_path):
    input_path = tmp_path / "input.txt"
    input_path.write_text("  first  \n second ", encoding="utf-8")
    processor = FakeProcessor()

    run(
        "tn",
        ["--file", str(input_path)],
        stdout=io.StringIO(),
        processor_factory=lambda _direction, _args: processor,
    )

    assert processor.tag_calls == ["  first  ", " second "]


def test_text_and_file_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args("tn", ["--text", "a", "--file", "input.txt"])
