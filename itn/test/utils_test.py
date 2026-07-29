from pathlib import Path

import pytest

from itn.utils import get_abs_path


def test_get_abs_path_resolves_inside_itn_package():
    path = Path(get_abs_path("english/data/numbers/digit.tsv"))

    assert path.is_file()
    assert "itn" in path.parts


def test_get_abs_path_rejects_package_escape():
    with pytest.raises(ValueError, match="escapes the package"):
        get_abs_path("../tn/processor.py")
