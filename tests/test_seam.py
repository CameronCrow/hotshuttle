"""The core/ vs profiles/ seam (docs/PLAN.md section 5, issue #7).

Generalising to a second model is meant to be a move, not surgery. That only stays true
if core/ never learns anything about Bonsai -- and the way it stops being true is one
innocuous default at a time (a port here, a context length there), which no reviewer
notices. So it is a test.
"""
from __future__ import annotations

import pathlib
import re

import pytest

CORE = pathlib.Path(__file__).resolve().parent.parent / "hotshuttle" / "core"

# Identifiers that name this specific model, its serving fork, or its prompt format.
FORBIDDEN_WORDS = ["bonsai", "prism", "qwen", "im_start", "im_end", "deltanet", "ternary"]

# Measured constants that belong to profiles/bonsai.py. A bare number in core/ is how
# "12288" quietly becomes the default context length for every future model.
FORBIDDEN_NUMBERS = ["12288", "16384", "8080", "149.626", "34.1", "18.1", "q8_0", "q4_0"]


def core_sources():
    return sorted(p for p in CORE.glob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("path", core_sources(), ids=lambda p: p.name)
def test_core_module_names_no_model(path):
    text = path.read_text(encoding="utf-8")
    # Prose may reference the model when explaining *why* an invariant exists; code may
    # not. Strip comments and docstrings, check what is left.
    code = _strip_prose(text).lower()
    for word in FORBIDDEN_WORDS:
        assert word not in code, (
            f"{path.name} references '{word}' in code. Model-specific naming belongs in "
            f"hotshuttle/profiles/, and core/ must import only the ModelProfile protocol.")


@pytest.mark.parametrize("path", core_sources(), ids=lambda p: p.name)
def test_core_module_hardcodes_no_sizing(path):
    code = _strip_prose(path.read_text(encoding="utf-8"))
    for num in FORBIDDEN_NUMBERS:
        assert num not in code, (
            f"{path.name} hardcodes '{num}'. Sizing constants, ports and quant names are "
            f"profile data -- pass them in rather than defaulting to Bonsai's values.")


def test_core_does_not_import_a_concrete_profile():
    for path in core_sources():
        code = path.read_text(encoding="utf-8")
        assert "profiles" not in re.sub(r"#.*", "", code).split('"""')[0] + "", \
            f"{path.name} imports from profiles/"
        for line in code.splitlines():
            if line.strip().startswith(("import ", "from ")):
                assert "profiles" not in line, f"{path.name}: {line.strip()}"


def _strip_prose(text: str) -> str:
    """Remove docstrings and comments, leaving executable code."""
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    text = re.sub(r"'''[\s\S]*?'''", "", text)
    return re.sub(r"#.*", "", text)
