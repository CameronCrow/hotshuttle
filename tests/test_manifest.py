"""The manifest is the operator's contract, so its failures must be loud and specific."""
from __future__ import annotations

import pathlib

import pytest

from hotshuttle.core import manifest
from hotshuttle.profiles.bonsai import BonsaiProfile

REPO = pathlib.Path(__file__).resolve().parent.parent

GOOD = {
    "model": {"server": "http://127.0.0.1:8080", "n_slots": 1, "ctx_per_slot": 12288,
              "kv_quant": "q8_0", "sampling_defaults": {"temperature": 0.2}},
    "roles": [{"name": "reader", "system": "You read.", "ctx_budget": 8192,
               "tools": ["read_file"], "max_out": 512, "compact_at": 0.8}],
}


def test_the_shipped_manifest_loads():
    m = manifest.load(REPO / "workers.yaml")
    assert set(m.roles) == {"code-reader", "test-writer", "summarizer"}
    assert m.roles["code-reader"].tools == ("read_file", "grep")


def test_the_shipped_manifest_builds_a_working_profile():
    """model_kwargs() must actually satisfy a real profile constructor -- otherwise the
    manifest validates happily and then explodes at startup."""
    m = manifest.load(REPO / "workers.yaml")
    profile = BonsaiProfile(**m.model_kwargs())
    assert profile.ctx_per_slot == 12288
    assert profile.kv_quant == "q8_0"
    assert profile.server_url == "http://127.0.0.1:8080"
    for role in m.roles.values():
        assert role.ctx_budget <= profile.ctx_per_slot


def test_parses_a_good_manifest():
    m = manifest.parse(GOOD)
    assert m.roles["reader"].max_out == 512
    assert m.model_kwargs()["server_url"] == "http://127.0.0.1:8080"


@pytest.mark.parametrize("mutate, expected", [
    (lambda d: d.update(extra=1), "unknown top-level keys"),
    (lambda d: d["model"].update(gpu="yes"), "unknown model keys"),
    (lambda d: d["roles"][0].update(ctx_budgett=1), "unknown keys"),
    (lambda d: d["roles"][0].pop("system"), "missing `system`"),
    (lambda d: d["roles"].append(dict(d["roles"][0])), "duplicate role name"),
    (lambda d: d["roles"][0].update(ctx_budget=99999), "could never fit in a slot"),
    (lambda d: d["roles"][0].update(compact_at=1.5), "compact_at must be in (0, 1)"),
    (lambda d: d["roles"][0].update(max_out=99999), ">= ctx_budget"),
    (lambda d: d.update(roles=[]), "non-empty list"),
])
def test_rejects_bad_manifests(mutate, expected):
    import copy
    d = copy.deepcopy(GOOD)
    mutate(d)
    with pytest.raises(manifest.ManifestError, match=re_escape(expected)):
        manifest.parse(d)


def test_missing_file_is_a_manifest_error():
    with pytest.raises(manifest.ManifestError, match="no manifest at"):
        manifest.load("/nonexistent/workers.yaml")


def re_escape(s: str) -> str:
    import re
    return re.escape(s)
