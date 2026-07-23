"""Prompt bytes (I3) and the template that produces them.

Every test here is guarding against a silent failure: a prompt that is not a byte
extension of its predecessor still generates perfectly good text, it just re-prefills the
whole context every turn and nothing complains.
"""
from __future__ import annotations

import pathlib

import pytest

from hotshuttle.core import compiler
from hotshuttle.core.orchestrator import Orchestrator
from hotshuttle.core.pool import SlotPool
from hotshuttle.core.worker import Role, Task
from hotshuttle.profiles.bonsai import BonsaiProfile, QwenChatML
from tests.fake import FakeLlama

GROUND_TRUTH = (pathlib.Path(__file__).resolve().parent.parent
                / "experiments" / "m0_template_ground_truth.txt")


# --- the template must match the server, byte for byte -----------------------

def test_template_matches_the_servers_own_jinja_render():
    """Captured from POST /apply-template with enable_thinking=false in M0.

    If these bytes drift from what the server would render, nothing breaks loudly -- the
    prefix cache just stops matching and every turn re-prefills.
    """
    expected = GROUND_TRUTH.read_text(encoding="utf-8")
    t = QwenChatML()
    assert t.system("SYSTEM_MARKER") + t.user("USER_MARKER") + t.assistant_open() == expected


def test_assistant_open_suppresses_thinking():
    """The empty think block is what enable_thinking=false renders. Without it Bonsai
    spends its whole budget reasoning and returns empty content."""
    assert QwenChatML().assistant_open().endswith("<think>\n\n</think>\n\n")


# --- seed and turn rendering -------------------------------------------------

def test_render_is_deterministic():
    t = QwenChatML()
    a = compiler.render_turn(t, "do the thing", ["ctx one", "ctx two"])
    b = compiler.render_turn(t, "do the thing", ["ctx one", "ctx two"])
    assert a == b


def test_seed_embeds_tools_and_prior_progress():
    t = QwenChatML()
    seed = compiler.render_seed(t, "You are a worker.", ["read_file"], prior="did step 1")
    assert "read_file" in seed and "Prior progress:" in seed and "did step 1" in seed


def test_attachments_are_delimited():
    t = QwenChatML()
    turn = compiler.render_turn(t, "summarize", ["file contents here"])
    assert "<context>\nfile contents here\n</context>" in turn


# --- the append-only contract ------------------------------------------------

def test_prefix_guard_accepts_appends_and_rejects_edits():
    g = compiler.PrefixGuard()
    g.check("A", "seed + turn1")
    g.check("A", "seed + turn1 + turn2")
    with pytest.raises(AssertionError, match="not append-only"):
        g.check("A", "SEED + turn1 + turn2")


def test_prefix_guard_reports_where_it_diverged():
    g = compiler.PrefixGuard()
    g.check("A", "aaaabbbb")
    with pytest.raises(AssertionError, match="byte 4"):
        g.check("A", "aaaaXbbb")


def test_prefix_guard_reset_allows_a_new_seed_after_compaction():
    g = compiler.PrefixGuard()
    g.check("A", "old seed + history")
    g.reset("A")
    g.check("A", "new seed with summary")      # I4: compaction legitimately changes it


def test_prefix_guard_tracks_workers_independently():
    g = compiler.PrefixGuard()
    g.check("A", "seed A")
    g.check("B", "seed B")
    g.check("A", "seed A + more")


@pytest.mark.asyncio
async def test_every_dispatched_prompt_extends_the_previous_one():
    """The contract end to end: drive real turns and assert each prompt the orchestrator
    sends starts with the one before it."""
    fake = FakeLlama(n_slots=1)
    profile = BonsaiProfile(n_slots=1)
    orc = Orchestrator(fake, SlotPool(fake, 1), profile)
    orc.spawn("A", Role("A", "You are A.", ctx_budget=profile.ctx_per_slot))
    orc.spawn("B", Role("B", "You are B.", ctx_budget=profile.ctx_per_slot))

    seen: dict[str, list[str]] = {"A": [], "B": []}
    original = fake.complete

    async def spy(prompt, id_slot, n_predict, cache_prompt=True, **kw):
        wid = "A" if "You are A." in prompt else "B"
        seen[wid].append(prompt)
        return await original(prompt, id_slot, n_predict, cache_prompt, **kw)

    fake.complete = spy
    for i in range(3):
        await orc.dispatch(Task("A", f"turn {i}"))
        await orc.dispatch(Task("B", f"turn {i}"))

    for wid, prompts in seen.items():
        assert len(prompts) == 3
        for earlier, later in zip(prompts, prompts[1:]):
            assert later.startswith(earlier), f"{wid} rewrote its prompt instead of appending"
