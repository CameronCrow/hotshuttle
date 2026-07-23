"""The model seam.

Nothing in core/ may import a concrete profile or hardcode a model's constants; a profile
supplies them as data. `tests/test_seam.py` greps core/ to enforce that, so extracting a
second model later is a move rather than surgery (docs/PLAN.md section 5, issue #7).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatTemplate(Protocol):
    """The model's prompt bytes. Bytes are load-bearing -- see invariant I3.

    The compiler composes these into seeds and turns; a profile only has to say what a
    system block, a user block, and an assistant handoff look like for its model.
    """

    def system(self, text: str) -> str: ...
    def user(self, text: str) -> str: ...
    def assistant_open(self) -> str: ...
    def assistant_close(self) -> str: ...

    def to_plain(self, rendered: str) -> str:
        """Rendered prompt bytes back to readable text, control tokens removed.

        Needed whenever a transcript becomes *content* for another prompt -- summarizing
        for a compaction, or logging. Feeding raw template bytes back in nests one
        conversation inside another and the model answers it instead of reading it.
        """
        ...


@runtime_checkable
class ModelProfile(Protocol):
    name: str
    server_url: str
    n_slots: int
    ctx_per_slot: int
    slot_save_path: str
    per_slot_floor_bytes: int      # fixed recurrent state; 0 for a dense model
    kv_bytes_per_token: int
    sampling_defaults: dict
    template: ChatTemplate

    def launch_argv(self, server: str, model: str) -> list[str]: ...
