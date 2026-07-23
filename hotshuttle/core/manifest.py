"""workers.yaml -- the operator's contract.

Declarative worker roles plus the runtime binding for the model serving them. Validation
fails loud, including on unknown keys: a typo'd `compact_at` that silently keeps the
default is exactly the kind of thing you discover three hours into a run.

The manifest deliberately does NOT construct a profile. Which profile class to build is
model knowledge, and core/ does not have any (docs/PLAN.md §5) -- `model_kwargs()` hands
the caller the arguments and the caller picks the class.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

from .worker import Role

MODEL_KEYS = {"server", "n_slots", "slot_save_path", "ctx_per_slot", "kv_quant",
              "sampling_defaults"}
ROLE_KEYS = {"name", "system", "tools", "ctx_budget", "max_out", "compact_at", "sampling"}


class ManifestError(ValueError):
    """The manifest is wrong. Say precisely how."""


@dataclass
class Manifest:
    model: dict
    roles: dict[str, Role]

    def model_kwargs(self) -> dict:
        """Arguments for a ModelProfile constructor."""
        kw = {k: v for k, v in self.model.items() if k != "server"}
        if "server" in self.model:
            kw["server_url"] = self.model["server"]
        return kw


def load(path: str | pathlib.Path) -> Manifest:
    try:
        import yaml
    except ImportError as e:                        # the only non-stdlib import in the package
        raise ManifestError(
            "reading a manifest needs PyYAML (pip install pyyaml). Everything else in "
            "hotshuttle is stdlib-only -- build Role objects directly to avoid it.") from e

    path = pathlib.Path(path)
    if not path.exists():
        raise ManifestError(f"no manifest at {path}")
    return parse(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, source=str(path))


def parse(data: dict, source: str = "<manifest>") -> Manifest:
    if not isinstance(data, dict):
        raise ManifestError(f"{source}: expected a mapping at the top level")

    unknown = set(data) - {"model", "roles"}
    if unknown:
        raise ManifestError(f"{source}: unknown top-level keys {sorted(unknown)}")

    model = data.get("model") or {}
    if not isinstance(model, dict):
        raise ManifestError(f"{source}: `model` must be a mapping")
    unknown = set(model) - MODEL_KEYS
    if unknown:
        raise ManifestError(f"{source}: unknown model keys {sorted(unknown)} "
                            f"(known: {sorted(MODEL_KEYS)})")

    ctx_per_slot = model.get("ctx_per_slot")
    raw_roles = data.get("roles") or []
    if not isinstance(raw_roles, list) or not raw_roles:
        raise ManifestError(f"{source}: `roles` must be a non-empty list")

    roles: dict[str, Role] = {}
    for i, r in enumerate(raw_roles):
        if not isinstance(r, dict):
            raise ManifestError(f"{source}: roles[{i}] must be a mapping")
        unknown = set(r) - ROLE_KEYS
        if unknown:
            raise ManifestError(f"{source}: roles[{i}] has unknown keys {sorted(unknown)} "
                                f"(known: {sorted(ROLE_KEYS)})")
        for required in ("name", "system", "ctx_budget"):
            if required not in r:
                raise ManifestError(f"{source}: roles[{i}] is missing `{required}`")
        name = r["name"]
        if name in roles:
            raise ManifestError(f"{source}: duplicate role name {name!r}")
        if ctx_per_slot is not None and r["ctx_budget"] > ctx_per_slot:
            raise ManifestError(
                f"{source}: role {name!r} has ctx_budget {r['ctx_budget']} > "
                f"ctx_per_slot {ctx_per_slot}; it could never fit in a slot")
        try:
            roles[name] = Role(
                name=name, system=r["system"], ctx_budget=r["ctx_budget"],
                tools=tuple(r.get("tools") or ()),
                max_out=r.get("max_out", 1024),
                compact_at=r.get("compact_at", 0.8),
                sampling=dict(r.get("sampling") or {}))
        except ValueError as e:                     # Role's own __post_init__ checks
            raise ManifestError(f"{source}: role {name!r}: {e}") from e

    return Manifest(model=model, roles=roles)
