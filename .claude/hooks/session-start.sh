#!/bin/bash
# SessionStart hook (per-repo STUB -- identical in every repo).
#
# In a cloud/remote session it fetches the workforce marketplace (which is where
# the real logic + the plugin ledger live) and runs its canonical bootstrap, so
# this sandbox gets Cameron's whole skill/plugin set. Keeping the logic in
# workforce means this stub almost never changes -- update the ledger or the
# bootstrap there and every repo picks it up on its next cloud session.
#
# Local machines skip this entirely (they install skills via the workforce
# junction -- see workforce/INSTALL.md). Do not put repo-specific logic here;
# this file is deployed verbatim by workforce/plugins/cloud/deploy-cloud-hooks.py.
set -uo pipefail

# Skip only when we're clearly on a machine that already has the workforce skills
# installed locally (the junction install puts every skill in ~/.claude/skills).
# Gating on CLAUDE_CODE_REMOTE alone was fragile: if that variable isn't set in a
# cloud sandbox the hook silently no-ops and nothing gets installed. Checking for
# the local install instead is correct in both directions.
if [ -z "${CLAUDE_CODE_REMOTE:-}" ] && [ -d "$HOME/.claude/skills/git-safe-ops" ]; then
  exit 0
fi

command -v claude >/dev/null 2>&1 || { echo "[cloud-bootstrap] no claude CLI on PATH; skip"; exit 0; }
echo "[cloud-bootstrap] starting (remote=${CLAUDE_CODE_REMOTE:-unset})"

claude plugin marketplace add CameronCrow/workforce --scope user >/dev/null 2>&1 || true

BOOT="$HOME/.claude/plugins/marketplaces/workforce/plugins/cloud/bootstrap.sh"
if [ -f "$BOOT" ]; then
  bash "$BOOT" || true
else
  echo "[cloud-bootstrap] bootstrap.sh not found after marketplace add -- workforce fetch failed?"
fi
