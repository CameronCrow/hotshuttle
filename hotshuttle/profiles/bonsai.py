"""BonsaiProfile -- every Bonsai-specific number, path, and template quirk.

This is the only module that is allowed to know what model we are running. It is data,
not logic: adding a second model means writing another file like this one, not touching
core/ (docs/PLAN.md section 5, issue #7).

All sizing constants below are MEASURED, not computed -- see bench-results.md
"Correction" and experiments/m0-results.md, m1-results.md.
"""
from __future__ import annotations

import os

# --- measured constants ------------------------------------------------------

# Fixed Gated DeltaNet recurrent state per slot. llama.cpp reports it as
# "created context checkpoint ... size = 149.626 MiB", independent of token count, and a
# saved blob for a 5-token prompt weighs 149.8 MiB. f32 and NOT quantizable: decoding is
# recurrent, so quantization error accumulates down the sequence instead of staying local
# to its token. At these context lengths this floor, not the KV cache, is the dominant
# per-slot cost -- 48 of Bonsai's 64 layers are DeltaNet.
RECURRENT_STATE_BYTES = int(149.626 * 2**20)

# Attention KV for the other 16 layers. Measured by differencing saved blobs: a
# 2611-token worker saves at 236.5 MiB (q8_0) / 195.7 MiB (q4_0) less the floor above.
KV_BYTES_PER_TOKEN = {"q8_0": int(34.1 * 1024), "q4_0": int(18.1 * 1024)}

# 12288 @ q8_0 measures 30.2 t/s and leaves ~845 MiB of VRAM for the Windows desktop.
# 16384 also runs at full speed but needs the desktop under ~680 MiB, and going over
# that is SILENT: the server still starts and answers, roughly 4x slower, reporting a
# similar nvidia-smi memory.used. Hence the conservative default.
DEFAULT_CTX = 12288
DEFAULT_KV_QUANT = "q8_0"


class QwenChatML:
    """Bonsai is a Qwen3.6 build, so ChatML -- with thinking suppressed.

    The empty <think></think> pair is not decoration: it is exactly what the server's
    Jinja template emits under enable_thinking=false, verified byte-for-byte against
    /apply-template in M0 (experiments/m0_template_ground_truth.txt). Without it Bonsai
    spends its whole token budget reasoning and returns empty content.

    These bytes must not drift from the server's template or the prefix cache misses on
    every single turn -- tests/test_template.py pins them, and the gpu-marked test
    re-checks against the live server.
    """

    def system(self, text: str) -> str:
        return f"<|im_start|>system\n{text}<|im_end|>\n"

    def user(self, text: str) -> str:
        return f"<|im_start|>user\n{text}<|im_end|>\n"

    def assistant_open(self) -> str:
        return "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def assistant_close(self) -> str:
        return "<|im_end|>\n"


class BonsaiProfile:
    name = "bonsai-27b"

    def __init__(self, ctx_per_slot: int = DEFAULT_CTX, n_slots: int = 1,
                 kv_quant: str = DEFAULT_KV_QUANT, server_url: str | None = None,
                 slot_save_path: str | None = None):
        if kv_quant not in KV_BYTES_PER_TOKEN:
            raise ValueError(f"kv_quant must be one of {sorted(KV_BYTES_PER_TOKEN)}")
        self.ctx_per_slot = ctx_per_slot
        self.n_slots = n_slots
        self.kv_quant = kv_quant
        self.server_url = server_url or os.environ.get("BONSAI_URL", "http://127.0.0.1:8080")
        # Paths come from the environment, never hardcoded here (issue #1).
        base = os.environ.get("BONSAI_DIR", os.path.expanduser("~/Projects/bonsai"))
        self.slot_save_path = slot_save_path or os.environ.get(
            "BONSAI_SLOT_DIR", os.path.join(base, "bench-logs", "slots"))
        self.per_slot_floor_bytes = RECURRENT_STATE_BYTES
        self.kv_bytes_per_token = KV_BYTES_PER_TOKEN[kv_quant]
        # Workers want repeatable output, not the model card's creative defaults
        # (temp 0.7 / top-p 0.95, which bonsai.sh still passes for chat callers).
        self.sampling_defaults = {"temperature": 0.2, "top_p": 0.9}
        self.template = QwenChatML()

    def slot_bytes(self, n_tokens: int) -> int:
        """VRAM a single hot worker costs at a given fill level."""
        return self.per_slot_floor_bytes + n_tokens * self.kv_bytes_per_token

    def launch_argv(self, server: str, model: str, port: int = 8080) -> list[str]:
        """The llama-server command line this profile implies.

        --slots and --slot-save-path are mandatory for the orchestration layer: without
        the save path the save/restore endpoints return 200 and do nothing.
        """
        return [server, "-m", model, "--alias", self.name,
                "--host", "127.0.0.1", "--port", str(port),
                "-ngl", "99", "-c", str(self.ctx_per_slot * self.n_slots),
                "--parallel", str(self.n_slots), "-fa", "1",
                "--cache-type-k", self.kv_quant, "--cache-type-v", self.kv_quant,
                "--slots", "--slot-save-path", self.slot_save_path,
                "--jinja"]
