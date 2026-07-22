#!/usr/bin/env python3
"""bonsai_client.py -- minimal, dependency-free client for the local Bonsai 27B server.

Encodes the two things callers get wrong (see bench-results.md / bonsai.sh):
  * thinking mode -> always sends chat_template_kwargs.enable_thinking=false (the model
    otherwise spends its whole budget on <think> and returns empty content).
  * <think> leakage -> strips any residual reasoning block from the reply.

Stdlib only. Assumes the server is up (`bash bonsai.sh start`) or call .ensure_up().

    from bonsai_client import Bonsai
    b = Bonsai()
    b.ensure_up()                                   # starts the server if needed
    print(b.complete("Summarize STDP in one sentence."))
    print(b.chat([{"role": "user", "content": "hi"}]))
    print(b.complete("Explain your reasoning step by step.", think=True))  # opt back into thinking

Self-check:  python bonsai_client.py
"""
import json, os, re, subprocess, time, urllib.request

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class Bonsai:
    def __init__(self, base_url=None, model="bonsai-27b", timeout=900):
        self.base = (base_url or os.environ.get("BONSAI_URL", "http://127.0.0.1:8080")).rstrip("/")
        self.model = model
        self.timeout = timeout

    def _post(self, path, payload, retries=3):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.base + path, data=data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(3)

    def chat(self, messages, temperature=0.4, top_p=0.95, max_tokens=1024, think=False, **extra):
        """OpenAI-style chat. Returns the reply text (reasoning stripped). think=False by default."""
        payload = {"model": self.model, "messages": messages, "temperature": temperature,
                   "top_p": top_p, "max_tokens": max_tokens,
                   "chat_template_kwargs": {"enable_thinking": bool(think)}, **extra}
        d = self._post("/v1/chat/completions", payload)
        return _THINK.sub("", d["choices"][0]["message"]["content"]).strip()

    def complete(self, prompt, system=None, **kw):
        msgs = ([{"role": "system", "content": system}] if system else [])
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, **kw)

    def healthy(self):
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def ensure_up(self, wait_s=180):
        """Start the server via bonsai.sh if it isn't already answering."""
        if self.healthy():
            return True
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bonsai.sh")
        subprocess.run(["bash", script, "start"], check=True)
        for _ in range(wait_s // 2):
            if self.healthy():
                return True
            time.sleep(2)
        raise RuntimeError("Bonsai did not become ready; see bench-logs/bonsai-server.log")


if __name__ == "__main__":
    b = Bonsai()
    assert b.healthy(), "Bonsai not up -- run: bash bonsai.sh start"
    out = b.complete("Reply with exactly: OK")
    print("response:", repr(out))
    assert out.strip(), "empty response -- thinking may not be disabled"
    print("bonsai_client OK")
