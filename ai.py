from __future__ import annotations

import os
import time
from urllib.parse import urlparse

import requests

# ── OpenCode (`opencode serve`) ───────────────────────────────────────────────
# OpenCode's server is NOT OpenAI-compatible: it is a session-based REST API.
# One prompt = create session → prompt_async → poll messages → delete session.
OPENCODE_DEFAULT_URL = "http://127.0.0.1:4096"
OPENCODE_USER        = "opencode"   # basic-auth user when OPENCODE_SERVER_PASSWORD is set
OPENCODE_TIMEOUT     = 120          # seconds to wait for a completion
# Fallback if /experimental/tool/ids is unavailable. The server's own list is
# preferred — it also covers MCP/plugin tools this static list can't know about.
_OPENCODE_TOOLS = ["invalid", "question", "bash", "read", "glob", "grep", "edit", "write",
                   "task", "webfetch", "todowrite", "websearch", "skill", "apply_patch"]


class OpenCodeError(RuntimeError):
    pass


def _oc_session(credential: dict) -> tuple[requests.Session, str]:
    base = (credential.get("base_url") or OPENCODE_DEFAULT_URL).strip().rstrip("/")
    u = urlparse(base)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise OpenCodeError("OpenCode server URL must start with http:// or https://")
    s = requests.Session()
    if credential.get("password"):
        s.auth = (credential.get("username") or OPENCODE_USER, credential["password"])
    return s, base


def _oc_call(s: requests.Session, base: str, method: str, path: str, **kw):
    # Error text reaches the browser, and the URL is user-supplied: never echo the
    # target URL, the connection error, or the response body (SSRF information leak).
    # Redirects are refused so a server can't bounce the request (and the password)
    # on to another host.
    try:
        r = s.request(method, base + path, timeout=15, allow_redirects=False, **kw)
    except requests.RequestException as exc:
        raise OpenCodeError("Cannot reach the OpenCode server — check the URL and that `opencode serve` is running") from exc
    if r.status_code == 401:
        raise OpenCodeError("OpenCode server rejected the credentials (401) — check the password in Vault")
    if r.is_redirect or not r.ok:
        raise OpenCodeError(f"OpenCode {method} {path} failed (HTTP {r.status_code}) — is this an OpenCode server?")
    if not r.content:
        return None
    try:
        return r.json()
    except ValueError as exc:
        raise OpenCodeError("Unexpected response — is this URL an OpenCode server?") from exc


def list_opencode_models(credential: dict) -> list[dict]:
    """Models from providers the server has credentials for, as 'provider/model' ids."""
    s, base = _oc_session(credential)
    data = _oc_call(s, base, "GET", "/config/providers")
    out = []
    for p in data.get("providers", []):
        for mid, m in (p.get("models") or {}).items():
            out.append({"id": f"{p['id']}/{mid}", "provider": p.get("name") or p["id"],
                        "name": m.get("name") or mid})
    return sorted(out, key=lambda m: m["id"])


def _opencode_prompt(prompt: str, credential: dict, timeout: int = OPENCODE_TIMEOUT) -> str:
    model = credential.get("model") or ""
    if "/" not in model:
        raise OpenCodeError("OpenCode model must be 'provider/model' (e.g. opencode/big-pickle) — pick one in Vault")
    provider_id, model_id = model.split("/", 1)

    s, base = _oc_session(credential)
    try:
        tool_ids = _oc_call(s, base, "GET", "/experimental/tool/ids")
    except OpenCodeError:
        tool_ids = _OPENCODE_TOOLS
    # Reddit text is untrusted and goes straight into this prompt: no tool may run.
    tools = {t: False for t in {*tool_ids, *_OPENCODE_TOOLS}}

    # A fixed title stops OpenCode spending an extra LLM call auto-titling the session.
    sid = _oc_call(s, base, "POST", "/session", json={"title": "redsignal"})["id"]
    try:
        _oc_call(s, base, "POST", f"/session/{sid}/prompt_async", json={
            "model":  {"providerID": provider_id, "modelID": model_id},
            "system": "You are a text-only assistant. Reply with plain text only.",
            "tools":  tools,
            "parts":  [{"type": "text", "text": prompt}],
        })
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.6)
            # The server retries failures (rate limit, free tier) silently and forever.
            st = (_oc_call(s, base, "GET", "/session/status") or {}).get(sid) or {}
            if st.get("type") == "retry":
                raise OpenCodeError(st.get("message") or "OpenCode is retrying a failed request")
            for m in _oc_call(s, base, "GET", f"/session/{sid}/message"):
                if m["info"].get("role") != "assistant":
                    continue
                err = m["info"].get("error")
                if err:
                    raise OpenCodeError((err.get("data") or {}).get("message") or err.get("name") or "OpenCode error")
                parts = m.get("parts", [])
                if any(p["type"] == "tool" for p in parts):
                    raise OpenCodeError("OpenCode model attempted a tool call; refusing (tools are disabled)")
                if any(p["type"] == "step-finish" for p in parts):
                    text = "".join(p.get("text", "") for p in parts
                                   if p["type"] == "text" and not p.get("synthetic")).strip()
                    if not text:
                        raise OpenCodeError("OpenCode returned an empty response")
                    return text
        raise OpenCodeError(f"OpenCode did not respond within {timeout}s")
    finally:
        for method, path in (("POST", f"/session/{sid}/abort"), ("DELETE", f"/session/{sid}")):
            try:
                _oc_call(s, base, method, path)
            except OpenCodeError:
                pass


def run_ai_prompt(prompt: str, credential: dict, max_tokens: int = 300) -> str:
    ctype = credential.get("type", "anthropic")

    if ctype == "opencode":
        # OpenCode has no max_tokens knob on the prompt API; the arg is ignored.
        return _opencode_prompt(prompt, credential)

    if ctype == "anthropic":
        import anthropic  # type: ignore
        model  = credential.get("model") or "claude-haiku-4-5-20251001"
        client = anthropic.Anthropic(api_key=credential["api_key"])
        msg    = client.messages.create(model=model, max_tokens=max_tokens,
                                        messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()

    if ctype in ("openai", "ollama"):
        from openai import OpenAI  # type: ignore
        if ctype == "ollama":
            base   = (credential.get("base_url") or "http://localhost:11434").rstrip("/")
            client = OpenAI(api_key="ollama", base_url=f"{base}/v1/")
            model  = credential.get("model") or "llama3.2"
        else:
            client = OpenAI(api_key=credential["api_key"])
            model  = credential.get("model") or "gpt-4o-mini"
        resp = client.chat.completions.create(model=model, max_tokens=max_tokens,
                                              messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()

    raise ValueError(f"Unknown credential type: {ctype}")


def resolve_ai_cred(cfg: dict, credential_id: str | None) -> dict | None:
    if credential_id:
        return next((c for c in cfg.get("credentials", []) if c.get("id") == credential_id), None)
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if key:
        return {"type": "anthropic", "api_key": key}
    return None
