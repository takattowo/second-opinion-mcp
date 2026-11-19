"""second-opinion-mcp: an MCP server that asks another model for a second opinion.

Two wire formats cover the common endpoints:
  openai     POST {base_url}/chat/completions  (OpenAI, Azure Foundry v1, DeepSeek, OpenRouter, Ollama, ...)
  anthropic  POST {base_url}/v1/messages       (Anthropic, Claude on Azure Foundry)
"""
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

__version__ = "0.2.0"
TYPES = ("openai", "anthropic")
PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")  # newest first
DEFAULT_CONFIG = Path.home() / ".second-opinion.json"
DEFAULT_MAX_CHARS = 200_000
UNVERIFIED = "Unverified second opinion: check each point against the code before acting on it."
REVIEW_SYSTEM = ("You are reviewing a code change you have not seen before. Look for real bugs, security problems and "
                 "missed edge cases; ignore style unless it hides a bug. Report each finding on its own line as "
                 "`file:line | severity | issue`, with severity blocker, bug or minor. Say when a finding depends on "
                 "code outside the diff. If you find nothing, reply `no findings`.")
# never sent by review, even when named in paths
EXCLUDE = tuple(f":(exclude,glob)**/{p}" for p in (".env*", "*.pem", "*.key", "*.p12", "*.pfx",
                                                  "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*"))
# shortcut: common secret shapes only, not a full scanner; add patterns as new key formats turn up
SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
     "[REDACTED PRIVATE KEY]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED]"),
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{20,}"), "[REDACTED]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "[REDACTED]"),
    # .env-style NAME_KEY=value lines (diff +/- prefix allowed). The value runs to a quote, space or line end;
    # a value that looks like code (a call or an index) is kept, as is one an earlier pattern already redacted.
    (re.compile(r"(?m)^([-+ ]?\s*(?:export\s+)?[A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD)[A-Z0-9_]*"
                r"\s*[=:]\s*[\"']?)(?!\[REDACTED|[A-Za-z_][\w.]*[(\[])[^\s\"']{8,}"), r"\1[REDACTED]"),
]


def load_config(path=None):
    path = Path(path or os.environ.get("SECOND_OPINION_CONFIG") or DEFAULT_CONFIG)
    config = json.loads(path.read_text(encoding="utf-8"))
    if not (isinstance(config, dict) and isinstance(config.get("providers"), dict)
            and isinstance(config.get("models"), dict)):
        raise ValueError('config needs "providers" and "models" objects')
    for name, provider in config["providers"].items():
        missing = [repr(k) for k in ("type", "base_url", "api_key_env") if k not in provider]
        if missing:
            raise ValueError(f"provider {name!r}: missing {', '.join(missing)}")
        if provider["type"] not in TYPES:
            raise ValueError(f"provider {name!r}: type {provider['type']!r} must be one of {TYPES}")
    if not config["models"]:
        raise ValueError("no models configured")
    for alias, model in config["models"].items():
        missing = [repr(k) for k in ("provider", "model") if k not in model]
        if missing:
            raise ValueError(f"model {alias!r}: missing {', '.join(missing)}")
        if model["provider"] not in config["providers"]:
            raise ValueError(f"model {alias!r}: unknown provider {model['provider']!r}")
    max_chars = config.get("max_chars", DEFAULT_MAX_CHARS)
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError(f"max_chars must be a positive integer, got {max_chars!r}")
    return config


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None  # following it would resend the API key to wherever it points


_opener = urllib.request.build_opener(_NoRedirect)


def _retry_delay(value):
    try:
        delay = float(value)
    except (TypeError, ValueError):
        return 1  # no Retry-After, or an HTTP date
    return min(max(delay, 0), 30) if math.isfinite(delay) else 1


def _post(url, headers, body, timeout):
    headers = {"content-type": "application/json", "user-agent": f"second-opinion-mcp/{__version__}", **headers}
    request = urllib.request.Request(url, json.dumps(body).encode(), headers)
    for attempt in (1, 2):
        try:
            with _opener.open(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            if attempt == 1 and (e.code == 429 or e.code >= 500):
                time.sleep(_retry_delay(e.headers.get("retry-after")))
                continue
            raise RuntimeError(f"HTTP {e.code} from {url}: {e.read(2000).decode(errors='replace')}") from None


def ask(config, alias, prompt, system=None):
    """Send one prompt to the model configured as `alias`; return (text, {input_tokens, output_tokens})."""
    if alias not in config["models"]:
        raise ValueError(f"unknown model {alias!r}; configured: {', '.join(config['models'])}")
    model = config["models"][alias]
    name = model["provider"]
    provider = config["providers"][name]
    key = os.environ.get(provider["api_key_env"])
    if not key:
        raise RuntimeError(f"environment variable {provider['api_key_env']} is not set (provider {name!r})")
    base = provider["base_url"].rstrip("/")
    timeout = provider.get("timeout", 300)
    messages = [{"role": "user", "content": prompt}]

    if provider["type"] == "anthropic":
        body = {"model": model["model"], "max_tokens": 4096, **model.get("params", {}), "messages": messages}
        if system:
            body["system"] = system
        reply = _post(base + "/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"}, body, timeout)
        usage = reply.get("usage") or {}
        text = "".join(block["text"] for block in reply["content"] if block.get("type") == "text")
        return text, {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}

    if system:
        messages.insert(0, {"role": "system", "content": system})
    body = {"model": model["model"], **model.get("params", {}), "messages": messages}
    reply = _post(base + "/chat/completions", {"authorization": f"Bearer {key}"}, body, timeout)
    usage = reply.get("usage") or {}
    text = reply["choices"][0]["message"]["content"] or ""
    return text, {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}


def scrub(config, text):
    """Redact the configured API keys and common secret shapes; return (text, number redacted)."""
    count = 0
    for provider in config["providers"].values():
        key = os.environ.get(provider["api_key_env"]) or ""
        if len(key) >= 8 and key in text:
            count += text.count(key)
            text = text.replace(key, "[REDACTED]")
    for pattern, replacement in SECRET_PATTERNS:
        text, n = pattern.subn(replacement, text)
        count += n
    return text, count


def review_prompt(args):
    """Build the review prompt from `git diff <ref> -- <paths>` in the repo: tracked files only."""
    repo, ref, paths = args.get("repo"), args.get("ref") or "HEAD", args.get("paths") or ["."]
    if not isinstance(paths, list) or any(str(p).startswith("-") for p in [ref, *paths]):
        raise ValueError("ref and paths can't start with '-', and paths must be a list")
    out = subprocess.run(["git", "-C", str(repo), "diff", "--no-color", "--no-ext-diff", "--no-textconv",
                          "--submodule=short", str(ref), "--", *map(str, paths), *EXCLUDE],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if out.returncode:
        raise RuntimeError(f"git diff failed: {out.stderr.strip()[:1000]}")
    if not out.stdout.strip():
        raise ValueError(f"no tracked changes for git diff {ref}; new files show up once staged or committed")
    focus = args.get("focus")
    return (f"Review this change (git diff {ref} -- {' '.join(map(str, paths))}).\n"
            + (f"Focus: {focus}\n" if focus else "") + f"\n```diff\n{out.stdout}```")


def call_tool(config, name, args):
    alias = args.get("model")
    if name == "review":
        prompt, system = review_prompt(args), REVIEW_SYSTEM
    else:
        prompt, system = args.get("prompt", ""), args.get("system")
    prompt, redacted = scrub(config, prompt)
    if system:
        system, more = scrub(config, system)
        redacted += more
    size, limit = len(prompt) + len(system or ""), config.get("max_chars", DEFAULT_MAX_CHARS)
    if size > limit:
        raise ValueError(f"prompt is {size} chars, over max_chars ({limit}): send a smaller slice, "
                         "or raise max_chars in the config")
    text, usage = ask(config, alias, prompt, system)
    text = re.sub(r"(?i)</second_opinion", "&lt;/second_opinion", text)  # the reply can't end the wrapper early
    tokens = "".join(f' {k}="{v}"' for k, v in usage.items() if v is not None)
    note = UNVERIFIED + (f" {redacted} secret(s) were redacted before sending." if redacted else "")
    return f'<second_opinion model="{alias}"{tokens}>\n{text}\n</second_opinion>\n{note}'


def tools(config):
    model = {"type": "string", "enum": list(config["models"]), "description": "Configured model alias"}
    annotations = {"readOnlyHint": True, "openWorldHint": True}
    return [{
        "name": "ask",
        "description": ("Ask another AI model for a second opinion: a design check, a fact check, or code you paste in. "
                        "It sees only what you send. Ask neutrally: don't share your own conclusion, and for code "
                        "ask for `file:line | severity | issue` lines. Secrets are redacted before sending. "
                        "The reply is unverified: check each point before acting on it."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": model,
                "prompt": {"type": "string", "description": "The question, with all the context it needs"},
                "system": {"type": "string", "description": "Optional system prompt, e.g. a reviewer role"},
            },
            "required": ["model", "prompt"],
        },
        "annotations": annotations,
    }, {
        "name": "review",
        "description": ("Have another AI model review a code change. The server runs `git diff <ref> -- <paths>` in "
                        "the repo itself, so don't paste code: tracked files only, key files (.env*, *.pem, *.key, "
                        "id_rsa, ...) are left out and secrets are redacted. Findings come back as "
                        "`file:line | severity | issue` lines "
                        "and are unverified: check each against the code before acting on it."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": model,
                "repo": {"type": "string", "description": "Absolute path of the git repository"},
                "ref": {"type": "string", "description": ("What git diff compares against: a commit or range such as "
                                                          "HEAD~1 or main...my-branch. Default HEAD: uncommitted "
                                                          "changes to tracked files")},
                "paths": {"type": "array", "items": {"type": "string"},
                          "description": "Only diff these paths, relative to the repo"},
                "focus": {"type": "string", "description": ("Optional area to look at, e.g. 'error handling'. "
                                                            "Don't include your own verdict")},
            },
            "required": ["model", "repo"],
        },
        "annotations": annotations,
    }]


def _error(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(config, request):
    """Answer one JSON-RPC request; notifications (no id) get None."""
    if "id" not in request:
        return None
    method, params = request.get("method"), request.get("params") or {}
    if not isinstance(params, dict):
        return _error(request["id"], -32602, "params must be an object")
    if method == "initialize":
        asked = params.get("protocolVersion")
        result = {"protocolVersion": asked if asked in PROTOCOLS else PROTOCOLS[0],
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "second-opinion-mcp", "version": __version__}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": tools(config)}
    elif method == "tools/call" and params.get("name") in ("ask", "review"):
        try:
            text, failed = call_tool(config, params["name"], params.get("arguments") or {}), False
        except Exception as e:  # any failure goes back to the model as a tool error, not a crash
            text, failed = f"{type(e).__name__}: {e}", True
        result = {"content": [{"type": "text", "text": text}], "isError": failed}
    elif method == "tools/call":
        return _error(request["id"], -32602, f"unknown tool: {params.get('name')!r}")
    else:
        return _error(request["id"], -32601, f"method not found: {method}")
    return {"jsonrpc": "2.0", "id": request["id"], "result": result}


def main():
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    try:
        config = load_config()
    except Exception as e:
        sys.exit(f"second-opinion-mcp: bad config: {type(e).__name__}: {e}")
    lock = threading.Lock()

    def send(response):
        if response:
            with lock:
                print(json.dumps(response), flush=True)

    def respond(request):
        try:
            send(handle(config, request))
        except Exception as e:  # every request gets an answer, even when handling it hits a bug
            send(_error(request.get("id"), -32603, f"internal error: {type(e).__name__}: {e}"))

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except ValueError:
            send(_error(None, -32700, "parse error"))
            continue
        if not isinstance(request, dict):
            send(_error(None, -32600, "invalid request"))
        elif request.get("method") == "tools/call":
            # model calls are slow, so each gets its own thread; Python waits for them before exiting
            threading.Thread(target=respond, args=(request,)).start()
        else:
            respond(request)


if __name__ == "__main__":
    main()
