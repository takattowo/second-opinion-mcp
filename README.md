# second-opinion-mcp

An MCP server that asks another model for a second opinion: have a different model review a diff, check a design,
or double-check a fact, from any MCP client (Claude Code, Cursor, AWS Quick, Claude Desktop, ...).

One Python file, standard library only, no SDKs.

## Tools

| Tool | What it sends |
| --- | --- |
| `ask(model, prompt, system?)` | Only the prompt you give it |
| `review(model, repo, ref?, paths?, focus?)` | `git diff <ref> -- <paths>`, read by the server itself, with a neutral reviewer prompt |

`review` compares against `HEAD` by default: the uncommitted changes to tracked files. Pass a commit or range
(`HEAD~1`, `main...my-branch`) to review committed work. Untracked files are never sent; a new file shows up once it
is staged or committed. `focus` points the reviewer at an area ("error handling") without telling it your verdict.

Every reply comes back wrapped and marked as unverified:

```
<second_opinion model="sol" input_tokens="1840" output_tokens="212">
app.py:2 | bug | add() subtracts instead of adding
</second_opinion>
Unverified second opinion: check each point against the code before acting on it.
```

## What leaves your machine

- `ask` sends only the prompt and system prompt. `review` sends the diff, not whole files.
- `review` leaves out key files (`.env*`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `id_rsa` and the other SSH key names),
  even when you name them in `paths`.
- Before sending, both tools redact your configured API keys and common secret shapes: AWS access keys, private key
  blocks, `sk-` and `ghp_` style tokens, and `.env`-style `NAME_KEY=value` lines. The reply says how many were
  redacted. This catches common patterns only; it is not a secret scanner. A private key in a file with another
  name is only caught when the diff shows its whole `BEGIN`…`END` block.
- A prompt over `max_chars` (default 200,000) is refused instead of sent.
- Redirects are not followed, so the key is only ever sent to `base_url`.
- `review` runs `git diff` in the repo you name, so only point it at repos you trust: a repo's own git config can
  run filters, as it can for any git command there.

## Endpoints

Two wire formats cover most providers:

| `type` | Request | Works with |
| --- | --- | --- |
| `openai` | `POST {base_url}/chat/completions`, `Authorization: Bearer` | OpenAI, Azure Foundry (v1 API), DeepSeek, OpenRouter, Ollama, LM Studio, ... |
| `anthropic` | `POST {base_url}/v1/messages`, `x-api-key`, `anthropic-version: 2023-06-01` | Anthropic, Claude on Azure Foundry |

`base_url` is the same value you would give that vendor's SDK:

| Provider | `type` | `base_url` |
| --- | --- | --- |
| OpenAI | `openai` | `https://api.openai.com/v1` |
| DeepSeek | `openai` | `https://api.deepseek.com` |
| Azure Foundry (GPT and other non-Claude models) | `openai` | `https://<resource>.openai.azure.com/openai/v1` |
| Anthropic | `anthropic` | `https://api.anthropic.com` |
| Azure Foundry (Claude) | `anthropic` | `https://<resource>.services.ai.azure.com/anthropic` |

## Setup

```shell
git clone https://github.com/takattowo/second-opinion-mcp
cd second-opinion-mcp
cp config.example.json ~/.second-opinion.json                # macOS / Linux
Copy-Item config.example.json $HOME\.second-opinion.json     # Windows PowerShell
```

Then edit the config. Python 3.10+, and git for `review`. To run the tests: `pip install pytest`, then
`python -m pytest`.

## Config

The server reads `SECOND_OPINION_CONFIG`, or `~/.second-opinion.json` when that is unset.

```json
{
  "max_chars": 200000,
  "providers": {
    "azure": {"type": "openai", "base_url": "https://<resource>.openai.azure.com/openai/v1", "api_key_env": "AZURE_FOUNDRY_API_KEY"}
  },
  "models": {
    "luna": {"provider": "azure", "model": "<deployment name>", "params": {"reasoning_effort": "low"}}
  }
}
```

- **max_chars**: optional, the largest prompt (plus system prompt) that is sent. Default 200,000.
- **log**: optional, `tokens` (default), `full` or `off`. See [Usage log](#usage-log).
- **log_file**: optional, where the log goes. Default `~/.second-opinion-log.jsonl`; a relative path is next to the
  config file.
- **providers**: `type`, `base_url`, `api_key_env` (the name of the environment variable that holds the key), and
  optionally `timeout` in seconds (default 300).
- **models**: the alias the client picks, mapped to a `provider` and the `model` id. On Azure the model id is your
  deployment name. `params` is merged into the request body as is, so any provider option works there
  (`temperature`, `reasoning_effort`, ...). Use the option names the provider expects: `anthropic` models take
  `max_tokens` (default 4096), while OpenAI reasoning models reject it and take `max_completion_tokens`.

Keys never go in the config file, only the name of the variable that holds them.

See [config.example.json](config.example.json) for all five providers.

## Add it to your MCP client

Claude Code:

```shell
claude mcp add --transport stdio --env AZURE_FOUNDRY_API_KEY=<key> second-opinion -- python /path/to/second_opinion.py
```

Other clients:

```json
"second-opinion": {
  "command": "python",
  "args": ["/path/to/second_opinion.py"],
  "env": {"AZURE_FOUNDRY_API_KEY": "<key>"}
}
```

Use the full path to `python` if the client's PATH doesn't include it. If you install the package
(`pip install .`), the command is `second-opinion-mcp` with no arguments. Restart the client after changing the
config: the model list is read at startup.

Errors (missing key, HTTP errors with the provider's message, unknown model, nothing to review) come back as tool
errors, so the calling model can read them and fix the call.

## Usage log

Each tool call appends one JSON line to `log_file`, failed calls included:

```json
{"time": "2026-01-05T14:03:11+07:00", "tool": "review", "model": "sol", "repo": "/path/to/repo", "ref": "HEAD",
 "paths": null, "model_id": "gpt-6.2-sol", "input_tokens": 1840, "output_tokens": 212, "redacted": 0, "ok": true,
 "seconds": 17.7}
```

| `log` | What is saved |
| --- | --- |
| `tokens` (default) | Time, tool, model, tokens, duration, ok or the error, how many secrets were redacted, and for `review` the repo, ref and paths |
| `full` | All of that, plus the prompt and system prompt as sent (after redaction) and the reply |
| `off` | Nothing |

`full` stores the code you sent, so keep the log out of git and anywhere shared.

## Limits

- Text in, text out: no images, tools or streaming.
- `openai` type uses Chat Completions. A model that only supports the Responses API won't work.
- Rate limits and server errors (429, 5xx) are retried once, after the `Retry-After` the server asks for (at most
  30 seconds).
