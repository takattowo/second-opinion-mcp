# second-opinion-mcp

An MCP server with one tool, `ask`: send a prompt to another model and get its answer back. Use it to have a
different model review a diff, check a design, or double-check a fact, from any MCP client (Claude Code, Cursor,
AWS Quick, Claude Desktop, ...).

One Python file, standard library only, no SDKs.

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

```powershell
git clone https://github.com/takattowo/second-opinion-mcp
cd second-opinion-mcp
Copy-Item config.example.json $HOME\.second-opinion.json   # then edit it
```

Python 3.10+. To run the tests: `pip install pytest`, then `python -m pytest`.

## Config

The server reads `SECOND_OPINION_CONFIG`, or `~/.second-opinion.json` when that is unset.

```json
{
  "providers": {
    "azure": {"type": "openai", "base_url": "https://<resource>.openai.azure.com/openai/v1", "api_key_env": "AZURE_FOUNDRY_API_KEY"}
  },
  "models": {
    "luna": {"provider": "azure", "model": "<deployment name>", "params": {"reasoning_effort": "low"}}
  }
}
```

- **providers**: `type`, `base_url`, `api_key_env` (the name of the environment variable that holds the key), and
  optionally `timeout` in seconds (default 300).
- **models**: the alias the client picks, mapped to a `provider` and the `model` id. On Azure the model id is your
  deployment name. `params` is merged into the request body as is, so any provider option works there
  (`temperature`, `reasoning_effort`, ...). Use the option names the provider expects: `anthropic` models take
  `max_tokens` (default 4096), while OpenAI reasoning models reject it and take `max_completion_tokens`.

Keys never go in the config file, only the name of the variable that holds them.

See [config.example.json](config.example.json) for all five providers.

## Add it to your MCP client

```json
"second-opinion": {
  "command": "python",
  "args": ["<repo>/second_opinion.py"],
  "env": {"AZURE_FOUNDRY_API_KEY": "<key>"}
}
```

Use the full path to `python` if the client's PATH doesn't include it. If you install the package
(`pip install .`), the command is `second-opinion-mcp` with no arguments. Restart the client after changing the
config: the model list is read at startup.

## The `ask` tool

| Argument | |
| --- | --- |
| `model` | One of the configured aliases (the client sees them as a list) |
| `prompt` | The question. The other model sees only this, so include the code, diff or context |
| `system` | Optional system prompt, e.g. "You are a strict code reviewer" |

Errors (missing key, HTTP errors with the provider's message, unknown model) come back as tool errors, so the
calling model can read them and fix the call. Redirects are not followed, so the key is only ever sent to
`base_url`.

## Limits

- Text in, text out: no images, tools or streaming.
- `openai` type uses Chat Completions. A model that only supports the Responses API won't work.
- One request at a time; a slow model holds up the next call.
