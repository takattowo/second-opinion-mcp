import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import second_opinion

OK = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 11, "completion_tokens": 3}}


@pytest.fixture
def fake_api():
    """Local HTTP server that records each request. It answers from `queue` first, then with `status` and `reply`."""
    state = {"requests": [], "status": 200, "reply": {}, "headers": {}, "queue": [], "barrier": None}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            state["requests"].append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                      "body": json.loads(body)})
            if state["barrier"]:
                state["barrier"].wait()
            status, headers, reply = (state["queue"].pop(0) if state["queue"]
                                      else (state["status"], state["headers"], state["reply"]))
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()


def config_for(url, type_, base_path):
    return {
        "providers": {"p": {"type": type_, "base_url": url + base_path, "api_key_env": "TEST_KEY"}},
        "models": {"m": {"provider": "p", "model": "model-x", "params": {"temperature": 0.2}}},
    }


def call(config, name, arguments):
    return second_opinion.handle(config, {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                          "params": {"name": name, "arguments": arguments}})["result"]


def text_of(result):
    return result["content"][0]["text"]


def sent_prompt(fake_api):
    return fake_api["requests"][0]["body"]["messages"][-1]["content"]


def test_openai_style_request_and_reply(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    fake_api["reply"] = {"choices": [{"message": {"content": "looks fine"}}],
                         "usage": {"prompt_tokens": 11, "completion_tokens": 3}}

    text, usage = second_opinion.ask(config_for(fake_api["url"], "openai", "/openai/v1/"), "m", "review this", "be strict")

    assert text == "looks fine"
    assert usage == {"input_tokens": 11, "output_tokens": 3}
    req = fake_api["requests"][0]
    assert req["path"] == "/openai/v1/chat/completions"
    assert req["headers"]["authorization"] == "Bearer" + " sk-test"
    assert req["body"] == {
        "model": "model-x",
        "temperature": 0.2,
        "messages": [{"role": "system", "content": "be strict"}, {"role": "user", "content": "review this"}],
    }


def test_anthropic_style_request_and_reply(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-ant")
    fake_api["reply"] = {"content": [{"type": "thinking", "thinking": "hmm"},
                                     {"type": "text", "text": "one bug"}, {"type": "text", "text": " on line 3"}],
                         "usage": {"input_tokens": 20, "output_tokens": 5}}

    text, usage = second_opinion.ask(config_for(fake_api["url"], "anthropic", "/anthropic"), "m", "review this", "be strict")

    assert text == "one bug on line 3"
    assert usage == {"input_tokens": 20, "output_tokens": 5}
    req = fake_api["requests"][0]
    assert req["path"] == "/anthropic/v1/messages"
    assert req["headers"]["x-api-key"] == "sk-ant"
    assert req["headers"]["anthropic-version"] == "2023-06-01"
    assert req["body"] == {
        "model": "model-x",
        "max_tokens": 4096,
        "temperature": 0.2,
        "system": "be strict",
        "messages": [{"role": "user", "content": "review this"}],
    }


def test_tools_list_offers_configured_models(fake_api):
    config = config_for(fake_api["url"], "openai", "/v1")
    config["models"]["other"] = {"provider": "p", "model": "y"}

    tools = second_opinion.handle(config, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]

    assert [t["name"] for t in tools] == ["ask", "review"]
    assert [t["inputSchema"]["properties"]["model"]["enum"] for t in tools] == [["m", "other"]] * 2


@pytest.fixture
def repo(tmp_path):
    """Git repo with one commit, then uncommitted edits to app.py, .env and server.pem, and an untracked file."""
    root = tmp_path / "repo"
    root.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.com",
           "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.com"}

    def git(*args):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, env=env)

    (root / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (root / ".env").write_text("NAME=one\n")
    (root / "server.pem").write_text("cert one\n")
    git("init", "-q")
    git("add", "app.py", ".env", "server.pem")
    git("-c", "commit.gpgsign=false", "commit", "-q", "-m", "init")
    (root / "app.py").write_text("def add(a, b):\n    return a - b\n")
    (root / ".env").write_text("NAME=two\n")
    (root / "server.pem").write_text("cert two\n")
    (root / "new.py").write_text("untracked = True\n")
    return root


def test_review_sends_the_tracked_diff_without_secret_files(fake_api, repo, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    fake_api["reply"] = OK

    result = call(config_for(fake_api["url"], "openai", "/v1"), "review",
                  {"model": "m", "repo": str(repo), "focus": "arithmetic"})

    assert result["isError"] is False
    messages = fake_api["requests"][0]["body"]["messages"]
    assert "file:line | severity | issue" in messages[0]["content"]
    user = messages[1]["content"]
    assert "-    return a + b" in user and "+    return a - b" in user
    assert "Focus: arithmetic" in user
    for hidden in (".env", "NAME=two", "server.pem", "cert two", "untracked"):
        assert hidden not in user


def test_review_of_only_excluded_files_is_an_error(fake_api, repo, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")

    result = call(config_for(fake_api["url"], "openai", "/v1"), "review",
                  {"model": "m", "repo": str(repo), "paths": [".env", "server.pem"]})

    assert result["isError"] is True
    assert "no tracked changes" in text_of(result)
    assert fake_api["requests"] == []


@pytest.mark.parametrize("args", [{"ref": "--output=x.txt"}, {"paths": ["--cached"]}])
def test_review_refuses_option_like_arguments(fake_api, repo, monkeypatch, args):
    monkeypatch.setenv("TEST_KEY", "sk-test")

    result = call(config_for(fake_api["url"], "openai", "/v1"), "review", {"model": "m", "repo": str(repo), **args})

    assert result["isError"] is True
    assert not (repo / "x.txt").exists()
    assert fake_api["requests"] == []


def test_review_outside_a_repo_is_an_error(fake_api, tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")

    result = call(config_for(fake_api["url"], "openai", "/v1"), "review", {"model": "m", "repo": str(tmp_path)})

    assert result["isError"] is True
    assert "git diff failed" in text_of(result)
    assert fake_api["requests"] == []


def test_tool_result_is_wrapped_and_marked_unverified(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    fake_api["reply"] = OK

    result = call(config_for(fake_api["url"], "openai", "/v1"), "ask", {"model": "m", "prompt": "hi"})

    assert result["isError"] is False
    assert text_of(result).startswith('<second_opinion model="m" input_tokens="11" output_tokens="3">\nok\n'
                                      '</second_opinion>\nUnverified second opinion')


def test_secrets_are_redacted_before_sending(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "configured-key-value-123")
    fake_api["reply"] = OK
    prompt = ("aws = 'AKIAIOSFODNN7EXAMPLE'\n"
              "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----\n"
              "+DB_PASSWORD=hunter2hunter2\n"
              "leaked = 'configured-key-value-123'\n")

    result = call(config_for(fake_api["url"], "openai", "/v1"), "ask", {"model": "m", "prompt": prompt})

    sent = sent_prompt(fake_api)
    for secret in ("AKIAIOSFODNN7EXAMPLE", "MIIEow", "hunter2hunter2", "configured-key-value-123"):
        assert secret not in sent
    assert "DB_PASSWORD=[REDACTED]" in sent
    assert "4 secret(s) were redacted" in text_of(result)


def test_ordinary_code_is_not_redacted(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    fake_api["reply"] = OK
    prompt = 'key = os.environ.get(provider["api_key_env"])\nMAX_TOKENS = 4096\nAPI_KEY = os.environ["API_KEY"]\n'

    result = call(config_for(fake_api["url"], "openai", "/v1"), "ask", {"model": "m", "prompt": prompt})

    assert sent_prompt(fake_api) == prompt
    assert "redacted" not in text_of(result)


def test_prompt_over_max_chars_is_refused(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    config = config_for(fake_api["url"], "openai", "/v1")
    config["max_chars"] = 10

    result = call(config, "ask", {"model": "m", "prompt": "x" * 6, "system": "y" * 5})

    assert result["isError"] is True
    assert "max_chars" in text_of(result)
    assert fake_api["requests"] == []


def test_rate_limit_is_retried_once(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    fake_api["queue"] = [(429, {"Retry-After": "0"}, {"error": "slow down"})]
    fake_api["reply"] = OK

    result = call(config_for(fake_api["url"], "openai", "/v1"), "ask", {"model": "m", "prompt": "hi"})

    assert result["isError"] is False
    assert len(fake_api["requests"]) == 2


def test_server_error_is_retried_only_once(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    fake_api["status"], fake_api["headers"] = 503, {"Retry-After": "0"}

    result = call(config_for(fake_api["url"], "openai", "/v1"), "ask", {"model": "m", "prompt": "hi"})

    assert result["isError"] is True
    assert "HTTP 503" in text_of(result)
    assert len(fake_api["requests"]) == 2


def test_client_error_is_not_retried(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    fake_api["status"] = 400

    result = call(config_for(fake_api["url"], "openai", "/v1"), "ask", {"model": "m", "prompt": "hi"})

    assert result["isError"] is True
    assert len(fake_api["requests"]) == 1


def test_missing_key_is_a_tool_error(fake_api, monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)

    result = call(config_for(fake_api["url"], "openai", "/v1"), "ask", {"model": "m", "prompt": "hi"})

    assert result["isError"] is True
    assert "TEST_KEY" in text_of(result)
    assert fake_api["requests"] == []


def test_http_error_body_reaches_the_caller(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    fake_api["status"], fake_api["reply"] = 404, {"error": {"message": "deployment not found"}}

    result = call(config_for(fake_api["url"], "openai", "/v1"), "ask", {"model": "m", "prompt": "hi"})

    assert result["isError"] is True
    assert "HTTP 404" in text_of(result)
    assert "deployment not found" in text_of(result)
    assert "sk-test" not in text_of(result)


def test_redirect_is_not_followed(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    fake_api["status"], fake_api["headers"] = 302, {"Location": fake_api["url"] + "/elsewhere"}

    result = call(config_for(fake_api["url"], "openai", "/v1"), "ask", {"model": "m", "prompt": "hi"})

    assert result["isError"] is True
    assert "HTTP 302" in text_of(result)
    assert [r["path"] for r in fake_api["requests"]] == ["/v1/chat/completions"]


def test_requests_name_the_client(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    fake_api["reply"] = OK

    second_opinion.ask(config_for(fake_api["url"], "openai", "/v1"), "m", "hi")

    assert fake_api["requests"][0]["headers"]["user-agent"] == f"second-opinion-mcp/{second_opinion.__version__}"


def test_unknown_tool_is_invalid_params(fake_api):
    response = second_opinion.handle(config_for(fake_api["url"], "openai", "/v1"),
                                     {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "nope"}})

    assert response["error"]["code"] == -32602


def test_unknown_method_is_method_not_found(fake_api):
    response = second_opinion.handle(config_for(fake_api["url"], "openai", "/v1"),
                                     {"jsonrpc": "2.0", "id": 4, "method": "resources/list"})

    assert response["error"]["code"] == -32601


@pytest.mark.parametrize("asked, answered", [("2025-03-26", "2025-03-26"), ("1999-01-01", "2025-06-18")])
def test_initialize_negotiates_protocol_version(fake_api, asked, answered):
    response = second_opinion.handle(config_for(fake_api["url"], "openai", "/v1"),
                                     {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                      "params": {"protocolVersion": asked}})

    assert response["result"]["protocolVersion"] == answered


def test_unknown_model_lists_the_configured_ones(fake_api):
    result = call(config_for(fake_api["url"], "openai", "/v1"), "ask", {"model": "nope", "prompt": "hi"})

    assert result["isError"] is True
    assert "configured: m" in text_of(result)


def test_config_with_unknown_provider_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"providers": {}, "models": {"m": {"provider": "ghost", "model": "x"}}}))

    with pytest.raises(ValueError, match="ghost"):
        second_opinion.load_config(path)


@pytest.mark.parametrize("config, message", [
    ({"providers": {"p": {"type": "openai", "api_key_env": "K"}}, "models": {"m": {"provider": "p", "model": "x"}}},
     "base_url"),
    ({"providers": {"p": {"type": "openai", "base_url": "x", "api_key_env": "K"}}, "models": {"m": {"provider": "p"}}},
     "'model'"),
    ({"providers": {}, "models": {}}, "no models"),
    ({"providers": [], "models": {}}, "providers"),
    ({"providers": {"p": {"type": "openai", "base_url": "x", "api_key_env": "K"}},
      "models": {"m": {"provider": "p", "model": "x"}}, "max_chars": 0}, "max_chars"),
])
def test_incomplete_config_is_rejected(tmp_path, config, message):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=message):
        second_opinion.load_config(path)


def test_example_config_is_valid():
    second_opinion.load_config(Path(__file__).parent.parent / "config.example.json")


def test_config_with_unknown_type_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"providers": {"p": {"type": "gemini", "base_url": "x", "api_key_env": "K"}},
                                "models": {}}))

    with pytest.raises(ValueError, match="gemini"):
        second_opinion.load_config(path)


def test_stdio_server_answers_initialize_and_tools_list(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_for("http://127.0.0.1:9", "openai", "/v1")))
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        "{not json",
        "[1, 2]",
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ]

    out = subprocess.run([sys.executable, second_opinion.__file__], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, encoding="utf-8", timeout=30,
                         env={**os.environ, "SECOND_OPINION_CONFIG": str(path)})

    responses = [json.loads(line) for line in out.stdout.splitlines()]
    assert [r["id"] for r in responses] == [1, None, None, 2]
    assert responses[0]["result"]["protocolVersion"] == "2025-06-18"
    assert responses[0]["result"]["serverInfo"]["name"] == "second-opinion-mcp"
    assert [r["error"]["code"] for r in responses[1:3]] == [-32700, -32600]
    assert responses[3]["result"]["tools"][0]["name"] == "ask"


def test_stdio_server_runs_tool_calls_in_parallel(fake_api, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_for(fake_api["url"], "openai", "/v1")))
    # each request waits until the other one arrives, so calls made one at a time fail
    fake_api["reply"], fake_api["barrier"] = OK, threading.Barrier(2, timeout=10)
    lines = [json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                         "params": {"name": "ask", "arguments": {"model": "m", "prompt": "hi"}}}) for i in (1, 2)]

    started = time.monotonic()
    out = subprocess.run([sys.executable, second_opinion.__file__], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, encoding="utf-8", timeout=30,
                         env={**os.environ, "SECOND_OPINION_CONFIG": str(path), "TEST_KEY": "sk-test"})

    responses = sorted((json.loads(line) for line in out.stdout.splitlines()), key=lambda r: r["id"])
    assert [r["id"] for r in responses] == [1, 2]
    assert [r["result"]["isError"] for r in responses] == [False, False]
    assert time.monotonic() - started < 8
