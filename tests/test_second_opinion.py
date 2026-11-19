import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import second_opinion


@pytest.fixture
def fake_api():
    """Local HTTP server that records each request and answers with `reply`."""
    state = {"requests": [], "status": 200, "reply": {}, "headers": {}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            state["requests"].append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": json.loads(body)})
            self.send_response(state["status"])
            for name, value in state["headers"].items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(json.dumps(state["reply"]).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()


def config_for(url, type_, base_path):
    return {
        "providers": {"p": {"type": type_, "base_url": url + base_path, "api_key_env": "TEST_KEY"}},
        "models": {"m": {"provider": "p", "model": "model-x", "params": {"temperature": 0.2}}},
    }


def test_openai_style_request_and_reply(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    fake_api["reply"] = {"choices": [{"message": {"content": "looks fine"}}]}

    text = second_opinion.ask(config_for(fake_api["url"], "openai", "/openai/v1/"), "m", "review this", "be strict")

    assert text == "looks fine"
    req = fake_api["requests"][0]
    assert req["path"] == "/openai/v1/chat/completions"
    assert req["headers"]["authorization"] == "Bearer sk-test"
    assert req["body"] == {
        "model": "model-x",
        "temperature": 0.2,
        "messages": [{"role": "system", "content": "be strict"}, {"role": "user", "content": "review this"}],
    }


def test_anthropic_style_request_and_reply(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-ant")
    fake_api["reply"] = {"content": [{"type": "thinking", "thinking": "hmm"},
                                     {"type": "text", "text": "one bug"}, {"type": "text", "text": " on line 3"}]}

    text = second_opinion.ask(config_for(fake_api["url"], "anthropic", "/anthropic"), "m", "review this", "be strict")

    assert text == "one bug on line 3"
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


def call(config, name, arguments):
    return second_opinion.handle(config, {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                          "params": {"name": name, "arguments": arguments}})["result"]


def test_tools_list_offers_configured_models(fake_api):
    config = config_for(fake_api["url"], "openai", "/v1")
    config["models"]["other"] = {"provider": "p", "model": "y"}

    tools = second_opinion.handle(config, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]

    assert [t["name"] for t in tools] == ["ask"]
    assert tools[0]["inputSchema"]["properties"]["model"]["enum"] == ["m", "other"]


def test_missing_key_is_a_tool_error(fake_api, monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)

    result = call(config_for(fake_api["url"], "openai", "/v1"), "ask", {"model": "m", "prompt": "hi"})

    assert result["isError"] is True
    assert "TEST_KEY" in result["content"][0]["text"]
    assert fake_api["requests"] == []


def test_http_error_body_reaches_the_caller(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    fake_api["status"], fake_api["reply"] = 404, {"error": {"message": "deployment not found"}}

    result = call(config_for(fake_api["url"], "openai", "/v1"), "ask", {"model": "m", "prompt": "hi"})

    assert result["isError"] is True
    assert "HTTP 404" in result["content"][0]["text"]
    assert "deployment not found" in result["content"][0]["text"]
    assert "sk-test" not in result["content"][0]["text"]


def test_redirect_is_not_followed(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    fake_api["status"], fake_api["headers"] = 302, {"Location": fake_api["url"] + "/elsewhere"}

    result = call(config_for(fake_api["url"], "openai", "/v1"), "ask", {"model": "m", "prompt": "hi"})

    assert result["isError"] is True
    assert "HTTP 302" in result["content"][0]["text"]
    assert [r["path"] for r in fake_api["requests"]] == ["/v1/chat/completions"]


def test_requests_name_the_client(fake_api, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    fake_api["reply"] = {"choices": [{"message": {"content": "ok"}}]}

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
    assert "configured: m" in result["content"][0]["text"]


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
