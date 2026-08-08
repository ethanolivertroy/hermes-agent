"""Tests for the OpenAI-compatible Cursor bridge client.

The bridge itself is stubbed — these tests exercise the prompt/tool
conversion, the loopback callback server (JSON, binary proto, chunked
bodies, auth), and the loop-mode capture flow end to end against a fake
transport.
"""

import json
import threading
import urllib.error
import urllib.request

import pytest

from agent.cursor_bridge_client import (
    CursorBridgeClient,
    _ToolCallbackServer,
    convert_openai_tools_to_custom_tools,
    format_messages_as_prompt,
)
from agent.cursor_bridge_wire import (
    _encode_varint,
    decode_struct,
    encode_struct,
)


class TestPromptFormatting:
    def test_transcript_includes_roles_and_tool_results(self):
        messages = [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "terminal", "arguments": '{"command": "ls"}'}}
                ],
            },
            {"role": "tool", "content": "file.txt"},
        ]
        prompt = format_messages_as_prompt(messages)
        assert "System:\nYou are Hermes." in prompt
        assert "User:\nhi" in prompt
        assert '[tool call] terminal({"command": "ls"})' in prompt
        assert "Tool result:\nfile.txt" in prompt
        assert prompt.index("System:") < prompt.index("User:")

    def test_multipart_content_flattened(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "part one"}, "part two"]}]
        prompt = format_messages_as_prompt(messages)
        assert "part one" in prompt
        assert "part two" in prompt


class TestToolConversion:
    def test_openai_schema_maps_to_custom_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web.",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            },
            {"type": "function", "function": {"name": "no_params"}},
            {"not": "a tool"},
        ]
        custom = convert_openai_tools_to_custom_tools(tools)
        assert set(custom) == {"web_search", "no_params"}
        assert custom["web_search"]["description"] == "Search the web."
        assert custom["web_search"]["inputSchema"]["properties"]["q"]["type"] == "string"
        # Tools without parameters still get a valid empty JSON Schema object.
        assert custom["no_params"]["inputSchema"] == {"type": "object", "properties": {}}

    def test_none_and_empty(self):
        assert convert_openai_tools_to_custom_tools(None) == {}
        assert convert_openai_tools_to_custom_tools([]) == {}


# ── Callback server ───────────────────────────────────────────────────────


@pytest.fixture()
def callback_server():
    received = []

    def handler(request):
        received.append(request)
        return {"value": "handled"}

    server = _ToolCallbackServer(handler)
    server.start()
    yield server, received
    server.stop()


def _post(url, path, body, *, token, content_type="application/json", chunked=False):
    headers = {"Content-Type": content_type, "Authorization": f"Bearer {token}"}
    if chunked:
        # urllib sends Transfer-Encoding: chunked when data is an iterable
        # without a length.
        def gen():
            yield body

        request = urllib.request.Request(url + path, data=gen(), method="POST", headers=headers)
    else:
        request = urllib.request.Request(url + path, data=body, method="POST", headers=headers)
    return urllib.request.urlopen(request, timeout=10)


CALLBACK_PATH = "/sdk.v1.SdkCustomToolCallbackService/CallCustomTool"


class TestCallbackServer:
    def test_json_request_dispatches_and_wraps_result(self, callback_server):
        server, received = callback_server
        body = json.dumps(
            {"toolName": "memory", "args": {"action": "save"}, "agentId": "agent-1"}
        ).encode()
        with _post(server.url, CALLBACK_PATH, body, token=server.auth_token) as reply:
            payload = json.loads(reply.read())
        assert payload == {"result": {"value": "handled"}}
        assert received[0]["toolName"] == "memory"
        assert received[0]["args"] == {"action": "save"}

    def test_chunked_json_request_decodes(self, callback_server):
        server, received = callback_server
        body = json.dumps({"toolName": "t", "args": {}, "agentId": "a"}).encode()
        with _post(server.url, CALLBACK_PATH, body, token=server.auth_token, chunked=True) as reply:
            payload = json.loads(reply.read())
        assert payload["result"] == {"value": "handled"}
        assert received[0]["toolName"] == "t"

    def test_proto_request_decodes_and_responds_proto(self, callback_server):
        server, received = callback_server
        name = b"proto_tool"
        agent = b"agent-2"
        args = encode_struct({"k": "v"})
        raw = (
            _encode_varint((1 << 3) | 2) + _encode_varint(len(name)) + name
            + _encode_varint((2 << 3) | 2) + _encode_varint(len(args)) + args
            + _encode_varint((4 << 3) | 2) + _encode_varint(len(agent)) + agent
        )
        with _post(
            server.url, CALLBACK_PATH, raw,
            token=server.auth_token, content_type="application/proto",
        ) as reply:
            assert reply.headers["Content-Type"] == "application/proto"
            encoded = reply.read()
        # response = field 1 (Struct)
        length = encoded[1]
        assert decode_struct(encoded[2 : 2 + length]) == {"value": "handled"}
        assert received[0]["toolName"] == "proto_tool"
        assert received[0]["args"] == {"k": "v"}

    def test_bad_token_rejected(self, callback_server):
        server, _ = callback_server
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(server.url, CALLBACK_PATH, b"{}", token="wrong-token")
        assert excinfo.value.code == 401

    def test_unknown_path_rejected(self, callback_server):
        server, _ = callback_server
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(server.url, "/sdk.v1.Other/Rpc", b"{}", token=server.auth_token)
        assert excinfo.value.code == 404

    def test_connection_reset_is_swallowed(self, callback_server, capfd):
        """Loop mode resets callback connections on CancelRun; the server must
        not dump a ConnectionResetError traceback to stderr."""
        import socket
        import struct
        import time

        server, _ = callback_server
        host, port = server.server_address
        for _ in range(3):
            sock = socket.create_connection((host, port))
            sock.sendall(
                b"POST " + CALLBACK_PATH.encode() + b" HTTP/1.1\r\nHost: x\r\n"
            )
            # SO_LINGER 0 => abortive close => RST => ConnectionResetError
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            sock.close()
            time.sleep(0.1)
        time.sleep(0.2)
        err = capfd.readouterr().err
        assert "ConnectionResetError" not in err
        assert "Traceback" not in err
        # The server is still serving after the resets.
        body = json.dumps({"toolName": "t", "agentId": "a"}).encode()
        with _post(server.url, CALLBACK_PATH, body, token=server.auth_token) as reply:
            assert reply.status == 200

    def test_handler_exception_returns_error_result(self):
        def handler(request):
            raise RuntimeError("tool blew up")

        server = _ToolCallbackServer(handler)
        server.start()
        try:
            body = json.dumps({"toolName": "t", "agentId": "a"}).encode()
            with _post(server.url, CALLBACK_PATH, body, token=server.auth_token) as reply:
                payload = json.loads(reply.read())
            assert "tool blew up" in payload["result"]["error"]
        finally:
            server.stop()


# ── Loop-mode end-to-end against a fake transport ─────────────────────────


class _FakeTransport:
    """Programmable stand-in for ConnectJsonTransport."""

    def __init__(self):
        self.calls = []
        self.stream_events = []
        self.on_stream_start = None
        self.list_runs_items = []

    def unary(self, service, method, request, timeout=60.0):
        self.calls.append((service, method, request))
        if method == "CreateAgent":
            return {"agentId": "agent-test", "model": {"id": request["options"]["model"]["id"]}}
        if method == "ListRuns":
            return {"items": self.list_runs_items}
        return {}

    def server_stream(self, service, method, request, deadline=None, read_timeout=90.0):
        self.calls.append((service, method, request))
        if self.on_stream_start:
            self.on_stream_start()
        yield from self.stream_events


def _make_client(fake, tool_mode="loop"):
    client = CursorBridgeClient(api_key="key-1", tool_mode=tool_mode)
    client._transport = fake

    def fake_ensure():
        return fake

    client._ensure_bridge = fake_ensure
    return client


def _finished_event(text, usage=None):
    run_result = {"result": text}
    if usage:
        run_result["usage"] = usage
    return {
        "result": {
            "agentId": "agent-test",
            "runId": "run-1",
            "status": "RUN_LIFECYCLE_STATUS_FINISHED",
            "result": run_result,
        }
    }


class TestLoopModeCompletion:
    def test_plain_text_completion_with_usage(self):
        fake = _FakeTransport()
        fake.stream_events = [
            {"sdkMessage": {"type": "assistant", "message": {"text": "thinking..."}}},
            _finished_event(
                "The answer is 4.",
                usage={"inputTokens": "120", "outputTokens": "8", "totalTokens": "128",
                       "cacheReadTokens": "50"},
            ),
            {"done": {"agentId": "agent-test", "runId": "run-1"}},
        ]
        client = _make_client(fake)
        completion = client.chat.completions.create(
            model="composer-2.5",
            messages=[{"role": "user", "content": "what is 2+2"}],
        )
        message = completion.choices[0].message
        assert message.content == "The answer is 4."
        assert message.tool_calls == []
        assert completion.choices[0].finish_reason == "stop"
        assert completion.usage.prompt_tokens == 120
        assert completion.usage.completion_tokens == 8
        assert completion.usage.total_tokens == 128
        assert completion.usage.prompt_tokens_details.cached_tokens == 50

    def test_builtin_tools_restricted_to_mcp_when_tools_declared(self):
        fake = _FakeTransport()
        fake.stream_events = [_finished_event("done"), {"done": {}}]
        client = _make_client(fake)
        client.chat.completions.create(
            model="composer-2.5",
            messages=[{"role": "user", "content": "go"}],
            tools=[{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}],
        )
        create_request = next(r for s, m, r in fake.calls if m == "CreateAgent")
        options = create_request["options"]
        # Loop mode with tools: only the harness's `mcp` meta-tools stay
        # enabled — that is the channel custom tools ride on (verified live
        # against bridge 1.0.27). Shell/file built-ins stay off so the
        # Cursor harness cannot bypass Hermes approvals.
        assert options["tools"] == {"names": ["mcp"]}
        assert "terminal" in options["local"]["customTools"]

    def test_no_tools_means_pure_text_generation(self):
        fake = _FakeTransport()
        fake.stream_events = [_finished_event("done"), {"done": {}}]
        client = _make_client(fake)
        client.chat.completions.create(
            model="composer-2.5",
            messages=[{"role": "user", "content": "go"}],
        )
        create_request = next(r for s, m, r in fake.calls if m == "CreateAgent")
        assert create_request["options"]["tools"] == {"names": []}

    def test_agent_deleted_after_run(self):
        fake = _FakeTransport()
        fake.stream_events = [_finished_event("ok"), {"done": {}}]
        client = _make_client(fake)
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        assert any(m == "DeleteAgent" for _, m, _ in fake.calls)

    def test_error_status_raises(self):
        fake = _FakeTransport()
        fake.stream_events = [
            {
                "result": {
                    "status": "RUN_LIFECYCLE_STATUS_ERROR",
                    "errorCode": "quota_exceeded",
                    "result": {"result": ""},
                }
            },
            {"done": {}},
        ]
        client = _make_client(fake)
        with pytest.raises(Exception) as excinfo:
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        assert "quota_exceeded" in str(excinfo.value)

    def test_stream_kwarg_returns_chunks(self):
        fake = _FakeTransport()
        fake.stream_events = [_finished_event("streamed text"), {"done": {}}]
        client = _make_client(fake)
        chunks = client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}], stream=True
        )
        assert chunks[0].choices[0].delta.content == "streamed text"
        assert chunks[1].usage is not None

    def test_default_model_placeholder_maps_to_auto(self):
        fake = _FakeTransport()
        fake.stream_events = [_finished_event("ok"), {"done": {}}]
        client = _make_client(fake)
        client.chat.completions.create(model="cursor", messages=[{"role": "user", "content": "x"}])
        create_request = next(r for s, m, r in fake.calls if m == "CreateAgent")
        assert create_request["options"]["model"]["id"] == "auto"


class TestLoopModeToolCapture:
    def test_tool_callback_mid_run_becomes_openai_tool_call(self):
        fake = _FakeTransport()
        fake.list_runs_items = [
            {"runId": "run-1", "status": "RUN_LIFECYCLE_STATUS_RUNNING"}
        ]
        client = _make_client(fake)

        cancelled = threading.Event()
        original_unary = fake.unary

        def tracking_unary(service, method, request, timeout=60.0):
            if method == "CancelRun":
                cancelled.set()
            return original_unary(service, method, request, timeout=timeout)

        fake.unary = tracking_unary

        def simulate_bridge_tool_call():
            # The bridge invokes the callback while the Send stream is live.
            response = client._handle_tool_callback(
                {
                    "toolName": "send_message",
                    "args": {"text": "hello"},
                    "toolCallId": "call-7",
                    "agentId": "agent-test",
                }
            )
            assert response["status"] == "deferred"

        fake.on_stream_start = simulate_bridge_tool_call
        fake.stream_events = [
            {
                "result": {
                    "status": "RUN_LIFECYCLE_STATUS_CANCELLED",
                    "result": {"result": ""},
                }
            },
            {"done": {}},
        ]

        completion = client.chat.completions.create(
            model="composer-2.5",
            messages=[{"role": "user", "content": "message me"}],
            tools=[{"type": "function", "function": {"name": "send_message", "parameters": {"type": "object"}}}],
        )
        message = completion.choices[0].message
        assert completion.choices[0].finish_reason == "tool_calls"
        assert len(message.tool_calls) == 1
        call = message.tool_calls[0]
        assert call.id == "call-7"
        assert call.function.name == "send_message"
        assert json.loads(call.function.arguments) == {"text": "hello"}
        assert cancelled.wait(timeout=5), "CancelRun was never issued"

    def test_callback_for_unknown_agent_returns_error(self):
        fake = _FakeTransport()
        client = _make_client(fake)
        response = client._handle_tool_callback(
            {"toolName": "t", "args": {}, "agentId": "agent-unknown"}
        )
        assert "error" in response


class TestHarnessMode:
    def test_dispatcher_executes_inline_and_returns_result(self):
        fake = _FakeTransport()
        executed = []

        def dispatcher(name, args, call_id):
            executed.append((name, args, call_id))
            return json.dumps({"success": True, "data": 42})

        client = CursorBridgeClient(
            api_key="k", tool_mode="harness", tool_dispatcher=dispatcher
        )
        client._transport = fake
        client._ensure_bridge = lambda: fake
        client._active_runs["agent-test"] = __import__(
            "agent.cursor_bridge_client", fromlist=["_ActiveRun"]
        )._ActiveRun("agent-test", {"my_tool"})

        response = client._handle_tool_callback(
            {"toolName": "my_tool", "args": {"a": 1}, "toolCallId": "c1", "agentId": "agent-test"}
        )
        assert response == {"success": True, "data": 42}
        assert executed == [("my_tool", {"a": 1}, "c1")]

    def test_harness_mode_keeps_builtin_tools_configurable(self):
        fake = _FakeTransport()
        fake.stream_events = [_finished_event("ok"), {"done": {}}]
        client = CursorBridgeClient(api_key="k", tool_mode="harness", builtin_tools=True)
        client._transport = fake
        client._ensure_bridge = lambda: fake
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        create_request = next(r for s, m, r in fake.calls if m == "CreateAgent")
        assert "tools" not in create_request["options"]


class TestMissingCredentials:
    def test_no_credentials_raise_actionable_error(self, tmp_path, monkeypatch):
        from pathlib import Path

        # Isolate the SDK credential store: a developer's real
        # ~/.cursor/sdk/auth.json must not satisfy the fallback here.
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        client = CursorBridgeClient(api_key="")
        with pytest.raises(Exception) as excinfo:
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        message = str(excinfo.value)
        assert "hermes cursor login" in message
        assert "CURSOR_API_KEY" in message

    def test_sdk_login_store_satisfies_credential_check(self, tmp_path, monkeypatch):
        from pathlib import Path

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        from agent.cursor_sdk_auth import save_sdk_credentials

        save_sdk_credentials(backend_url="https://api2.cursor.sh", api_key="key_from_login")
        client = CursorBridgeClient(api_key="")
        # Force bridge resolution to fail AFTER the credential check so the
        # test proves which credential was picked up without spawning.
        monkeypatch.setattr(
            "agent.cursor_bridge_client.resolve_bridge_command", lambda *_a, **_k: None
        )
        with pytest.raises(Exception) as excinfo:
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        assert "bridge not found" in str(excinfo.value)
        assert client.api_key == "key_from_login"
