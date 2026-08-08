"""Tests for bridge discovery parsing and the Connect JSON transport."""

import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent.cursor_bridge_transport import (
    ConnectJsonTransport,
    CursorBridgeError,
    READY_LINE_PREFIX,
    endpoint_from_discovery,
    parse_ready_line,
    resolve_bridge_command,
)


def _discovery(tmp_path, **overrides):
    token_file = tmp_path / "auth-token"
    token_file.write_text("secret-token\n", encoding="utf-8")
    payload = {
        "schemaVersion": 1,
        "serverVersion": "1.0.0",
        "pid": 4242,
        "transport": "tcp",
        "protocol": "connect",
        "host": "127.0.0.1",
        "port": 49152,
        "url": "http://127.0.0.1:49152",
        "authTokenFile": str(token_file),
        "workspaceRef": "/repo",
        "stateRoot": "/state",
    }
    payload.update(overrides)
    return payload


class TestReadyLine:
    def test_non_ready_lines_return_none(self):
        assert parse_ready_line("some diagnostic output") is None
        assert parse_ready_line("") is None

    def test_ready_line_parses_json(self, tmp_path):
        payload = _discovery(tmp_path)
        line = READY_LINE_PREFIX + json.dumps(payload)
        assert parse_ready_line(line) == payload

    def test_invalid_json_raises(self):
        with pytest.raises(CursorBridgeError):
            parse_ready_line(READY_LINE_PREFIX + "{not json")

    def test_endpoint_reads_token_file(self, tmp_path):
        endpoint = endpoint_from_discovery(_discovery(tmp_path))
        assert endpoint.url == "http://127.0.0.1:49152"
        assert endpoint.auth_token == "secret-token"
        assert endpoint.pid == 4242
        assert endpoint.workspace_ref == "/repo"

    def test_inline_auth_token_preferred(self, tmp_path):
        endpoint = endpoint_from_discovery(_discovery(tmp_path, authToken="inline-tok"))
        assert endpoint.auth_token == "inline-tok"

    def test_host_port_fallback_when_url_missing(self, tmp_path):
        endpoint = endpoint_from_discovery(_discovery(tmp_path, url=""))
        assert endpoint.url == "http://127.0.0.1:49152"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"schemaVersion": 2},
            {"transport": "unix"},
            {"protocol": "grpc"},
        ],
    )
    def test_unsupported_discovery_rejected(self, tmp_path, overrides):
        with pytest.raises(CursorBridgeError):
            endpoint_from_discovery(_discovery(tmp_path, **overrides))

    def test_missing_token_file_raises(self, tmp_path):
        payload = _discovery(tmp_path, authTokenFile=str(tmp_path / "missing"))
        with pytest.raises(CursorBridgeError):
            endpoint_from_discovery(payload)


class TestResolveBridgeCommand:
    def test_configured_command_wins(self, tmp_path):
        binary = tmp_path / "my-bridge"
        binary.write_text("#!/bin/sh\n")
        assert resolve_bridge_command(str(binary)) == str(binary)

    def test_managed_install_found_via_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("CURSOR_SDK_BRIDGE_BIN", raising=False)
        monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
        launcher = tmp_path / "cursor-sdk-bridge" / "cursor-sdk-bridge" / "bin" / "cursor-sdk-bridge"
        launcher.parent.mkdir(parents=True)
        launcher.write_text("#!/bin/sh\n")
        assert resolve_bridge_command() == str(launcher)

    def test_returns_none_when_nothing_installed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("CURSOR_SDK_BRIDGE_BIN", raising=False)
        monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
        assert resolve_bridge_command() is None

    def test_env_var_respected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
        binary = tmp_path / "env-bridge"
        binary.write_text("#!/bin/sh\n")
        monkeypatch.setenv("CURSOR_SDK_BRIDGE_BIN", str(binary))
        assert resolve_bridge_command() == str(binary)


# ── Connect transport against a stub server ───────────────────────────────


class _StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_POST(self):
        server = self.server
        server.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "body": self._body(),
            }
        )
        route = server.routes.get(self.path)
        if route is None:
            payload = json.dumps({"code": "unimplemented", "message": "no route"}).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        status, content_type, body = route
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _StubServer(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), _StubHandler)
        self.routes = {}
        self.requests = []

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


def _frame(payload: bytes, flags: int = 0) -> bytes:
    return struct.pack(">BI", flags, len(payload)) + payload


@pytest.fixture()
def stub_server():
    server = _StubServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


class TestConnectJsonTransport:
    def test_unary_success_sends_bearer_and_parses_json(self, stub_server):
        stub_server.routes["/sdk.v1.SdkAgentService/CreateAgent"] = (
            200,
            "application/json",
            json.dumps({"agentId": "agent-1"}).encode(),
        )
        transport = ConnectJsonTransport(stub_server.url, "tok-123")
        response = transport.unary(
            "SdkAgentService", "CreateAgent", {"options": {"name": "x"}}
        )
        assert response == {"agentId": "agent-1"}
        request = stub_server.requests[0]
        assert request["headers"]["Authorization"] == "Bearer tok-123"
        assert request["headers"]["Connect-Protocol-Version"] == "1"
        assert json.loads(request["body"]) == {"options": {"name": "x"}}

    def test_unary_connect_error_raises_with_code(self, stub_server):
        stub_server.routes["/sdk.v1.SdkAgentService/GetRun"] = (
            401,
            "application/json",
            json.dumps({"code": "unauthenticated", "message": "bad key"}).encode(),
        )
        transport = ConnectJsonTransport(stub_server.url, "tok")
        with pytest.raises(CursorBridgeError) as excinfo:
            transport.unary("SdkAgentService", "GetRun", {"runId": "r"})
        assert excinfo.value.code == "unauthenticated"
        assert "bad key" in str(excinfo.value)

    def test_server_stream_yields_frames_until_end(self, stub_server):
        body = (
            _frame(json.dumps({"sdkMessage": {"type": "assistant"}}).encode())
            + _frame(b"")  # keepalive-style empty frame is skipped
            + _frame(json.dumps({"result": {"status": "RUN_LIFECYCLE_STATUS_FINISHED"}}).encode())
            + _frame(json.dumps({"done": {}}).encode())
            + _frame(json.dumps({}).encode(), flags=0x02)
        )
        stub_server.routes["/sdk.v1.SdkAgentService/Send"] = (
            200,
            "application/connect+json",
            body,
        )
        transport = ConnectJsonTransport(stub_server.url, "tok")
        messages = list(
            transport.server_stream("SdkAgentService", "Send", {"agentId": "a"})
        )
        assert [set(m) for m in messages] == [
            {"sdkMessage"},
            {"result"},
            {"done"},
        ]
        # The request itself must be a single enveloped JSON frame.
        raw = stub_server.requests[0]["body"]
        flags, length = struct.unpack(">BI", raw[:5])
        assert flags == 0
        assert json.loads(raw[5 : 5 + length]) == {"agentId": "a"}

    def test_server_stream_end_frame_error_raises(self, stub_server):
        body = _frame(
            json.dumps({"error": {"code": "internal", "message": "boom"}}).encode(),
            flags=0x02,
        )
        stub_server.routes["/sdk.v1.SdkAgentService/Send"] = (
            200,
            "application/connect+json",
            body,
        )
        transport = ConnectJsonTransport(stub_server.url, "tok")
        with pytest.raises(CursorBridgeError) as excinfo:
            list(transport.server_stream("SdkAgentService", "Send", {"agentId": "a"}))
        assert excinfo.value.code == "internal"

    def test_server_stream_truncated_raises(self, stub_server):
        stub_server.routes["/sdk.v1.SdkAgentService/Send"] = (
            200,
            "application/connect+json",
            _frame(json.dumps({"done": {}}).encode()),  # no end-stream frame
        )
        transport = ConnectJsonTransport(stub_server.url, "tok")
        with pytest.raises(CursorBridgeError):
            list(transport.server_stream("SdkAgentService", "Send", {"agentId": "a"}))
