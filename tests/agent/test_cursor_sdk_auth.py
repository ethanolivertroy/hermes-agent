"""Tests for the Cursor SDK browser-login flow (`hermes cursor login`)."""

import base64
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from agent.cursor_sdk_auth import (
    CursorAuthError,
    create_login_handshake,
    login,
    mint_user_api_key,
    poll_for_login_tokens,
    read_sdk_credentials,
    resolve_cursor_api_key,
    save_sdk_credentials,
    sdk_auth_path,
)


class TestHandshake:
    def test_challenge_is_sha256_of_verifier(self):
        handshake = create_login_handshake("https://cursor.com")
        query = parse_qs(urlparse(handshake.login_url).query)
        expected = (
            base64.urlsafe_b64encode(
                hashlib.sha256(handshake.verifier.encode()).digest()
            )
            .decode()
            .rstrip("=")
        )
        assert query["challenge"] == [expected]
        assert query["uuid"] == [handshake.uuid]
        assert query["mode"] == ["login"]
        assert query["redirectTarget"] == ["sdk"]

    def test_verifier_not_in_login_url(self):
        handshake = create_login_handshake("https://cursor.com")
        assert handshake.verifier not in handshake.login_url

    def test_each_handshake_is_unique(self):
        first = create_login_handshake("https://cursor.com")
        second = create_login_handshake("https://cursor.com")
        assert first.verifier != second.verifier
        assert first.uuid != second.uuid


# ── Stub backend ──────────────────────────────────────────────────────────


class _AuthStubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        server = self.server
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else {}
        server.requests.append(("POST", self.path, dict(self.headers), body))
        if self.path == "/auth/poll":
            if server.poll_route_missing:
                self._reply(404, {"message": "Route POST:/auth/poll not found"})
                return
            server.poll_count += 1
            if server.poll_count < server.succeed_on_poll:
                self._reply(404, {"error": "pending"})
                return
            self._reply(200, {"accessToken": "sess-token", "refreshToken": "refresh"})
            return
        if self.path == "/aiserver.v1.DashboardService/CreateUserApiKey":
            if self.headers.get("Authorization") != "Bearer sess-token":
                self._reply(401, {"code": "unauthenticated", "message": "bad token"})
                return
            server.minted.append(body)
            self._reply(200, {"apiKey": "key_minted_123"})
            return
        if self.path == "/aiserver.v1.DashboardService/GetMe":
            self._reply(200, {"email": "user@example.com"})
            return
        self._reply(404, {"message": f"Route POST:{self.path} not found"})

    def do_GET(self):
        server = self.server
        server.requests.append(("GET", self.path, dict(self.headers), None))
        if self.path.startswith("/auth/poll"):
            self._reply(200, {"accessToken": "sess-token-get", "refreshToken": "r"})
            return
        self._reply(404, {})


class _AuthStub(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), _AuthStubHandler)
        self.requests = []
        self.poll_count = 0
        self.succeed_on_poll = 1
        self.poll_route_missing = False
        self.minted = []

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


@pytest.fixture()
def auth_stub():
    server = _AuthStub()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


class TestPoll:
    def test_pending_then_success_via_post(self, auth_stub):
        auth_stub.succeed_on_poll = 3
        tokens = poll_for_login_tokens(
            api_url=auth_stub.url, uuid="u1", verifier="v1", sleep=lambda _s: None
        )
        assert tokens == {"accessToken": "sess-token", "refreshToken": "refresh"}
        # Verifier travels in the POST body, never in a URL.
        post_polls = [r for r in auth_stub.requests if r[0] == "POST" and r[1] == "/auth/poll"]
        assert post_polls and all(r[3] == {"uuid": "u1", "verifier": "v1"} for r in post_polls)

    def test_route_not_found_falls_back_to_get(self, auth_stub):
        auth_stub.poll_route_missing = True
        tokens = poll_for_login_tokens(
            api_url=auth_stub.url, uuid="u2", verifier="v2", sleep=lambda _s: None
        )
        assert tokens == {"accessToken": "sess-token-get", "refreshToken": "r"}
        assert any(r[0] == "GET" for r in auth_stub.requests)

    def test_timeout_returns_none(self, auth_stub):
        auth_stub.succeed_on_poll = 10**9
        tokens = poll_for_login_tokens(
            api_url=auth_stub.url,
            uuid="u3",
            verifier="v3",
            max_attempts=3,
            sleep=lambda _s: None,
        )
        assert tokens is None


class TestMint:
    def test_mint_sends_name_and_expiry_with_bearer(self, auth_stub):
        key = mint_user_api_key(
            backend_url=auth_stub.url,
            access_token="sess-token",
            name="Hermes Agent (test)",
            expires_at_ms=1234567890123,
        )
        assert key == "key_minted_123"
        minted = auth_stub.minted[0]
        assert minted["name"] == "Hermes Agent (test)"
        assert minted["expiresAt"] == "1234567890123"  # proto3 JSON int64 = string

    def test_mint_auth_failure_is_actionable(self, auth_stub):
        with pytest.raises(CursorAuthError):
            mint_user_api_key(
                backend_url=auth_stub.url, access_token="wrong", name="n"
            )


class TestCredentialStore:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        return tmp_path

    def test_store_lives_in_cursor_sdk_dir_not_hermes_home(self, _home):
        assert sdk_auth_path() == _home / ".cursor" / "sdk" / "auth.json"

    def test_round_trip_and_permissions(self):
        path = save_sdk_credentials(
            backend_url="https://api2.cursor.sh",
            api_key="key_abc",
            api_key_expires_at_ms=int(time.time() * 1000) + 60_000,
            email="me@example.com",
        )
        assert oct(path.stat().st_mode & 0o777) == "0o600"
        stored = read_sdk_credentials()
        assert stored["apiKey"] == "key_abc"
        assert stored["email"] == "me@example.com"

    def test_expired_credentials_read_as_none(self):
        save_sdk_credentials(
            backend_url="https://api2.cursor.sh",
            api_key="key_old",
            api_key_expires_at_ms=int(time.time() * 1000) - 1,
        )
        assert read_sdk_credentials() is None

    def test_foreign_shaped_file_reads_as_none(self):
        path = sdk_auth_path()
        path.parent.mkdir(parents=True)
        path.write_text('{"something": "else"}', encoding="utf-8")
        assert read_sdk_credentials() is None

    def test_resolve_prefers_env_then_sdk_login(self, monkeypatch):
        save_sdk_credentials(backend_url="https://api2.cursor.sh", api_key="key_store")
        key, source = resolve_cursor_api_key()
        assert (key, source) == ("key_store", "sdk_login")
        monkeypatch.setenv("CURSOR_API_KEY", "key_env")
        key, source = resolve_cursor_api_key()
        assert (key, source) == ("key_env", "env")


class TestFullLogin:
    def test_end_to_end_against_stub(self, auth_stub, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        urls = []
        result = login(
            on_login_url=urls.append,
            backend_url=auth_stub.url,
            website_url="https://cursor.example",
            open_browser=False,
        )
        assert result["apiKey"] == "key_minted_123"
        assert result["email"] == "user@example.com"
        assert urls and urls[0].startswith("https://cursor.example/loginDeepControl?")
        stored = read_sdk_credentials()
        assert stored["apiKey"] == "key_minted_123"
