"""Interactive Cursor SDK login for Hermes (`hermes cursor login`).

Implements the same browser PKCE flow the Cursor SDK's ``Cursor.auth.login()``
ships (sdk 1.0.27+): generate a one-time ``verifier``, send the user's browser
to ``cursor.com/loginDeepControl`` with ``challenge = sha256(verifier)``, poll
``api2.cursor.sh/auth/poll`` until the browser completes, then use the session
token **once** to mint a named, expiring user API key via
``aiserver.v1.DashboardService/CreateUserApiKey`` — and drop the session
tokens.  The minted key is the only credential persisted.

Credentials are stored in the SDK's own store (``~/.cursor/sdk/auth.json``,
0600) so a login done through Hermes, the Cursor SDK, or any other adapter is
shared — this path is deliberately NOT ``HERMES_HOME``-scoped; it belongs to
the user's Cursor identity, not to a Hermes profile.

The login URL can be opened on ANY device logged into cursor.com (it is a
device-code-style flow): only the process holding the verifier can redeem it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import socket
import time
import urllib.error
import urllib.request
import uuid as uuid_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_WEBSITE_URL = "https://cursor.com"
DEFAULT_BACKEND_URL = "https://api2.cursor.sh"
# Matches the SDK's DEFAULT_LOGIN_API_KEY_TTL_MS: 90 days.
DEFAULT_API_KEY_TTL_MS = 90 * 24 * 60 * 60 * 1000

_POLL_MAX_ATTEMPTS = 150
_POLL_BASE_DELAY_S = 1.0
_POLL_MAX_DELAY_S = 10.0
_POLL_BACKOFF = 1.2
_MAX_CONSECUTIVE_ERRORS = 3


class CursorAuthError(RuntimeError):
    pass


def resolve_website_url(url: str = "") -> str:
    return (url or os.getenv("CURSOR_WEBSITE_URL", "") or DEFAULT_WEBSITE_URL).rstrip("/")


def resolve_backend_url(url: str = "") -> str:
    return (url or os.getenv("CURSOR_BACKEND_URL", "") or DEFAULT_BACKEND_URL).rstrip("/")


# ── Handshake ─────────────────────────────────────────────────────────────


@dataclass
class LoginHandshake:
    uuid: str
    verifier: str
    login_url: str


def create_login_handshake(website_url: str = "") -> LoginHandshake:
    website_url = resolve_website_url(website_url)
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    login_uuid = str(uuid_module.uuid4())
    login_url = (
        f"{website_url}/loginDeepControl?challenge={challenge}"
        f"&uuid={login_uuid}&mode=login&redirectTarget=sdk"
    )
    return LoginHandshake(uuid=login_uuid, verifier=verifier, login_url=login_url)


# ── Poll ──────────────────────────────────────────────────────────────────


def _is_route_not_found_body(body: bytes) -> bool:
    text = body.decode("utf-8", "replace").strip()
    if not text.startswith("{"):
        return False
    try:
        parsed = json.loads(text)
    except ValueError:
        return False
    message = parsed.get("message")
    return (
        isinstance(message, str)
        and message.startswith("Route ")
        and "not found" in message
    )


def poll_for_login_tokens(
    *,
    api_url: str,
    uuid: str,
    verifier: str,
    on_status: Callable[[str], None] | None = None,
    max_attempts: int = _POLL_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, str] | None:
    """Poll ``/auth/poll`` until the browser completes the login.

    POST with the verifier in the JSON body (it is redeemable — it must not
    reach a URL or access log).  Backends predating the POST route answer
    with route-not-found; those get one sticky fallback to GET.  A 404 with
    any other body means the login is still pending.

    Returns ``{"accessToken", "refreshToken"}`` or None on timeout/errors.
    """
    use_get = False
    consecutive_errors = 0
    for attempt in range(max_attempts):
        delay = min(_POLL_BASE_DELAY_S * (_POLL_BACKOFF ** attempt), _POLL_MAX_DELAY_S)
        try:
            if use_get:
                request = urllib.request.Request(
                    f"{api_url}/auth/poll?uuid={uuid}&verifier={verifier}",
                    method="GET",
                    headers={"Accept": "application/json"},
                )
            else:
                request = urllib.request.Request(
                    f"{api_url}/auth/poll",
                    method="POST",
                    data=json.dumps({"uuid": uuid, "verifier": verifier}).encode(),
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                )
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    raw = response.read()
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    body = err.read()
                    if not use_get and _is_route_not_found_body(body):
                        use_get = True
                        if on_status:
                            on_status(
                                "backend has no POST /auth/poll — falling back to GET"
                            )
                        continue
                    consecutive_errors = 0
                    sleep(delay)
                    continue
                raise
            consecutive_errors = 0
            result = json.loads(raw.decode("utf-8"))
            if (
                isinstance(result, dict)
                and isinstance(result.get("accessToken"), str)
                and isinstance(result.get("refreshToken"), str)
            ):
                return {
                    "accessToken": result["accessToken"],
                    "refreshToken": result["refreshToken"],
                }
            return None
        except Exception as exc:
            consecutive_errors += 1
            logger.debug("auth/poll attempt %d failed: %s", attempt, exc)
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                return None
            sleep(delay)
    return None


# ── Mint ──────────────────────────────────────────────────────────────────


def _dashboard_rpc(
    backend_url: str, method: str, payload: dict[str, Any], access_token: str
) -> dict[str, Any]:
    """Connect unary POST to aiserver.v1.DashboardService with JSON encoding."""
    request = urllib.request.Request(
        f"{backend_url}/aiserver.v1.DashboardService/{method}",
        method="POST",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
            "Connect-Protocol-Version": "1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", "replace")[:500]
        raise CursorAuthError(
            f"DashboardService/{method} failed (HTTP {err.code}): {body}"
        ) from err
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise CursorAuthError(f"DashboardService/{method}: invalid JSON response") from exc
    return parsed if isinstance(parsed, dict) else {}


def mint_user_api_key(
    *,
    backend_url: str,
    access_token: str,
    name: str,
    expires_at_ms: int | None = None,
) -> str:
    """Mint a named user API key; the session token is used once and dropped."""
    payload: dict[str, Any] = {"name": name}
    if expires_at_ms is not None:
        # proto3 JSON encodes int64 as a string.
        payload["expiresAt"] = str(int(expires_at_ms))
    response = _dashboard_rpc(backend_url, "CreateUserApiKey", payload, access_token)
    api_key = response.get("apiKey")
    if not isinstance(api_key, str) or not api_key:
        raise CursorAuthError(
            "Login succeeded but CreateUserApiKey returned no key — your team's "
            "settings may restrict user API keys; ask a team admin, or add an "
            "existing CURSOR_API_KEY to ~/.hermes/.env instead."
        )
    return api_key


def get_login_email(backend_url: str, access_token: str) -> str:
    """Best-effort GetMe for a friendly status line."""
    try:
        response = _dashboard_rpc(backend_url, "GetMe", {}, access_token)
        email = response.get("email")
        return email if isinstance(email, str) else ""
    except Exception:
        return ""


# ── Credential store (shared with the Cursor SDK) ─────────────────────────


def sdk_auth_path() -> Path:
    """The SDK's credential store: ``~/.cursor/sdk/auth.json``.

    Intentionally HOME-anchored, not HERMES_HOME-anchored: this is Cursor's
    own credential store, shared with `@cursor/sdk` / `cursor-sdk`, so one
    login covers every SDK consumer on the machine.
    """
    return Path.home() / ".cursor" / "sdk" / "auth.json"


def read_sdk_credentials() -> dict[str, Any] | None:
    """Return valid stored credentials, or None (missing/foreign/expired)."""
    try:
        parsed = json.loads(sdk_auth_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(parsed, dict) or parsed.get("version") != 1:
        return None
    api_key = parsed.get("apiKey")
    if not isinstance(api_key, str) or not api_key:
        return None
    expires = parsed.get("apiKeyExpiresAtMs")
    if isinstance(expires, (int, float)) and expires <= time.time() * 1000:
        return None
    return parsed


def save_sdk_credentials(
    *,
    backend_url: str,
    api_key: str,
    api_key_expires_at_ms: int | None = None,
    email: str = "",
) -> Path:
    path = sdk_auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    payload: dict[str, Any] = {
        "version": 1,
        "backendUrl": backend_url,
        "apiKey": api_key,
        "createdAtMs": int(time.time() * 1000),
    }
    if api_key_expires_at_ms is not None:
        payload["apiKeyExpiresAtMs"] = int(api_key_expires_at_ms)
    if email:
        payload["email"] = email
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.chmod(0o600)
    tmp_path.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def clear_sdk_credentials() -> bool:
    try:
        sdk_auth_path().unlink()
        return True
    except OSError:
        return False


# ── Top-level login ───────────────────────────────────────────────────────


def _default_api_key_name() -> str:
    try:
        host = socket.gethostname() or "unknown-host"
    except OSError:
        host = "unknown-host"
    return f"Hermes Agent ({host})"


def login(
    *,
    on_login_url: Callable[[str], None],
    on_status: Callable[[str], None] | None = None,
    api_key_name: str = "",
    api_key_ttl_ms: int = DEFAULT_API_KEY_TTL_MS,
    backend_url: str = "",
    website_url: str = "",
    open_browser: bool = True,
) -> dict[str, Any]:
    """Run the full interactive login; returns the stored credential payload.

    The caller receives the login URL via ``on_login_url`` and may show it
    anywhere — the user can complete it on another device.
    """
    backend_url = resolve_backend_url(backend_url)
    handshake = create_login_handshake(website_url)
    on_login_url(handshake.login_url)
    if open_browser and not os.getenv("NO_OPEN_BROWSER") and not os.getenv("SSH_CONNECTION"):
        try:
            import webbrowser

            webbrowser.open(handshake.login_url)
        except Exception:
            pass

    if on_status:
        on_status("Waiting for the browser login to complete...")
    tokens = poll_for_login_tokens(
        api_url=backend_url,
        uuid=handshake.uuid,
        verifier=handshake.verifier,
        on_status=on_status,
    )
    if tokens is None:
        raise CursorAuthError(
            "Login did not complete (timed out or was cancelled). Run "
            "`hermes cursor login` to try again."
        )

    expires_at_ms = int(time.time() * 1000) + int(api_key_ttl_ms)
    api_key = mint_user_api_key(
        backend_url=backend_url,
        access_token=tokens["accessToken"],
        name=api_key_name or _default_api_key_name(),
        expires_at_ms=expires_at_ms,
    )
    email = get_login_email(backend_url, tokens["accessToken"])
    save_sdk_credentials(
        backend_url=backend_url,
        api_key=api_key,
        api_key_expires_at_ms=expires_at_ms,
        email=email,
    )
    return {
        "apiKey": api_key,
        "email": email,
        "apiKeyExpiresAtMs": expires_at_ms,
        "path": str(sdk_auth_path()),
    }


def resolve_cursor_api_key() -> tuple[str, str]:
    """Resolve the Cursor credential: explicit env/.env key, else SDK login.

    Returns ``(api_key, source)`` where source is ``"env"``,
    ``"sdk_login"``, or ``("", "")`` when nothing usable exists.
    """
    env_key = os.getenv("CURSOR_API_KEY", "").strip()
    if env_key:
        return env_key, "env"
    try:
        from hermes_cli.config import get_env_value

        dotenv_key = (get_env_value("CURSOR_API_KEY") or "").strip()
        if dotenv_key:
            return dotenv_key, "env"
    except Exception:
        pass
    stored = read_sdk_credentials()
    if stored:
        return str(stored["apiKey"]), "sdk_login"
    return "", ""
