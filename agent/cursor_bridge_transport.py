"""Cursor SDK bridge process management + Connect JSON transport.

Implements the `sdk.v1` bridge protocol from https://github.com/cursor/sdk-bridge:

* locate or install the ``cursor-sdk-bridge`` launcher,
* spawn it with ``CURSOR_API_KEY`` and parse the ``cursor-sdk-bridge ready``
  stderr discovery line,
* speak Connect over HTTP/1.1 with JSON encoding — unary RPCs are plain JSON
  POSTs, server streams use enveloped ``application/connect+json`` frames
  (1 flags byte + 4-byte big-endian length + payload).

The bridge is HTTP/1.1 only; classic gRPC clients do not work.  JSON encoding
is used so Hermes does not need generated protobuf stubs (proto3 has a
canonical JSON mapping and Connect servers accept it natively).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

READY_LINE_PREFIX = "cursor-sdk-bridge ready "
_STARTUP_TIMEOUT_SECONDS = 45.0
_SHUTDOWN_TIMEOUT_SECONDS = 5.0

# Default bridge version installed by `hermes model` when no bridge is found.
# Matches a released @cursor/sdk / cursor-sdk version that includes the
# custom-tools callback surface (landed in SDK 1.0.2x, Aug 2026).  Users can
# override via config.yaml `cursor_bridge.download_version`.
DEFAULT_BRIDGE_VERSION = "1.0.27"
# Primary distribution: GitHub release assets (ship with SHA256SUMS.txt so
# the install can be integrity-checked).  The downloads.cursor.com mirror
# documented in the sdk-bridge README is the fallback.
_GITHUB_RELEASE_URL_TEMPLATE = (
    "https://github.com/cursor/sdk-bridge/releases/download/v{version}/"
    "cursor-sdk-bridge-standalone-{os}-{arch}.tar.gz"
)
_GITHUB_SHASUMS_URL_TEMPLATE = (
    "https://github.com/cursor/sdk-bridge/releases/download/v{version}/SHA256SUMS.txt"
)
_MIRROR_URL_TEMPLATE = (
    "https://downloads.cursor.com/sdk-bridge/{version}/{os}/{arch}/"
    "cursor-sdk-bridge-package.tar.gz"
)


class CursorBridgeError(RuntimeError):
    """Raised for bridge process, transport, or Connect-level failures."""

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code


@dataclass
class BridgeEndpoint:
    """Where a running bridge listens and how to authenticate to it."""

    url: str
    auth_token: str
    server_version: str = ""
    pid: int | None = None
    workspace_ref: str = ""
    state_root: str = ""


def parse_ready_line(line: str) -> dict[str, Any] | None:
    """Parse one stderr line; return the discovery payload or None.

    Never log the returned payload verbatim — older bridges inline the auth
    token in the discovery JSON.
    """
    if not line.startswith(READY_LINE_PREFIX):
        return None
    raw = line[len(READY_LINE_PREFIX):].strip()
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise CursorBridgeError(f"invalid bridge discovery JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CursorBridgeError("bridge discovery payload is not a JSON object")
    return payload


def validate_discovery(payload: dict[str, Any]) -> None:
    if payload.get("schemaVersion") != 1:
        raise CursorBridgeError(
            f"unsupported bridge discovery schemaVersion={payload.get('schemaVersion')!r}"
        )
    if payload.get("transport") != "tcp":
        raise CursorBridgeError(
            f"unsupported bridge transport={payload.get('transport')!r}"
        )
    if payload.get("protocol") != "connect":
        raise CursorBridgeError(
            f"unsupported bridge protocol={payload.get('protocol')!r}"
        )


def endpoint_from_discovery(payload: dict[str, Any]) -> BridgeEndpoint:
    validate_discovery(payload)
    url = str(payload.get("url") or "").strip()
    if not url:
        host = str(payload.get("host") or "").strip()
        port = payload.get("port")
        if not host or not isinstance(port, int):
            raise CursorBridgeError("bridge discovery has no url/host/port")
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        url = f"http://{host}:{port}"

    token = str(payload.get("authToken") or "").strip()
    if not token:
        token_file = str(payload.get("authTokenFile") or "").strip()
        if not token_file:
            raise CursorBridgeError("bridge discovery has no authTokenFile")
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise CursorBridgeError(f"could not read bridge auth token file: {exc}") from exc
    if not token:
        raise CursorBridgeError("bridge auth token is empty")

    return BridgeEndpoint(
        url=url,
        auth_token=token,
        server_version=str(payload.get("serverVersion") or ""),
        pid=payload.get("pid") if isinstance(payload.get("pid"), int) else None,
        workspace_ref=str(payload.get("workspaceRef") or ""),
        state_root=str(payload.get("stateRoot") or ""),
    )


# ── Bridge binary resolution ──────────────────────────────────────────────


def bridge_install_dir() -> Path:
    """Hermes-managed bridge install location (profile-aware)."""
    return get_hermes_home() / "cursor-sdk-bridge"


def _launcher_name() -> str:
    return "cursor-sdk-bridge.cmd" if os.name == "nt" else "cursor-sdk-bridge"


def _bridge_from_cursor_sdk_wheel() -> str | None:
    """Locate the bridge bundled inside an installed ``cursor-sdk`` wheel."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("cursor_sdk")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for root in spec.submodule_search_locations:
        base = Path(root)
        for candidate in (
            base / "cursor-sdk-bridge" / "bin" / _launcher_name(),
            base / "_bridge" / "bin" / _launcher_name(),
            base / "bridge" / "bin" / _launcher_name(),
        ):
            if candidate.exists():
                return str(candidate)
        # Fall back to a shallow scan — wheel layout is not contractual.
        try:
            for candidate in base.glob(f"**/bin/{_launcher_name()}"):
                return str(candidate)
        except OSError:
            continue
    return None


def resolve_bridge_command(configured_command: str = "") -> str | None:
    """Resolve the bridge launcher path, or None when not installed.

    Resolution order:
      1. ``cursor_bridge.command`` from config.yaml (caller passes it in)
      2. the bridge embedded in an installed ``cursor-sdk`` PyPI wheel
      3. ``CURSOR_SDK_BRIDGE_BIN`` (the upstream adapter convention)
      4. ``cursor-sdk-bridge`` on PATH
      5. the Hermes-managed install under ``$HERMES_HOME/cursor-sdk-bridge/``
    """
    configured = (configured_command or "").strip()
    if configured:
        expanded = os.path.expanduser(configured)
        if shutil.which(expanded) or Path(expanded).exists():
            return expanded
        logger.warning("Configured cursor_bridge.command %r not found", configured)

    wheel_bridge = _bridge_from_cursor_sdk_wheel()
    if wheel_bridge:
        return wheel_bridge

    env_bin = os.getenv("CURSOR_SDK_BRIDGE_BIN", "").strip()
    if env_bin and Path(os.path.expanduser(env_bin)).exists():
        return os.path.expanduser(env_bin)

    on_path = shutil.which("cursor-sdk-bridge")
    if on_path:
        return on_path

    for managed in (
        bridge_install_dir() / "bin" / _launcher_name(),
        bridge_install_dir() / "cursor-sdk-bridge" / "bin" / _launcher_name(),
    ):
        if managed.exists():
            return str(managed)
    return None


def bridge_platform() -> tuple[str, str]:
    """Return the (os, arch) pair used by the bridge download URL."""
    import platform as _platform
    import sys

    if sys.platform == "win32":
        os_name = "win32"
    elif sys.platform == "darwin":
        os_name = "darwin"
    else:
        os_name = "linux"
    machine = _platform.machine().lower()
    arch = "arm64" if machine in {"arm64", "aarch64"} else "x64"
    if os_name == "win32":
        arch = "x64"  # win32 archives are x64-only
    return os_name, arch


def _fetch_url(url: str, timeout: float = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "hermes-cli"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _expected_archive_sha256(version: str, archive_name: str) -> str | None:
    """Fetch the release SHA256SUMS.txt and return the archive's digest."""
    try:
        sums = _fetch_url(
            _GITHUB_SHASUMS_URL_TEMPLATE.format(version=version), timeout=30
        ).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        return None
    for line in sums.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("*") == archive_name:
            return parts[0].lower()
    return None


def download_bridge(version: str = "", *, progress: bool = True) -> str:
    """Download and unpack the bridge archive into the Hermes-managed dir.

    Prefers the GitHub release asset (integrity-checked against the
    release's SHA256SUMS.txt) and falls back to the downloads.cursor.com
    mirror.  Returns the launcher path.  Called from setup flows
    (`hermes model`) — the runtime client never downloads implicitly.
    """
    import hashlib
    import tarfile
    import tempfile

    version = (version or DEFAULT_BRIDGE_VERSION).strip().lstrip("v")
    os_name, arch = bridge_platform()
    archive_name = f"cursor-sdk-bridge-standalone-{os_name}-{arch}.tar.gz"
    dest_root = bridge_install_dir()
    dest_root.mkdir(parents=True, exist_ok=True)

    if progress:
        print(f"  Downloading Cursor SDK bridge {version} ({os_name}/{arch})...")

    data: bytes | None = None
    errors: list[str] = []
    github_url = _GITHUB_RELEASE_URL_TEMPLATE.format(
        version=version, os=os_name, arch=arch
    )
    try:
        data = _fetch_url(github_url)
        expected = _expected_archive_sha256(version, archive_name)
        if expected:
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                raise CursorBridgeError(
                    f"bridge archive checksum mismatch: expected {expected}, got {actual}"
                )
            if progress:
                print("  ✓ SHA256 verified against the release manifest")
    except CursorBridgeError:
        raise
    except (urllib.error.URLError, OSError) as exc:
        errors.append(f"{github_url}: {exc}")
        data = None

    if data is None:
        mirror_url = _MIRROR_URL_TEMPLATE.format(version=version, os=os_name, arch=arch)
        try:
            data = _fetch_url(mirror_url)
        except (urllib.error.URLError, OSError) as exc:
            errors.append(f"{mirror_url}: {exc}")
            raise CursorBridgeError(
                "bridge download failed: " + "; ".join(errors)
            ) from exc

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp.write(data)
        archive_path = tmp.name

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # The archive unpacks to a single `cursor-sdk-bridge/` directory.
            tar.extractall(dest_root, filter="data")
    except (tarfile.TarError, OSError) as exc:
        raise CursorBridgeError(f"bridge archive extraction failed: {exc}") from exc
    finally:
        try:
            os.unlink(archive_path)
        except OSError:
            pass

    # The GitHub standalone archive is flat (bin/, manifest.json, proto/);
    # the packaged mirror archive nests everything under cursor-sdk-bridge/.
    bridge_root = None
    for candidate in (dest_root, dest_root / "cursor-sdk-bridge"):
        if (candidate / "manifest.json").exists():
            bridge_root = candidate
            break
    if bridge_root is None:
        raise CursorBridgeError(
            f"bridge manifest not found under {dest_root} after install"
        )
    try:
        manifest = json.loads((bridge_root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CursorBridgeError(f"bridge manifest unreadable after install: {exc}") from exc
    if manifest.get("protocol") != "sdk.v1":
        raise CursorBridgeError(
            f"installed bridge speaks protocol {manifest.get('protocol')!r}, expected 'sdk.v1'"
        )

    launcher = bridge_root / "bin" / _launcher_name()
    if not launcher.exists():
        raise CursorBridgeError(f"bridge launcher missing after install: {launcher}")
    if os.name != "nt":
        launcher.chmod(launcher.stat().st_mode | 0o755)
    if progress:
        print(f"  ✓ Installed Cursor SDK bridge → {launcher}")
    return str(launcher)


# ── Bridge process ────────────────────────────────────────────────────────


def _build_subprocess_env(api_key: str) -> dict[str, str]:
    # The bridge is a model-driving executor: it needs the Cursor credential
    # but must not inherit Tier-1 Hermes secrets (gateway bot tokens, etc.).
    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env(inherit_credentials=True)
    env["CURSOR_API_KEY"] = api_key
    env["CURSOR_SDK_CLIENT_LANGUAGE"] = "python"
    return env


class CursorBridgeProcess:
    """Owns one ``cursor-sdk-bridge`` child process."""

    def __init__(
        self,
        *,
        command: str,
        api_key: str,
        workspace: str,
        tool_callback_url: str = "",
        tool_callback_auth_token: str = "",
    ):
        self._command = command
        self._api_key = api_key
        self._workspace = str(Path(workspace).resolve())
        self._tool_callback_url = tool_callback_url
        self._tool_callback_auth_token = tool_callback_auth_token
        self._process: subprocess.Popen[str] | None = None
        self.endpoint: BridgeEndpoint | None = None

    @property
    def workspace(self) -> str:
        return self._workspace

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> BridgeEndpoint:
        argv = [self._command, "--workspace", self._workspace]
        if self._tool_callback_url:
            argv += ["--tool-callback-url", self._tool_callback_url]
            if self._tool_callback_auth_token:
                argv += ["--tool-callback-auth-token", self._tool_callback_auth_token]
        try:
            from hermes_cli._subprocess_compat import windows_hide_flags

            self._process = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_build_subprocess_env(self._api_key),
                creationflags=windows_hide_flags(),
            )
        except OSError as exc:
            raise CursorBridgeError(
                f"could not launch Cursor SDK bridge {self._command!r}: {exc}"
            ) from exc

        discovery = self._await_ready_line(self._process)
        endpoint = endpoint_from_discovery(discovery)
        self.endpoint = endpoint
        logger.info(
            "Cursor SDK bridge ready (pid=%s, version=%s)",
            endpoint.pid,
            endpoint.server_version,
        )
        return endpoint

    def _await_ready_line(self, process: subprocess.Popen[str]) -> dict[str, Any]:
        """Scan stderr for the discovery line; keep draining forever after.

        A full stderr pipe blocks the bridge, so the scanner thread never
        stops reading.
        """
        import queue as _queue

        found: _queue.Queue[dict[str, Any] | Exception] = _queue.Queue(maxsize=1)

        def scan() -> None:
            if process.stderr is None:
                found.put(CursorBridgeError("bridge process exposed no stderr pipe"))
                return
            diagnostics: list[str] = []
            for line in process.stderr:
                line = line.rstrip("\n")
                try:
                    payload = parse_ready_line(line)
                except CursorBridgeError as exc:
                    found.put(exc)
                    payload = None
                if payload is None:
                    if len(diagnostics) < 60:
                        diagnostics.append(line)
                    continue
                found.put(payload)
                for _ in process.stderr:  # drain so the bridge never blocks
                    pass
                return
            found.put(
                CursorBridgeError(
                    "Cursor SDK bridge exited before emitting its ready line. "
                    "Stderr tail:\n" + "\n".join(diagnostics[-20:])
                )
            )

        threading.Thread(target=scan, daemon=True, name="cursor-bridge-stderr").start()
        import queue as _queue

        try:
            result = found.get(timeout=_STARTUP_TIMEOUT_SECONDS)
        except _queue.Empty:
            self.stop()
            raise CursorBridgeError(
                f"timed out after {_STARTUP_TIMEOUT_SECONDS:.0f}s waiting for the "
                "Cursor SDK bridge ready line"
            ) from None
        if isinstance(result, Exception):
            self.stop()
            raise result
        return result

    def stop(self) -> None:
        process, self._process = self._process, None
        self.endpoint = None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
            except OSError:
                pass
        try:
            process.wait(timeout=1)
        except (subprocess.TimeoutExpired, OSError):
            pass


# ── Connect JSON transport ────────────────────────────────────────────────


class ConnectJsonTransport:
    """Connect-over-HTTP/1.1 client with JSON message encoding.

    Unary RPCs: ``POST {base}/sdk.v1.<Service>/<Method>`` with
    ``application/json``; errors arrive as non-200 with a Connect error JSON
    body (``{"code": ..., "message": ...}``).

    Server streams: ``application/connect+json`` enveloped frames.  Each frame
    is 1 flags byte + 4-byte big-endian payload length + payload.  A frame
    with flags bit ``0x02`` is the JSON EndStreamResponse (holds ``error``
    when the stream failed).
    """

    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url.rstrip("/")
        self._token = auth_token

    def _request(self, path: str, content_type: str, body: bytes) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method="POST",
            headers={
                "Content-Type": content_type,
                "Authorization": f"Bearer {self._token}",
                "Connect-Protocol-Version": "1",
            },
        )

    @staticmethod
    def _raise_connect_error(raw: bytes, http_status: int | None = None) -> None:
        code = None
        message = raw.decode("utf-8", "replace")[:2000]
        try:
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict):
                code = payload.get("code")
                message = str(payload.get("message") or message)
        except ValueError:
            pass
        prefix = f"HTTP {http_status} " if http_status else ""
        raise CursorBridgeError(f"{prefix}connect error [{code}]: {message}", code=code)

    def unary(
        self,
        service: str,
        method: str,
        request: dict[str, Any],
        *,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        body = json.dumps(request).encode("utf-8")
        req = self._request(f"/sdk.v1.{service}/{method}", "application/json", body)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as reply:
                raw = reply.read()
        except urllib.error.HTTPError as err:
            self._raise_connect_error(err.read(), http_status=err.code)
        except (urllib.error.URLError, OSError) as err:
            raise CursorBridgeError(f"{service}/{method}: {err}") from None
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise CursorBridgeError(f"{service}/{method}: invalid JSON response: {exc}") from exc
        return payload if isinstance(payload, dict) else {}

    def server_stream(
        self,
        service: str,
        method: str,
        request: dict[str, Any],
        *,
        read_timeout: float = 90.0,
        deadline: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield stream messages as dicts until the EndStreamResponse frame.

        ``read_timeout`` bounds a single socket read; the bridge emits
        keepalives every ~15s so a quiet-but-alive stream never trips it.
        ``deadline`` (monotonic timestamp) bounds the whole stream.
        """
        payload = json.dumps(request).encode("utf-8")
        body = struct.pack(">BI", 0, len(payload)) + payload
        req = self._request(f"/sdk.v1.{service}/{method}", "application/connect+json", body)
        try:
            reply = urllib.request.urlopen(req, timeout=read_timeout)
        except urllib.error.HTTPError as err:
            self._raise_connect_error(err.read(), http_status=err.code)
            return  # unreachable; keeps type-checkers happy
        except (urllib.error.URLError, OSError) as err:
            raise CursorBridgeError(f"{service}/{method}: {err}") from None

        with reply:
            while True:
                if deadline is not None and time.monotonic() > deadline:
                    raise CursorBridgeError(f"{service}/{method}: stream deadline exceeded")
                header = _read_exact(reply, 5, what=f"{service}/{method} frame header")
                flags, length = struct.unpack(">BI", header)
                frame = _read_exact(reply, length, what=f"{service}/{method} frame body")
                if flags & 0x02:
                    end = json.loads(frame) if frame else {}
                    error = end.get("error") if isinstance(end, dict) else None
                    if error:
                        self._raise_connect_error(json.dumps(error).encode("utf-8"))
                    return
                if not frame:
                    continue
                try:
                    message = json.loads(frame.decode("utf-8"))
                except ValueError as exc:
                    raise CursorBridgeError(
                        f"{service}/{method}: invalid JSON stream frame: {exc}"
                    ) from exc
                if isinstance(message, dict):
                    yield message


def _read_exact(stream: Any, count: int, *, what: str) -> bytes:
    chunks = b""
    while len(chunks) < count:
        try:
            chunk = stream.read(count - len(chunks))
        except OSError as exc:
            raise CursorBridgeError(f"{what}: stream read failed: {exc}") from None
        if not chunk:
            raise CursorBridgeError(f"{what}: stream ended before EndStreamResponse")
        chunks += chunk
    return chunks
