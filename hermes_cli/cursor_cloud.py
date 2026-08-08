"""Cursor Cloud Agent handoff — `hermes cursor <verb>` and the /cursor command.

Phase 2 of the Cursor subscription integration: hand a task to a **durable
Cursor cloud agent** running on Cursor-managed infrastructure.  The agent is
visible (and can be taken over) at cursor.com/agents, in the Cursor IDE's
Agents window, and on mobile, while Hermes can keep watching or prompting the
same agent.

Everything goes through the same official sdk.v1 bridge as the `cursor`
model provider (github.com/cursor/sdk-bridge) using the user's own
``CURSOR_API_KEY`` — billing lands on their Cursor plan.

Verbs:

    hermes cursor handoff "<prompt>" [--repo URL] [--ref REF] [--model M] [--pr] [--wait]
    hermes cursor send <agent-id> "<prompt>" [--wait]
    hermes cursor status [agent-id]
    hermes cursor runs [agent-id]
    hermes cursor pull [agent-id]
    hermes cursor watch [agent-id]
    hermes cursor open [agent-id]
    hermes cursor list

Mirroring runs into a live Hermes session reuses the existing
background-process machinery rather than a bespoke watcher: run
``hermes cursor watch <id>`` via ``terminal(background=true,
notify_on_complete=true)`` and the gateway's completion notification frames
the result into the conversation (cron-style header/footer, alternation-safe).
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home

AGENTS_DASHBOARD_URL = "https://cursor.com/agents"

_TERMINAL_STATUSES = {
    "RUN_LIFECYCLE_STATUS_FINISHED",
    "RUN_LIFECYCLE_STATUS_ERROR",
    "RUN_LIFECYCLE_STATUS_CANCELLED",
    "RUN_LIFECYCLE_STATUS_EXPIRED",
}

_STATUS_LABELS = {
    "RUN_LIFECYCLE_STATUS_CREATING": "creating",
    "RUN_LIFECYCLE_STATUS_RUNNING": "running",
    "RUN_LIFECYCLE_STATUS_FINISHED": "finished",
    "RUN_LIFECYCLE_STATUS_ERROR": "error",
    "RUN_LIFECYCLE_STATUS_CANCELLED": "cancelled",
    "RUN_LIFECYCLE_STATUS_EXPIRED": "expired",
}


def agent_url(agent_id: str) -> str:
    return f"{AGENTS_DASHBOARD_URL}?id={agent_id}"


def _status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, status or "unknown")


# ── Handoff state sidecar ─────────────────────────────────────────────────


def _state_path() -> Path:
    return get_hermes_home() / "cursor_cloud_agents.json"


def _load_state() -> list[dict[str, Any]]:
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _remember_agent(agent_id: str, repo: str, prompt: str) -> None:
    entries = [e for e in _load_state() if e.get("agent_id") != agent_id]
    entries.append(
        {
            "agent_id": agent_id,
            "repo": repo,
            "prompt": prompt[:200],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    entries = entries[-50:]
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    except OSError:
        pass


def _default_agent_id() -> str:
    entries = _load_state()
    return str(entries[-1]["agent_id"]) if entries else ""


# ── Bridge session ────────────────────────────────────────────────────────


def _resolve_api_key() -> str:
    from agent.cursor_sdk_auth import resolve_cursor_api_key

    key, _source = resolve_cursor_api_key()
    if not key:
        raise RuntimeError(
            "No Cursor credential found. Run `hermes cursor login` (browser "
            "login on your Cursor account), or add CURSOR_API_KEY to "
            "~/.hermes/.env (cursor.com/dashboard → API Keys)."
        )
    return key


class CursorCloudSession:
    """One bridge process + transport for a batch of cloud-agent operations."""

    def __init__(self, api_key: str | None = None, workspace: str | None = None):
        import os

        self._api_key = api_key or _resolve_api_key()
        self._workspace = workspace or os.getcwd()
        self._process = None
        self._transport = None

    def __enter__(self) -> "CursorCloudSession":
        from hermes_cli.config import load_config
        from agent.cursor_bridge_transport import (
            ConnectJsonTransport,
            CursorBridgeError,
            CursorBridgeProcess,
            resolve_bridge_command,
        )

        settings = load_config().get("cursor_bridge") or {}
        command = resolve_bridge_command(str(settings.get("command") or ""))
        if not command:
            raise CursorBridgeError(
                "Cursor SDK bridge not found. Run `hermes model` and pick Cursor "
                "to install it, or `pip install cursor-sdk`."
            )
        self._process = CursorBridgeProcess(
            command=command, api_key=self._api_key, workspace=self._workspace
        )
        endpoint = self._process.start()
        self._transport = ConnectJsonTransport(endpoint.url, endpoint.auth_token)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._process is not None:
            self._process.stop()
            self._process = None
        self._transport = None

    @property
    def transport(self):
        if self._transport is None:
            raise RuntimeError("CursorCloudSession is not started")
        return self._transport

    # ── operations ───────────────────────────────────────────────────────

    def create_cloud_agent(
        self,
        *,
        repo: str,
        ref: str = "",
        model: str = "",
        auto_create_pr: bool = False,
    ) -> str:
        repo_entry: dict[str, Any] = {"url": repo}
        if ref:
            repo_entry["startingRef"] = ref
        options: dict[str, Any] = {
            "name": "hermes-handoff",
            "cloud": {
                "env": {"type": "CLOUD_ENVIRONMENT_TYPE_CLOUD"},
                "repos": [repo_entry],
                "metadata": {"created_by": "hermes-agent"},
            },
        }
        if auto_create_pr:
            options["cloud"]["autoCreatePr"] = True
        if model:
            options["model"] = {"id": model}
        created = self.transport.unary(
            "SdkAgentService", "CreateAgent", {"options": options}, timeout=120.0
        )
        agent_id = str(created.get("agentId") or "")
        if not agent_id:
            raise RuntimeError("CreateAgent returned no agentId")
        return agent_id

    def send_detached(self, agent_id: str, prompt: str) -> None:
        """Start a run and return once it is accepted; do not wait for it.

        Dropping a Send stream does NOT cancel the run (sdk-bridge
        docs/streaming.md) — the cloud agent keeps executing and remains
        visible at cursor.com/agents.
        """
        started = threading.Event()
        failure: list[BaseException] = []

        def drain() -> None:
            try:
                stream = self.transport.server_stream(
                    "SdkAgentService",
                    "Send",
                    {"agentId": agent_id, "message": {"text": prompt}, "options": {}},
                )
                started.set()
                for _ in stream:
                    pass
            except BaseException as exc:  # surfaced via `failure` below
                failure.append(exc)
                started.set()

        thread = threading.Thread(target=drain, daemon=True, name="cursor-cloud-send")
        thread.start()
        started.wait(timeout=30.0)
        if failure:
            raise RuntimeError(f"cloud send failed: {failure[0]}")
        # Give the request a moment to register server-side before the
        # process (and with it the bridge) exits.
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            runs = self.list_runs(agent_id, limit=1)
            if runs:
                return
            time.sleep(1.0)

    def send_and_stream(
        self, agent_id: str, prompt: str, *, emit: Callable[[str], None]
    ) -> str:
        final = ""
        for message in self.transport.server_stream(
            "SdkAgentService",
            "Send",
            {"agentId": agent_id, "message": {"text": prompt}, "options": {}},
        ):
            final = _handle_stream_message(message, emit) or final
            if "done" in message:
                break
        return final

    def list_runs(self, agent_id: str, limit: int = 10) -> list[dict[str, Any]]:
        response = self.transport.unary(
            "SdkAgentService",
            "ListRuns",
            {"agentId": agent_id, "options": {"limit": limit}},
            timeout=60.0,
        )
        return [r for r in response.get("items") or [] if isinstance(r, dict)]

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        response = self.transport.unary(
            "SdkAgentService",
            "GetAgent",
            {"agentId": agent_id, "options": {}},
            timeout=60.0,
        )
        agent = response.get("agent")
        return agent if isinstance(agent, dict) else {}

    def list_agents(self, limit: int = 20) -> list[dict[str, Any]]:
        response = self.transport.unary(
            "SdkAgentService",
            "ListAgents",
            {"options": {"limit": limit, "runtime": "RUNTIME_CLOUD"}},
            timeout=60.0,
        )
        return [a for a in response.get("items") or [] if isinstance(a, dict)]

    def observe_run(
        self, run_id: str, *, emit: Callable[[str], None], after_offset: str = ""
    ) -> str:
        request: dict[str, Any] = {"runId": run_id}
        if after_offset:
            request["afterOffset"] = after_offset
        final = ""
        for message in self.transport.server_stream(
            "SdkAgentService", "ObserveRun", request
        ):
            final = _handle_stream_message(message, emit) or final
            if "done" in message:
                break
        return final


def _handle_stream_message(message: dict[str, Any], emit: Callable[[str], None]) -> str:
    """Render one RunStreamMessage; return final text when terminal."""
    result_env = message.get("result")
    if isinstance(result_env, dict):
        status = _status_label(str(result_env.get("status") or ""))
        run_result = result_env.get("result")
        text = ""
        if isinstance(run_result, dict):
            text = str(run_result.get("result") or "")
        emit(f"── run {status} ──")
        if text:
            emit(text)
        return text or f"(run {status}, no result text)"

    sdk_message = message.get("sdkMessage")
    if isinstance(sdk_message, dict):
        # Best-effort progress rendering; payload shapes follow the public
        # SDK message types and may evolve — unknown shapes are skipped.
        payload = sdk_message.get("message")
        if isinstance(payload, dict):
            text = payload.get("text")
            if isinstance(text, str) and text.strip():
                emit(text.strip())
    return ""


# ── Repo resolution ───────────────────────────────────────────────────────


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _strip_url_credentials(repo: str) -> str:
    """Remove userinfo (tokens!) from a git remote URL before it leaves Hermes.

    CI checkouts and credential-helper remotes often embed an access token
    (``https://x-access-token:TOKEN@github.com/...``). That token must never
    be sent to the cloud API or written to the handoff state file — cloud
    agents authenticate to the repo through the user's Cursor GitHub
    connection, not through this URL.
    """
    from urllib.parse import urlsplit, urlunsplit

    if "@" not in repo:
        return repo
    try:
        parts = urlsplit(repo)
    except ValueError:
        return repo
    if not parts.scheme or "@" not in parts.netloc:
        return repo  # scp-style git@host:path remotes carry no secret
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def resolve_repo_and_ref(repo: str = "", ref: str = "") -> tuple[str, str]:
    """Fill repo/ref from the current git checkout when omitted."""
    if not repo:
        repo = _git_output(["remote", "get-url", "origin"])
    if not ref:
        branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
        if branch and branch != "HEAD":
            ref = branch
    return _strip_url_credentials(repo), ref


# ── Verb implementations (return display strings) ─────────────────────────


def do_handoff(
    prompt: str,
    *,
    repo: str = "",
    ref: str = "",
    model: str = "",
    auto_pr: bool = False,
    wait: bool = False,
    emit: Callable[[str], None] = print,
) -> str:
    if not prompt.strip():
        return "Usage: hermes cursor handoff \"<task prompt>\" [--repo URL] [--ref REF]"
    repo, ref = resolve_repo_and_ref(repo, ref)
    if not repo:
        return (
            "No repository detected. Pass --repo <git-url> or run from a git "
            "checkout with an 'origin' remote (cloud agents work on a repo clone)."
        )

    with CursorCloudSession() as session:
        agent_id = session.create_cloud_agent(
            repo=repo, ref=ref, model=model, auto_create_pr=auto_pr
        )
        _remember_agent(agent_id, repo, prompt)
        header = (
            f"Cloud agent created: {agent_id}\n"
            f"  repo: {repo}" + (f" @ {ref}" if ref else "") + "\n"
            f"  take over anytime: {agent_url(agent_id)}"
        )
        if wait:
            emit(header)
            emit("── streaming run (Ctrl+C detaches; the run keeps going) ──")
            final = session.send_and_stream(agent_id, prompt, emit=emit)
            return f"{header}\n\nFinal result:\n{final}"
        session.send_detached(agent_id, prompt)
        return (
            f"{header}\n"
            f"  run started — it keeps going even after this command exits.\n"
            f"  check on it: hermes cursor status {agent_id}\n"
            f"  follow it:   hermes cursor watch {agent_id}"
        )


def do_send(agent_id: str, prompt: str, *, wait: bool = False,
            emit: Callable[[str], None] = print) -> str:
    agent_id = agent_id or _default_agent_id()
    if not agent_id:
        return "No agent id given and no previous handoff found."
    if not prompt.strip():
        return "Usage: hermes cursor send <agent-id> \"<follow-up prompt>\""
    with CursorCloudSession() as session:
        if wait:
            final = session.send_and_stream(agent_id, prompt, emit=emit)
            return f"Final result:\n{final}"
        session.send_detached(agent_id, prompt)
        return (
            f"Follow-up run started on {agent_id}.\n"
            f"  take over anytime: {agent_url(agent_id)}"
        )


def do_status(agent_id: str = "") -> str:
    agent_id = agent_id or _default_agent_id()
    if not agent_id:
        return "No agent id given and no previous handoff found."
    with CursorCloudSession() as session:
        agent = session.get_agent(agent_id)
        runs = session.list_runs(agent_id, limit=3)

    lines = [f"Cursor cloud agent {agent_id}"]
    if agent:
        name = agent.get("name") or ""
        summary = agent.get("summary") or ""
        if name:
            lines.append(f"  name: {name}")
        if summary:
            lines.append(f"  summary: {summary}")
    lines.append(f"  dashboard: {agent_url(agent_id)}")
    if runs:
        lines.append("  recent runs:")
        for run in runs:
            status = _status_label(str(run.get("status") or ""))
            run_id = run.get("runId") or "?"
            lines.append(f"    {run_id}: {status}")
    else:
        lines.append("  no runs recorded yet")
    return "\n".join(lines)


def do_runs(agent_id: str = "") -> str:
    agent_id = agent_id or _default_agent_id()
    if not agent_id:
        return "No agent id given and no previous handoff found."
    with CursorCloudSession() as session:
        runs = session.list_runs(agent_id, limit=20)
    if not runs:
        return f"No runs recorded for {agent_id}."
    lines = [f"Runs for {agent_id}:"]
    for run in runs:
        status = _status_label(str(run.get("status") or ""))
        preview = str(run.get("result") or "").strip().replace("\n", " ")[:80]
        lines.append(f"  {run.get('runId', '?')}: {status}" + (f" — {preview}" if preview else ""))
    return "\n".join(lines)


def do_pull(agent_id: str = "") -> str:
    agent_id = agent_id or _default_agent_id()
    if not agent_id:
        return "No agent id given and no previous handoff found."
    with CursorCloudSession() as session:
        runs = session.list_runs(agent_id, limit=1)
    if not runs:
        return f"No runs recorded for {agent_id}."
    run = runs[0]
    status = _status_label(str(run.get("status") or ""))
    text = str(run.get("result") or "").strip()
    header = f"Latest run on {agent_id} ({run.get('runId', '?')}): {status}"
    return f"{header}\n\n{text}" if text else f"{header}\n(no result text yet)"


def do_watch(agent_id: str = "", *, emit: Callable[[str], None] = print) -> str:
    """Follow the latest run to completion (the mirroring primitive).

    From a live Hermes session, run this via ``terminal(background=true,
    notify_on_complete=true)`` — the existing background-process
    notification machinery frames the completed output back into the
    conversation without breaking message-role alternation.
    """
    agent_id = agent_id or _default_agent_id()
    if not agent_id:
        return "No agent id given and no previous handoff found."
    with CursorCloudSession() as session:
        runs = session.list_runs(agent_id, limit=1)
        if not runs:
            return f"No runs recorded for {agent_id}."
        run = runs[0]
        run_id = str(run.get("runId") or "")
        status = str(run.get("status") or "")
        emit(f"── Cursor cloud run {run_id} on {agent_id} ──")
        if status in _TERMINAL_STATUSES:
            text = str(run.get("result") or "").strip()
            emit(f"── run {_status_label(status)} (already terminal) ──")
            return text or f"(run {_status_label(status)}, no result text)"
        final = session.observe_run(run_id, emit=emit)
        return final


def do_list() -> str:
    with CursorCloudSession() as session:
        agents = session.list_agents(limit=20)
    if not agents:
        return "No cloud agents found for this account."
    lines = ["Cursor cloud agents (newest first):"]
    for agent in agents:
        agent_id = agent.get("agentId") or "?"
        name = agent.get("name") or ""
        summary = str(agent.get("summary") or "").replace("\n", " ")[:60]
        lines.append(f"  {agent_id}  {name}" + (f" — {summary}" if summary else ""))
    return "\n".join(lines)


def do_open(agent_id: str = "") -> str:
    agent_id = agent_id or _default_agent_id()
    url = agent_url(agent_id) if agent_id else AGENTS_DASHBOARD_URL
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception:
        pass
    return f"Cursor Agents dashboard: {url}"


def do_login(*, emit: Callable[[str], None] = print) -> str:
    """Interactive browser login — mints a Cursor user API key for Hermes.

    The URL can be opened on any device that is signed in to cursor.com;
    only this process can redeem it (device-code-style PKCE).
    """
    from agent.cursor_sdk_auth import CursorAuthError, login, read_sdk_credentials

    existing = read_sdk_credentials()
    if existing:
        who = existing.get("email") or "your Cursor account"
        emit(f"Already logged in as {who} (re-running will mint a fresh key).")

    def show_url(url: str) -> None:
        emit("Open this URL in a browser signed in to cursor.com:")
        emit(f"  {url}")

    try:
        result = login(on_login_url=show_url, on_status=emit)
    except CursorAuthError as exc:
        return str(exc)
    who = result.get("email") or "your Cursor account"
    expires_days = max(
        0, int((result["apiKeyExpiresAtMs"] / 1000 - time.time()) / 86400)
    )
    return (
        f"✓ Logged in as {who}. Minted a user API key (visible and revocable "
        f"in the Cursor dashboard, expires in ~{expires_days} days), stored "
        f"at {result['path']}.\n"
        f"Hermes now uses it automatically — try `hermes chat --provider cursor`."
    )


def do_logout() -> str:
    from agent.cursor_sdk_auth import clear_sdk_credentials, sdk_auth_path

    if clear_sdk_credentials():
        return (
            f"Removed {sdk_auth_path()}. The minted key stays valid until it "
            "expires — revoke it in the Cursor dashboard's API-keys list if needed."
        )
    return "No stored Cursor SDK login found."


# ── Slash command + argparse surfaces ─────────────────────────────────────

_HELP = """/cursor — hand tasks to a durable Cursor cloud agent (your Cursor sub)

  /cursor login                  browser login (mints a Cursor user API key)
  /cursor logout                 forget the stored SDK login
  /cursor handoff <prompt>       start a cloud agent on this repo
  /cursor send [id] <prompt>     follow-up prompt to an agent
  /cursor status [id]            agent + recent run status
  /cursor runs [id]              list runs
  /cursor pull [id]              fetch the latest run's result
  /cursor watch [id]             follow the active run to completion
  /cursor list                   list your cloud agents
  /cursor open [id]              open cursor.com/agents

The agent stays visible (and can be taken over) at cursor.com/agents, in the
Cursor IDE Agents window, and on mobile. Usage bills to your Cursor plan."""


def run_slash(rest: str, *, emit: Callable[[str], None] = print) -> str:
    """Shared /cursor dispatcher for CLI and gateway. Returns display text."""
    import shlex

    rest = (rest or "").strip()
    if not rest or rest in {"help", "--help", "-h"}:
        return _HELP
    try:
        tokens = shlex.split(rest)
    except ValueError:
        tokens = rest.split()
    verb, args = tokens[0].lower(), tokens[1:]

    def _id_and_text(items: list[str]) -> tuple[str, str]:
        if items and (items[0].startswith("bc-") or items[0].startswith("agent-")):
            return items[0], " ".join(items[1:])
        return "", " ".join(items)

    try:
        if verb == "login":
            return do_login(emit=emit)
        if verb == "logout":
            return do_logout()
        if verb == "handoff":
            return do_handoff(" ".join(args), emit=emit)
        if verb == "send":
            agent_id, text = _id_and_text(args)
            return do_send(agent_id, text, emit=emit)
        if verb == "status":
            return do_status(args[0] if args else "")
        if verb == "runs":
            return do_runs(args[0] if args else "")
        if verb == "pull":
            return do_pull(args[0] if args else "")
        if verb == "watch":
            return do_watch(args[0] if args else "", emit=emit)
        if verb == "list":
            return do_list()
        if verb == "open":
            return do_open(args[0] if args else "")
    except KeyboardInterrupt:
        return "(detached — the cloud run keeps going; check cursor.com/agents)"
    except Exception as exc:
        return f"cursor cloud error: {exc}"
    return f"Unknown /cursor subcommand: {verb}\n\n{_HELP}"


def cmd_cursor(args) -> int:
    """``hermes cursor <verb>`` argparse handler."""
    verb = getattr(args, "cursor_command", None) or "help"
    try:
        if verb == "login":
            output = do_login()
        elif verb == "logout":
            output = do_logout()
        elif verb == "handoff":
            output = do_handoff(
                " ".join(args.prompt or []),
                repo=getattr(args, "repo", "") or "",
                ref=getattr(args, "ref", "") or "",
                model=getattr(args, "model", "") or "",
                auto_pr=bool(getattr(args, "pr", False)),
                wait=bool(getattr(args, "wait", False)),
            )
        elif verb == "send":
            output = do_send(
                getattr(args, "agent_id", "") or "",
                " ".join(args.prompt or []),
                wait=bool(getattr(args, "wait", False)),
            )
        elif verb == "status":
            output = do_status(getattr(args, "agent_id", "") or "")
        elif verb == "runs":
            output = do_runs(getattr(args, "agent_id", "") or "")
        elif verb == "pull":
            output = do_pull(getattr(args, "agent_id", "") or "")
        elif verb == "watch":
            output = do_watch(getattr(args, "agent_id", "") or "")
        elif verb == "list":
            output = do_list()
        elif verb == "open":
            output = do_open(getattr(args, "agent_id", "") or "")
        else:
            output = _HELP
    except KeyboardInterrupt:
        print("(detached — the cloud run keeps going; check cursor.com/agents)")
        return 130
    except Exception as exc:
        print(f"cursor cloud error: {exc}")
        return 1
    if output:
        print(output)
    return 0
