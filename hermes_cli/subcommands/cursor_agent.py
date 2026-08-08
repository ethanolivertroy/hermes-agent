"""``hermes cursor`` subcommand parser.

Cloud-agent handoff on the user's own Cursor subscription — see
``hermes_cli/cursor_cloud.py`` for the implementation.
"""

from __future__ import annotations

from typing import Callable


def build_cursor_parser(subparsers, *, cmd_cursor: Callable) -> None:
    """Attach the ``cursor`` subcommand to ``subparsers``."""
    cursor_parser = subparsers.add_parser(
        "cursor",
        help="Hand tasks to a Cursor cloud agent (your own Cursor subscription)",
        description=(
            "Create and follow durable Cursor cloud agents through the official "
            "Cursor SDK bridge. Agents are visible and can be taken over at "
            "cursor.com/agents, in the Cursor IDE Agents window, and on mobile. "
            "Requires CURSOR_API_KEY in ~/.hermes/.env; usage bills to your "
            "Cursor plan."
        ),
    )
    cursor_sub = cursor_parser.add_subparsers(dest="cursor_command")

    cursor_sub.add_parser(
        "login",
        help="Browser login to your Cursor account (mints a user API key)",
    )
    cursor_sub.add_parser(
        "logout", help="Forget the stored Cursor SDK login on this machine"
    )

    handoff = cursor_sub.add_parser(
        "handoff", help="Start a cloud agent on a repository with a task prompt"
    )
    handoff.add_argument("prompt", nargs="+", help="Task prompt for the cloud agent")
    handoff.add_argument("--repo", default="", help="Git repo URL (default: origin remote)")
    handoff.add_argument("--ref", default="", help="Starting ref (default: current branch)")
    handoff.add_argument("--model", default="", help="Model id (default: server-resolved)")
    handoff.add_argument("--pr", action="store_true", help="Auto-create a PR when done")
    handoff.add_argument(
        "--wait", action="store_true", help="Stream the run instead of detaching"
    )

    send = cursor_sub.add_parser("send", help="Send a follow-up prompt to an agent")
    send.add_argument("agent_id", help="Cloud agent id (bc-...)")
    send.add_argument("prompt", nargs="+", help="Follow-up prompt")
    send.add_argument("--wait", action="store_true", help="Stream the run instead of detaching")

    for verb, help_text in (
        ("status", "Show agent + recent run status"),
        ("runs", "List runs for an agent"),
        ("pull", "Print the latest run's result"),
        ("watch", "Follow the active run to completion"),
        ("open", "Open the cursor.com/agents dashboard"),
    ):
        sub = cursor_sub.add_parser(verb, help=help_text)
        sub.add_argument(
            "agent_id", nargs="?", default="", help="Cloud agent id (default: last handoff)"
        )

    cursor_sub.add_parser("list", help="List your Cursor cloud agents")

    cursor_parser.set_defaults(func=cmd_cursor)
