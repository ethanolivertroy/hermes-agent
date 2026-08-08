"""Cursor (subscription) provider profile.

Runs Hermes turns through the user's own Cursor subscription via the official
Cursor SDK bridge (https://github.com/cursor/sdk-bridge).  There is no REST
chat-completions endpoint — ``base_url`` is an internal marker scheme and the
transport is a local ``cursor-sdk-bridge`` subprocess driven by
``agent/cursor_bridge_client.py``.

Auth is the user's own ``CURSOR_API_KEY`` (cursor.com/dashboard → API Keys).
Usage bills to the user's Cursor plan; Hermes never proxies or resells
Cursor inference.
"""

from providers import register_provider
from providers.base import ProviderProfile


class CursorProfile(ProviderProfile):
    """Cursor subscription — local sdk.v1 bridge subprocess, no REST catalog."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch the account catalog through the bridge when it is installed.

        Spawning the bridge takes a few seconds of Node startup, so this only
        runs when a bridge binary is already resolvable AND an API key is
        available; otherwise it returns None immediately and callers fall
        back to the static list.
        """
        del base_url, timeout
        if not api_key:
            return None
        try:
            from agent.cursor_bridge_client import CursorBridgeClient
            from agent.cursor_bridge_transport import resolve_bridge_command

            if not resolve_bridge_command():
                return None
            client = CursorBridgeClient(api_key=api_key)
            try:
                models = client.list_models()
            finally:
                client.close()
            ids = [str(m.get("id") or "").strip() for m in models]
            return [m for m in ids if m] or None
        except Exception:
            return None


cursor = CursorProfile(
    name="cursor",
    aliases=("cursor-sdk", "cursor-agent"),
    display_name="Cursor",
    description="Cursor subscription (Composer + catalog via the Cursor SDK bridge)",
    signup_url="https://cursor.com/dashboard",
    api_mode="chat_completions",  # bridge subprocess uses chat_completions routing
    env_vars=("CURSOR_API_KEY",),
    base_url="sdkbridge://cursor",  # internal marker scheme, not a REST endpoint
    auth_type="api_key",
    supports_health_check=False,  # no /models REST probe — doctor skips it
    fallback_models=(
        "auto",
        "composer-2.5",
    ),
)

register_provider(cursor)
