"""Integration invariants for the Cursor subscription provider.

These assert how the provider's surfaces must relate to each other (profile
↔ registry ↔ overlay ↔ picker ↔ client marker), not snapshots of catalog
data that is expected to change.
"""

from agent.cursor_bridge_client import BRIDGE_MARKER_BASE_URL
from hermes_cli.cursor_cloud import (
    _strip_url_credentials,
    agent_url,
    run_slash,
)


class TestProviderRegistration:
    def test_profile_registered_with_marker_base_url(self):
        from providers import get_provider_profile

        profile = get_provider_profile("cursor")
        assert profile is not None
        assert profile.base_url == BRIDGE_MARKER_BASE_URL
        assert profile.auth_type == "api_key"
        assert "CURSOR_API_KEY" in profile.env_vars
        # No REST /models endpoint exists — doctor must skip the probe.
        assert profile.supports_health_check is False
        assert len(profile.fallback_models) >= 1

    def test_auth_registry_auto_wired_from_profile(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("cursor")
        assert pconfig is not None
        assert pconfig.auth_type == "api_key"
        assert pconfig.inference_base_url == BRIDGE_MARKER_BASE_URL
        assert "CURSOR_API_KEY" in pconfig.api_key_env_vars

    def test_env_var_exposed_in_optional_env_vars(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        entry = OPTIONAL_ENV_VARS.get("CURSOR_API_KEY")
        assert entry is not None
        assert entry.get("password") is True
        assert entry.get("category") == "provider"

    def test_identity_overlay_matches_marker(self):
        from hermes_cli.providers import get_provider, normalize_provider

        pdef = get_provider("cursor", allow_network=False)
        assert pdef is not None
        assert pdef.base_url == BRIDGE_MARKER_BASE_URL
        assert pdef.transport == "openai_chat"
        # Aliases resolve to the canonical id.
        assert normalize_provider("cursor-sdk") == "cursor"
        assert normalize_provider("cursor-agent") == "cursor"

    def test_model_picker_lists_cursor(self):
        from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_MODELS

        assert any(p.slug == "cursor" for p in CANONICAL_PROVIDERS)
        assert len(_PROVIDER_MODELS.get("cursor", [])) >= 1

    def test_default_config_has_cursor_bridge_section(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        section = DEFAULT_CONFIG.get("cursor_bridge")
        assert isinstance(section, dict)
        assert section.get("tool_mode") in {"loop", "harness"}
        # Loop mode must be the shipped default: it keeps Hermes approvals
        # and budget in charge of every tool call.
        assert section.get("tool_mode") == "loop"
        assert section.get("builtin_tools") is False

    def test_runtime_client_routes_on_marker(self):
        # The runtime routes to CursorBridgeClient on provider name OR the
        # marker scheme — both must stay in sync with the profile.
        from agent.cursor_bridge_client import CursorBridgeClient

        client = CursorBridgeClient(api_key="k")
        assert client.base_url == BRIDGE_MARKER_BASE_URL
        assert BRIDGE_MARKER_BASE_URL.startswith("sdkbridge://")


class TestSlashCommand:
    def test_registered_in_command_registry(self):
        from hermes_cli.commands import resolve_command

        command = resolve_command("cursor")
        assert command is not None
        assert command.name == "cursor"
        assert "handoff" in command.subcommands
        assert not command.cli_only  # available from messaging platforms

    def test_help_and_unknown_verbs(self):
        help_text = run_slash("")
        assert "/cursor handoff" in help_text
        unknown = run_slash("frobnicate")
        assert "Unknown /cursor subcommand" in unknown

    def test_agent_url_shape(self):
        assert agent_url("bc-abc") == "https://cursor.com/agents?id=bc-abc"


class TestSdkLoginRuntimeResolution:
    def test_runtime_credentials_fall_back_to_sdk_login_store(self, tmp_path, monkeypatch):
        """`hermes chat --provider cursor` must work after `hermes cursor login`.

        The runtime gate in runtime_provider.py fails resolution when the
        shared secret resolver returns empty — so the resolver itself must
        consult the SDK login store, not just the bridge client.
        """
        from pathlib import Path

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)

        from agent.cursor_sdk_auth import save_sdk_credentials
        from hermes_cli.auth import resolve_api_key_provider_credentials

        save_sdk_credentials(backend_url="https://api2.cursor.sh", api_key="key_login")
        creds = resolve_api_key_provider_credentials("cursor")
        assert creds["api_key"] == "key_login"
        assert creds["source"] == "cursor_sdk_login"


class TestRepoCredentialStripping:
    def test_token_stripped_from_https_remote(self):
        url = "https://x-access-token:ghs_secret123@github.com/org/repo"
        assert _strip_url_credentials(url) == "https://github.com/org/repo"

    def test_plain_https_unchanged(self):
        url = "https://github.com/org/repo.git"
        assert _strip_url_credentials(url) == url

    def test_scp_style_remote_unchanged(self):
        url = "git@github.com:org/repo.git"
        assert _strip_url_credentials(url) == url
