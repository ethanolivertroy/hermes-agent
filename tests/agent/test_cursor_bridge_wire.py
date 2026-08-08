"""Tests for the hand-rolled protobuf wire codec used by Cursor tool callbacks."""

import struct

import pytest

from agent.cursor_bridge_wire import (
    _encode_varint,
    decode_call_custom_tool_request,
    decode_struct,
    encode_call_custom_tool_response,
    encode_struct,
)


class TestStructRoundTrip:
    def test_empty(self):
        assert decode_struct(encode_struct({})) == {}

    def test_scalars(self):
        obj = {"s": "hello", "i": 42, "f": 2.5, "t": True, "x": False, "n": None}
        assert decode_struct(encode_struct(obj)) == obj

    def test_nested_and_lists(self):
        obj = {
            "list": [1, "two", None, True, {"deep": [3.5]}],
            "obj": {"inner": {"leaf": "v"}},
        }
        assert decode_struct(encode_struct(obj)) == obj

    def test_unicode_keys_and_values(self):
        obj = {"héllo": "wörld ∆", "emoji": "🎉"}
        assert decode_struct(encode_struct(obj)) == obj

    def test_integral_doubles_come_back_as_ints(self):
        decoded = decode_struct(encode_struct({"count": 7}))
        assert decoded["count"] == 7
        assert isinstance(decoded["count"], int)

    def test_non_integral_doubles_stay_floats(self):
        decoded = decode_struct(encode_struct({"ratio": 0.25}))
        assert decoded["ratio"] == 0.25
        assert isinstance(decoded["ratio"], float)

    def test_rejects_non_dict(self):
        with pytest.raises(TypeError):
            encode_struct(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_rejects_unencodable_value(self):
        with pytest.raises(TypeError):
            encode_struct({"bad": object()})


class TestCallCustomToolMessages:
    def _encode_request(
        self,
        tool_name: str = "",
        args: dict | None = None,
        tool_call_id: str | None = None,
        agent_id: str = "",
    ) -> bytes:
        out = b""
        if tool_name:
            payload = tool_name.encode()
            out += _encode_varint((1 << 3) | 2) + _encode_varint(len(payload)) + payload
        if args is not None:
            payload = encode_struct(args)
            out += _encode_varint((2 << 3) | 2) + _encode_varint(len(payload)) + payload
        if tool_call_id is not None:
            payload = tool_call_id.encode()
            out += _encode_varint((3 << 3) | 2) + _encode_varint(len(payload)) + payload
        if agent_id:
            payload = agent_id.encode()
            out += _encode_varint((4 << 3) | 2) + _encode_varint(len(payload)) + payload
        return out

    def test_full_request_decodes(self):
        raw = self._encode_request(
            tool_name="send_message",
            args={"text": "hi", "count": 2},
            tool_call_id="call-9",
            agent_id="agent-abc",
        )
        decoded = decode_call_custom_tool_request(raw)
        assert decoded == {
            "toolName": "send_message",
            "args": {"text": "hi", "count": 2},
            "toolCallId": "call-9",
            "agentId": "agent-abc",
        }

    def test_empty_request_yields_defaults(self):
        decoded = decode_call_custom_tool_request(b"")
        assert decoded["toolName"] == ""
        assert decoded["args"] == {}
        assert decoded["toolCallId"] is None
        assert decoded["agentId"] == ""

    def test_unknown_fields_are_skipped(self):
        raw = self._encode_request(tool_name="t", agent_id="a")
        # Append an unknown varint field (field 15) and a fixed64 field (16).
        raw += _encode_varint((15 << 3) | 0) + _encode_varint(12345)
        raw += _encode_varint((16 << 3) | 1) + struct.pack("<d", 1.0)
        decoded = decode_call_custom_tool_request(raw)
        assert decoded["toolName"] == "t"
        assert decoded["agentId"] == "a"

    def test_response_round_trips_through_request_struct_decoder(self):
        # The response is field 1 = Struct; reuse the request decoder's
        # Struct path by crafting a request whose field-2 payload matches.
        result = {"value": "ok", "meta": {"n": 1}}
        encoded = encode_call_custom_tool_response(result)
        # field 1, wiretype 2, then the struct payload
        tag = encoded[0]
        assert tag >> 3 == 1 and tag & 0x07 == 2
        length = encoded[1]
        assert decode_struct(encoded[2 : 2 + length]) == result

    def test_truncated_varint_raises(self):
        with pytest.raises(ValueError):
            decode_call_custom_tool_request(b"\xff")
