"""Async streaming clients for both dialects (bench-spec.md §3).

One request → one `RequestRecord` with the §3 timestamp taxonomy on the
client perf clock: t_send when the request coroutine fires, t_headers on
response headers, t_first_token on the first delta carrying visible text
(OpenAI: `content` OR `reasoning_content` — Qwen3 leads with thinking and
TTFT counts it; Anthropic: the first content_block_delta), t_done when the
stream closes. Token counts come from the server's usage payloads, never
from client-side counting.

`build_sender` binds dialect + options into the `send(prompt, ...)` shape
the load generators drive; loadgen.py never sees HTTP.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

import httpx

from cantollm.bench.records import RequestRecord
from cantollm.bench.workloads import Prompt


@dataclass(frozen=True)
class SendOptions:
    model: str
    dialect: str = "openai"          # "openai" | "anthropic"
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = True
    capture_text: bool = False


def build_sender(client: httpx.AsyncClient, options: SendOptions):
    """→ async send(prompt, cell_id, repeat, request_index, t_scheduled,
    excluded) -> (RequestRecord, text|None)."""
    dialect_fn = {
        "openai": _send_openai,
        "anthropic": _send_anthropic,
    }[options.dialect]

    async def send(
        prompt: Prompt,
        *,
        cell_id: str,
        repeat: int,
        request_index: int,
        t_scheduled: float | None = None,
        excluded: bool = False,
    ) -> tuple[RequestRecord, str | None]:
        record = RequestRecord(
            cell_id=cell_id,
            repeat=repeat,
            request_index=request_index,
            prompt_id=prompt.id,
            dialect=options.dialect,
            t_scheduled=t_scheduled,
            t_send=time.perf_counter(),
            excluded=excluded,
        )
        text: str | None = None
        try:
            text = await dialect_fn(client, options, prompt, record)
        except httpx.HTTPStatusError as e:
            record.error = f"http {e.response.status_code}"
            record.http_status = e.response.status_code
        except httpx.HTTPError as e:
            record.error = f"{type(e).__name__}: {e}"
        except asyncio.CancelledError:
            raise
        except Exception as e:  # parse bugs shouldn't kill the whole run
            record.error = f"{type(e).__name__}: {e}"
        return record.finalize(), (text if options.capture_text else None)

    return send


async def _send_openai(
    client: httpx.AsyncClient, options: SendOptions, prompt: Prompt,
    record: RequestRecord,
) -> str:
    body = {
        "model": options.model,
        "messages": _openai_messages(prompt),
        "max_tokens": options.max_tokens,
        "temperature": options.temperature,
        "top_p": options.top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": options.ignore_eos,
    }
    parts: list[str] = []
    async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        record.t_headers = time.perf_counter()
        record.http_status = resp.status_code
        if resp.status_code != 200:
            await resp.aread()
            raise httpx.HTTPStatusError("bench", request=resp.request, response=resp)

        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)

            usage = chunk.get("usage")
            if usage:
                record.input_tokens = usage.get("prompt_tokens", 0)
                record.output_tokens = usage.get("completion_tokens", 0)
                details = usage.get("completion_tokens_details") or {}
                record.reasoning_tokens = details.get("reasoning_tokens", 0)

            for choice in chunk.get("choices", ()):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    if record.t_first_token is None:
                        record.t_first_token = time.perf_counter()
                    parts.append(piece)
                if choice.get("finish_reason"):
                    record.finish_reason = choice["finish_reason"]
    record.t_done = time.perf_counter()
    return "".join(parts)


async def _send_anthropic(
    client: httpx.AsyncClient, options: SendOptions, prompt: Prompt,
    record: RequestRecord,
) -> str:
    body = {
        "model": options.model,
        "messages": _anthropic_messages(prompt),
        "max_tokens": options.max_tokens,
        "temperature": options.temperature,
        "top_p": options.top_p,
        "stream": True,
        "ignore_eos": options.ignore_eos,
    }
    if prompt.system:
        body["system"] = prompt.system

    parts: list[str] = []
    async with client.stream("POST", "/v1/messages", json=body) as resp:
        record.t_headers = time.perf_counter()
        record.http_status = resp.status_code
        if resp.status_code != 200:
            await resp.aread()
            raise httpx.HTTPStatusError("bench", request=resp.request, response=resp)

        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[len("data:"):].strip())
            kind = event.get("type")
            if kind == "content_block_delta":
                if record.t_first_token is None:
                    record.t_first_token = time.perf_counter()
                delta = event.get("delta") or {}
                piece = delta.get("text") or delta.get("thinking")
                if piece:
                    parts.append(piece)
            elif kind == "message_start":
                usage = (event.get("message") or {}).get("usage") or {}
                record.input_tokens = usage.get("input_tokens", 0)
            elif kind == "message_delta":
                record.finish_reason = (event.get("delta") or {}).get("stop_reason")
                usage = event.get("usage") or {}
                record.output_tokens = usage.get("output_tokens", record.output_tokens)
            elif kind == "error":
                record.error = (event.get("error") or {}).get("message", "stream error")
    record.t_done = time.perf_counter()
    return "".join(parts)


def _openai_messages(prompt: Prompt) -> list[dict]:
    messages = list(prompt.messages)
    if prompt.system:
        messages = [{"role": "system", "content": prompt.system}, *messages]
    return messages


def _anthropic_messages(prompt: Prompt) -> list[dict]:
    # Workload records are OpenAI-shape; the Anthropic dialect takes the
    # same user/assistant turns with system lifted to the top level.
    return [m for m in prompt.messages if m["role"] in ("user", "assistant")]
