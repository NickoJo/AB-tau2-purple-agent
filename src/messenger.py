import json
from uuid import uuid4

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
    Consumer,
)
from a2a.types import (
    Message,
    Part,
    Role,
    TextPart,
    DataPart,
)


REQUEST_TIMEOUT = 300


def build_message(
    *, role: Role = Role.user, text: str, context_id: str | None = None
) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def collect_text(parts: list[Part]) -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return "\n".join(chunks)


async def deliver_message(
    text: str,
    base_url: str,
    context_id: str | None = None,
    streaming: bool = False,
    timeout: int = REQUEST_TIMEOUT,
    consumer: Consumer | None = None,
):
    """Send a message to a remote agent and return its response as a dict."""
    async with httpx.AsyncClient(timeout=timeout) as http:
        resolver = A2ACardResolver(httpx_client=http, base_url=base_url)
        card = await resolver.get_agent_card()
        cfg = ClientConfig(httpx_client=http, streaming=streaming)
        client = ClientFactory(cfg).create(card)
        if consumer:
            await client.add_event_consumer(consumer)

        outgoing = build_message(text=text, context_id=context_id)
        last = None
        result = {"response": "", "context_id": None}

        async for event in client.send_message(outgoing):
            last = event

        match last:
            case Message() as msg:
                result["context_id"] = msg.context_id
                result["response"] += collect_text(msg.parts)

            case (task, _update):
                result["context_id"] = task.context_id
                result["status"] = task.status.state.value
                if task.status.message:
                    result["response"] += collect_text(task.status.message.parts)
                if task.artifacts:
                    for artifact in task.artifacts:
                        result["response"] += collect_text(artifact.parts)

            case _:
                pass

        return result


class Messenger:
    def __init__(self):
        self._threads: dict[str, str] = {}

    async def send(
        self,
        text: str,
        url: str,
        new_thread: bool = False,
        timeout: int = REQUEST_TIMEOUT,
    ) -> str:
        """
        Send a message to a remote agent and return its reply.

        Args:
            text: Message content to send.
            url: Remote agent endpoint.
            new_thread: Start a fresh conversation when True; continue existing thread otherwise.
            timeout: Request timeout in seconds.

        Returns:
            The agent's reply as a string.
        """
        ctx = None if new_thread else self._threads.get(url)
        result = await deliver_message(text=text, base_url=url, context_id=ctx, timeout=timeout)

        if result.get("status", "completed") != "completed":
            raise RuntimeError(f"Unexpected response from {url}: {result}")

        self._threads[url] = result.get("context_id")
        return result["response"]

    def clear(self):
        self._threads = {}
