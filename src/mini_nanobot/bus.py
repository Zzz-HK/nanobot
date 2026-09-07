from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InboundMessage:
    """一条从 Channel 流向 AgentService 的消息。"""

    channel: str
    session_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutboundMessage:
    """一条从 AgentService 流向 Channel 的消息。"""

    channel: str
    session_id: str
    content: str
    event: str = "final"  # "final" | "delta" | "stream_end" | "error"
    metadata: dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """两个 asyncio 队列，仅此而已。"""

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        return await self.outbound.get()
