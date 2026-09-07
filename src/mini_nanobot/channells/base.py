from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..bus import InboundMessage, MessageBus


class BaseChannel(ABC):
    name: str = "base"
    supports_streaming: bool = False

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """开始监听平台消息（长期运行的协程）。"""

    @abstractmethod
    async def stop(self) -> None:
        """停止监听、清理资源。"""

    @abstractmethod
    async def send(self, session_id: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """发送一条完整的回复。"""

    async def send_delta(
        self, session_id: str, delta: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """发送一段流式增量文本。默认不支持流式，子类可以覆盖。"""
        return

    async def send_delta_end(self, session_id: str, metadata: dict[str, Any] | None = None) -> None:
        """标记一段流式输出结束。默认不支持流式，子类可以覆盖。"""
        return

    async def _handle_message(
        self, session_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """把平台收到的消息，包装成 InboundMessage 推入总线。"""
        meta = dict(metadata or {})
        meta.setdefault("supports_stream", self.supports_streaming)
        await self.bus.publish_inbound(
            InboundMessage(
                channel=self.name,
                session_id=session_id,
                content=content,
                metadata=meta,
            )
        )

    @property
    def is_running(self) -> bool:
        return self._running