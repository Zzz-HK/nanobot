from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from langchain.agents.middleware import AgentState as LangChainAgentState
from typing_extensions import NotRequired


class PendingEventSource(Protocol):
    """pending queue 的最小读取协议，避免 middleware 依赖具体 SessionManager。"""

    async def drain_pending(self, session_id: str, *, limit: int = 3) -> list[Any]:
        """取出等待注入当前会话的事件。"""


class MemoryReader(Protocol):
    """动态提示词和摘要归档所需的最小记忆协议。"""

    def read_all_memory_files(self) -> Any:
        """读取三份 Markdown；允许同步兼容层或异步后端。"""

    def append_history(self, summary: str, *, kind: str = "summary") -> Any:
        """归档一条压缩摘要。"""


class AgentState(LangChainAgentState[Any]):
    """`create_agent` 与外层 Goal 图共享的 checkpoint 状态。"""

    goal_state: NotRequired[dict[str, Any] | None]
    continuation_count: NotRequired[int]
    empty_response_count: NotRequired[int]
    last_archived_summary_hash: NotRequired[str]
    goal_creation_allowed: NotRequired[bool]


@dataclass(slots=True)
class AgentContext:
    """一次图调用的非持久化依赖与授权信息。"""

    session_id: str
    memory: MemoryReader
    pending: PendingEventSource | None = None
    goal_creation_allowed: bool = False
    force_compact: bool = False
