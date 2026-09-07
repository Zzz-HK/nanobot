from __future__ import annotations

import time
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

_STATUS_MAP = {"complete": "completed", "cancel": "cancelled", "block": "blocked"}


@tool
def create_goal(
    objective: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """创建一个 sustained goal（长期目标）。

    只在用户明确要求"持续处理/一直做到完成"这类长任务时调用；普通一次性问答
    不需要创建 goal。创建后，只要目标状态是 active，agent 在给出"看起来完成了"
    的回复后还会被自动提醒继续推进，直到调用 update_goal 结束目标。
    """
    if not state.get("goal_creation_allowed", False):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="拒绝创建长期目标：本轮没有获得用户的显式授权。",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    goal_state: dict[str, Any] = {
        "status": "active",
        "objective": objective,
        "created_at": time.time(),
    }
    return Command(
        update={
            "goal_state": goal_state,
            "messages": [
                ToolMessage(content=f"目标已创建：{objective}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool
def update_goal(
    action: str,
    detail: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """更新当前 sustained goal 的状态。

    Args:
        action: "complete"（完成）/ "cancel"（取消）/ "block"（受阻，需要用户输入）
            / "replace"（换成一个新目标，detail 是新目标的描述）
        detail: 补充说明；action="replace" 时是新目标本身
    """
    goal_state: dict[str, Any] = dict(state.get("goal_state") or {})

    if action == "replace":
        goal_state = {"status": "active", "objective": detail, "created_at": time.time()}
        message = f"目标已替换为：{detail}"
    else:
        new_status = _STATUS_MAP.get(action, "cancelled")
        goal_state["status"] = new_status
        goal_state["detail"] = detail
        message = f"目标状态更新为：{new_status}（{detail}）"

    return Command(
        update={
            "goal_state": goal_state,
            "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)],
        }
    )


GOAL_TOOLS = [create_goal, update_goal]