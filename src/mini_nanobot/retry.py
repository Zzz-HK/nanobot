from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ErrorClassification:
    should_retry: bool
    wait_seconds: float
    kind: str


# 不可重试：认证错误、欠费/额度问题、请求参数错误——重试也不会变好。
_NO_RETRY_STATUS = {400, 401, 402, 403, 404, 422}
# 可重试：限流、服务端临时故障。
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def classify_error(exc: Exception, attempt: int) -> ErrorClassification:
    """检查异常上的 status_code（大部分 HTTP 客户端库都会挂这个属性），决定是否重试。"""
    status_code = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )

    if status_code in _NO_RETRY_STATUS:
        kind = "auth_or_quota" if status_code in (401, 402, 403) else "bad_request"
        return ErrorClassification(should_retry=False, wait_seconds=0.0, kind=kind)

    if status_code in _RETRYABLE_STATUS:
        # 429 限流：等久一点；5xx：指数退避
        base = 5.0 if status_code == 429 else 1.5
        return ErrorClassification(
            should_retry=True, wait_seconds=base * (2**attempt), kind="rate_limit_or_server_error"
        )

    # 未知异常（网络超时、连接被拒等）：谨慎重试几次，指数退避。
    return ErrorClassification(should_retry=True, wait_seconds=1.5 * (2**attempt), kind="unknown")
