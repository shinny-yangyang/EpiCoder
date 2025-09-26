import os
import time
import logging
from typing import List, Union, Tuple, Dict, Optional, Any

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
DEFAULT_MAX_RETRY = int(os.environ.get("OPENROUTER_MAX_RETRY", "5"))
DEFAULT_TIMEOUT = int(os.environ.get("OPENROUTER_TIMEOUT", "60"))


class OpenRouterError(Exception):
    """Raised when the OpenRouter API returns an error or an unexpected payload."""


def _ensure_api_key():
    if not OPENROUTER_API_KEY:
        raise OpenRouterError(
            "未检测到 OPENROUTER_API_KEY 环境变量，请参考 OpenRouter 文档配置 API Key"
        )


def _build_headers(extra_headers: Dict[str, str] = None) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "https://openrouter.ai"),
        "X-Title": os.environ.get("OPENROUTER_TITLE", "EpiCoder"),
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _normalize_messages(messages: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if not isinstance(messages, list):
        raise OpenRouterError("messages 参数需要是字符串或对话列表")
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise OpenRouterError("messages 列表每个元素必须包含 role 与 content")
    return messages


def call_openrouter(
    messages: Union[str, List[Dict[str, str]]],
    model: str = DEFAULT_MODEL,
    n: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_retry: int = DEFAULT_MAX_RETRY,
    timeout: int = DEFAULT_TIMEOUT,
    extra_headers: Dict[str, str] = None,
    extra_parameters: Dict[str, Union[str, float, int, Dict]] = None,
) -> Tuple[str, Dict]:
    _ensure_api_key()
    payload_messages = _normalize_messages(messages)

    body: Dict[str, Union[str, float, int, List, Dict]] = {
        "model": model,
        "messages": payload_messages,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
    }

    if extra_parameters:
        body.update(extra_parameters)

    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = _build_headers(extra_headers)

    last_exception: Optional[Exception] = None
    for attempt in range(1, max_retry + 1):
        try:
            response = requests.post(url, json=body, headers=headers, timeout=timeout)
            if response.status_code == 200:
                payload = response.json()
                choices = payload.get("choices")
                if not choices:
                    raise OpenRouterError(f"响应缺少 choices 字段: {payload}")
                message = choices[0]["message"]["content"]
                return message, payload

            if response.status_code in {429, 500, 502, 503, 504}:
                logging.warning(
                    "OpenRouter 调用失败 (status=%s, attempt=%s/%s): %s",
                    response.status_code,
                    attempt,
                    max_retry,
                    response.text,
                )
                time.sleep(min(5 * attempt, 30))
                continue

            raise OpenRouterError(
                f"OpenRouter 返回错误 (status={response.status_code}): {response.text}"
            )

        except requests.RequestException as exc:
            logging.warning(
                "OpenRouter 请求异常 (attempt=%s/%s): %s", attempt, max_retry, exc
            )
            time.sleep(min(5 * attempt, 30))
            last_exception = exc
    else:
        raise OpenRouterError(f"多次重试后仍无法访问 OpenRouter: {last_exception}")


def call_gpt4(messages, model: str = DEFAULT_MODEL, **kwargs):
    """与旧接口保持兼容，返回 (message, raw_response)"""
    unused_keys = {"client_idx", "stream", "logprobs"}
    filtered_kwargs: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in unused_keys:
            continue
        filtered_kwargs[key] = value
    return call_openrouter(messages, **filtered_kwargs)
