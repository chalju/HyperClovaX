"""Utility functions for HyperCLOVA API."""

import os
import time
import json
import base64
import logging
from typing import Any, Dict, Optional, Union
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


def get_api_key() -> Optional[str]:
    """Get API key from environment variable."""
    return os.environ.get("HYPERCLOVA_API_KEY")


def get_base_url() -> str:
    """Get base URL from environment or use default."""
    return os.environ.get(
        "HYPERCLOVA_BASE_URL", 
        "https://clovastudio.stream.ntruss.com"
    )


def build_url(base_url: str, path: str) -> str:
    """Build full URL from base URL and path."""
    return urljoin(base_url, path)


def get_headers(
    api_key: str,
    request_id: Optional[str] = None,
    content_type: str = "application/json",
    accept: Optional[str] = None,
) -> Dict[str, str]:
    """Build request headers."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": content_type,
    }
    
    if request_id:
        headers["X-NCP-CLOVASTUDIO-REQUEST-ID"] = request_id
    
    if accept:
        headers["Accept"] = accept
    
    return headers


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def is_image_url(url: str) -> bool:
    """Check if URL points to an image."""
    image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    return url.lower().endswith(image_extensions)


def prepare_message_content(content: Union[str, list]) -> Union[str, list]:
    """Prepare message content for API request."""
    if isinstance(content, str):
        return content
    
    # Process list of content items
    prepared_content = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text":
                prepared_content.append({
                    "type": "text",
                    "text": item["text"]
                })
            elif item.get("type") == "image_url":
                if "image_url" in item:
                    prepared_content.append({
                        "type": "image_url",
                        "imageUrl": {"url": item["image_url"]["url"]}
                    })
                elif "data_uri" in item:
                    prepared_content.append({
                        "type": "image_url",
                        "dataUri": {"data": item["data_uri"]["data"]}
                    })
    
    return prepared_content


def convert_to_openai_format(response: Dict[str, Any]) -> Dict[str, Any]:
    """Convert HyperCLOVA response to OpenAI-compatible format."""
    result = response.get("result", {})
    message = result.get("message", {})
    
    choices = [{
        "index": 0,
        "message": {
            "role": message.get("role", "assistant"),
            "content": message.get("content", ""),
        },
        "finish_reason": result.get("finishReason"),
    }]
    
    # Add tool calls if present
    if "toolCalls" in message:
        choices[0]["message"]["tool_calls"] = message["toolCalls"]
    
    # Add thinking content if present
    if "thinkingContent" in message:
        choices[0]["message"]["thinking_content"] = message["thinkingContent"]
    
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": result.get("created", int(time.time() * 1000)) // 1000,
        "model": "hyperclova",
        "choices": choices,
        "usage": {
            "prompt_tokens": result.get("usage", {}).get("promptTokens", 0),
            "completion_tokens": result.get("usage", {}).get("completionTokens", 0),
            "total_tokens": result.get("usage", {}).get("totalTokens", 0),
        },
        "system_fingerprint": str(result.get("seed", "")),
    }


def retry_with_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
):
    """Retry function with exponential backoff."""
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(f"All {max_retries} attempts failed.")
    
    raise last_exception


def validate_model_capability(
    model: str,
    feature: str,
) -> None:
    """Validate if model supports requested feature."""
    capabilities = {
        "HCX-005": ["vision", "function_calling"],
        "HCX-007": ["thinking", "structured_output", "function_calling"],
        "HCX-DASH-002": ["function_calling"],
    }
    
    model_caps = capabilities.get(model, [])
    
    if feature == "thinking" and "thinking" not in model_caps:
        raise ValueError(f"Model {model} does not support thinking mode")
    elif feature == "vision" and "vision" not in model_caps:
        raise ValueError(f"Model {model} does not support image input")
    elif feature == "structured_output" and "structured_output" not in model_caps:
        raise ValueError(f"Model {model} does not support structured output")
    elif feature == "function_calling" and "function_calling" not in model_caps:
        raise ValueError(f"Model {model} does not support function calling")


def get_max_tokens(model: str, thinking_effort: Optional[str] = None) -> int:
    """Get default max tokens for model and thinking effort."""
    if model == "HCX-007" and thinking_effort:
        effort_defaults = {
            "none": 512,
            "low": 5120,
            "medium": 10240,
            "high": 20480,
        }
        return effort_defaults.get(thinking_effort, 5120)
    
    model_defaults = {
        "HCX-005": 4096,
        "HCX-007": 4096,
        "HCX-DASH-002": 4096,
    }
    
    return model_defaults.get(model, 4096)
