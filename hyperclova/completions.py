"""Chat Completions API implementation for HyperCLOVA."""

import logging
from typing import Optional, List, Union, Dict, Any, Iterator, AsyncIterator, overload, Literal

import httpx

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Message,
    Tool,
    ResponseFormat,
    ThinkingConfig,
    APIResponse,
)
from .types import ModelName, ToolChoiceType
from .exceptions import (
    raise_for_status_code,
    APIConnectionError,
    APITimeoutError,
    ModelNotSupportedError,
)
from .utils import (
    build_url,
    get_headers,
    retry_with_backoff,
    prepare_message_content,
    validate_model_capability,
    get_max_tokens,
)
from .streaming import create_streaming_response

logger = logging.getLogger(__name__)


class ChatCompletions:
    """Chat Completions API client."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None
        self._async_client = None
    
    @property
    def client(self) -> httpx.Client:
        """Get or create sync HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client
    
    @property
    def async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client
    
    @overload
    def create(
        self,
        *,
        model: ModelName,
        messages: List[Union[Message, Dict[str, Any]]],
        stream: Literal[False] = False,
        **kwargs
    ) -> ChatCompletionResponse: ...
    
    @overload
    def create(
        self,
        *,
        model: ModelName,
        messages: List[Union[Message, Dict[str, Any]]],
        stream: Literal[True],
        **kwargs
    ) -> Iterator[ChatCompletionChunk]: ...
    
    def create(
        self,
        *,
        model: ModelName,
        messages: List[Union[Message, Dict[str, Any]]],
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        include_ai_filters: Optional[bool] = None,
        thinking: Optional[Union[ThinkingConfig, Dict[str, str]]] = None,
        tools: Optional[List[Union[Tool, Dict[str, Any]]]] = None,
        tool_choice: Optional[ToolChoiceType] = None,
        response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletionResponse, Iterator[ChatCompletionChunk]]:
        """Create a chat completion.
        
        Args:
            model: Model to use (HCX-005, HCX-007, HCX-DASH-002)
            messages: List of messages
            stream: Whether to stream the response
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate (mutually exclusive with max_completion_tokens)
            max_completion_tokens: Maximum completion tokens (for thinking model)
            repetition_penalty: Repetition penalty
            stop: Stop sequences
            seed: Random seed for reproducibility
            include_ai_filters: Whether to include AI filter results
            thinking: Thinking configuration (HCX-007 only)
            tools: List of available tools for function calling
            tool_choice: How to choose tools
            response_format: Response format configuration (HCX-007 only)
            request_id: Optional request ID
            
        Returns:
            ChatCompletionResponse or iterator of ChatCompletionChunk if streaming
        """
        # Validate model capabilities
        if thinking:
            validate_model_capability(model, "thinking")
        if response_format:
            validate_model_capability(model, "structured_output")
        if tools:
            validate_model_capability(model, "function_calling")
        
        # Check for image content in messages
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = msg.content if hasattr(msg, "content") else ""
            
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        validate_model_capability(model, "vision")
                        break
        
        # Prepare messages
        prepared_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                prepared_msg = {
                    "role": msg["role"],
                    "content": prepare_message_content(msg["content"]),
                }
                if "tool_calls" in msg:
                    prepared_msg["toolCalls"] = msg["tool_calls"]
                if "tool_call_id" in msg:
                    prepared_msg["toolCallId"] = msg["tool_call_id"]
                prepared_messages.append(prepared_msg)
            else:
                prepared_messages.append(msg.model_dump(by_alias=True, exclude_none=True))
        
        # Build request
        request_data = {
            "messages": prepared_messages,
        }
        
        # Add optional parameters
        if temperature is not None:
            request_data["temperature"] = temperature
        if top_p is not None:
            request_data["topP"] = top_p
        if top_k is not None:
            request_data["topK"] = top_k
        
        # Handle token parameters based on model
        if model == "HCX-007":
            # HCX-007 always requires maxCompletionTokens
            if max_completion_tokens is not None:
                request_data["maxCompletionTokens"] = max_completion_tokens
            elif max_tokens is not None:
                request_data["maxCompletionTokens"] = max_tokens
            elif thinking:
                # Set default based on thinking effort
                effort = thinking.get("effort", "low") if isinstance(thinking, dict) else thinking.effort
                request_data["maxCompletionTokens"] = get_max_tokens(model, effort)
            else:
                # Default for HCX-007
                request_data["maxCompletionTokens"] = 2048
        else:
            # Other models use maxTokens
            if max_tokens is not None:
                request_data["maxTokens"] = max_tokens
            elif max_completion_tokens is not None:
                request_data["maxTokens"] = max_completion_tokens
        if repetition_penalty is not None:
            request_data["repetitionPenalty"] = repetition_penalty
        if stop is not None:
            request_data["stop"] = stop
        if seed is not None:
            request_data["seed"] = seed
        if include_ai_filters is not None:
            request_data["includeAiFilters"] = include_ai_filters
        # Handle thinking parameter
        if thinking is not None:
            if isinstance(thinking, dict):
                request_data["thinking"] = thinking
            else:
                request_data["thinking"] = thinking.model_dump()
        elif response_format is not None and model == "HCX-007":
            # Structured output requires thinking.effort = "none"
            request_data["thinking"] = {"effort": "none"}
        if tools is not None:
            request_data["tools"] = [
                tool if isinstance(tool, dict) else tool.model_dump(by_alias=True)
                for tool in tools
            ]
        if tool_choice is not None:
            if isinstance(tool_choice, str):
                request_data["toolChoice"] = tool_choice
            else:
                request_data["toolChoice"] = tool_choice
        if response_format is not None:
            if isinstance(response_format, dict):
                request_data["responseFormat"] = response_format
            else:
                request_data["responseFormat"] = response_format.model_dump(by_alias=True)
        
        # Create request object for validation
        request = ChatCompletionRequest(**request_data)
        
        if stream:
            return self._create_stream(model, request, request_id)
        else:
            return self._create_sync(model, request, request_id)
    
    def _create_sync(
        self,
        model: str,
        request: ChatCompletionRequest,
        request_id: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Create a synchronous chat completion."""
        def _make_request():
            url = build_url(self.base_url, f"/v3/chat-completions/{model}")
            headers = get_headers(self.api_key, request_id)
            
            try:
                response = self.client.post(
                    url,
                    headers=headers,
                    json=request.model_dump(by_alias=True, exclude_none=True),
                )
                response.raise_for_status()
                
                data = response.json()
                api_response = APIResponse(**data)
                
                if api_response.status.code != "20000":
                    raise_for_status_code(
                        int(api_response.status.code),
                        data
                    )
                
                return ChatCompletionResponse(**api_response.result)
                
            except httpx.ConnectError as e:
                raise APIConnectionError(f"Failed to connect to API: {e}")
            except httpx.TimeoutException as e:
                raise APITimeoutError(f"Request timed out: {e}")
            except httpx.HTTPStatusError as e:
                raise_for_status_code(e.response.status_code, e.response.json())
        
        if self.max_retries > 0:
            return retry_with_backoff(
                _make_request,
                max_retries=self.max_retries
            )
        else:
            return _make_request()
    
    def _create_stream(
        self,
        model: str,
        request: ChatCompletionRequest,
        request_id: Optional[str] = None,
    ) -> Iterator[ChatCompletionChunk]:
        """Create a streaming chat completion."""
        url = build_url(self.base_url, f"/v3/chat-completions/{model}")
        headers = get_headers(self.api_key, request_id, accept="text/event-stream")
        
        try:
            response = self.client.post(
                url,
                headers=headers,
                json=request.model_dump(by_alias=True, exclude_none=True),
                timeout=None,  # No timeout for streaming
            )
            response.raise_for_status()
            
            streaming_response = create_streaming_response(response, is_async=False)
            yield from streaming_response
            
        except httpx.ConnectError as e:
            raise APIConnectionError(f"Failed to connect to API: {e}")
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"Request timed out: {e}")
        except httpx.HTTPStatusError as e:
            raise_for_status_code(e.response.status_code, e.response.json())
    
    @overload
    async def acreate(
        self,
        *,
        model: ModelName,
        messages: List[Union[Message, Dict[str, Any]]],
        stream: Literal[False] = False,
        **kwargs
    ) -> ChatCompletionResponse: ...
    
    @overload
    async def acreate(
        self,
        *,
        model: ModelName,
        messages: List[Union[Message, Dict[str, Any]]],
        stream: Literal[True],
        **kwargs
    ) -> AsyncIterator[ChatCompletionChunk]: ...
    
    async def acreate(
        self,
        *,
        model: ModelName,
        messages: List[Union[Message, Dict[str, Any]]],
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        include_ai_filters: Optional[bool] = None,
        thinking: Optional[Union[ThinkingConfig, Dict[str, str]]] = None,
        tools: Optional[List[Union[Tool, Dict[str, Any]]]] = None,
        tool_choice: Optional[ToolChoiceType] = None,
        response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        """Create a chat completion asynchronously."""
        # Validate model capabilities
        if thinking:
            validate_model_capability(model, "thinking")
        if response_format:
            validate_model_capability(model, "structured_output")
        if tools:
            validate_model_capability(model, "function_calling")
        
        # Check for image content in messages
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = msg.content if hasattr(msg, "content") else ""
            
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        validate_model_capability(model, "vision")
                        break
        
        # Prepare messages
        prepared_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                prepared_msg = {
                    "role": msg["role"],
                    "content": prepare_message_content(msg["content"]),
                }
                if "tool_calls" in msg:
                    prepared_msg["toolCalls"] = msg["tool_calls"]
                if "tool_call_id" in msg:
                    prepared_msg["toolCallId"] = msg["tool_call_id"]
                prepared_messages.append(prepared_msg)
            else:
                prepared_messages.append(msg.model_dump(by_alias=True, exclude_none=True))
        
        # Build request
        request_data = {
            "messages": prepared_messages,
        }
        
        # Add optional parameters
        if temperature is not None:
            request_data["temperature"] = temperature
        if top_p is not None:
            request_data["topP"] = top_p
        if top_k is not None:
            request_data["topK"] = top_k
        
        # Handle token parameters based on model
        if model == "HCX-007":
            # HCX-007 always requires maxCompletionTokens
            if max_completion_tokens is not None:
                request_data["maxCompletionTokens"] = max_completion_tokens
            elif max_tokens is not None:
                request_data["maxCompletionTokens"] = max_tokens
            elif thinking:
                # Set default based on thinking effort
                effort = thinking.get("effort", "low") if isinstance(thinking, dict) else thinking.effort
                request_data["maxCompletionTokens"] = get_max_tokens(model, effort)
            else:
                # Default for HCX-007
                request_data["maxCompletionTokens"] = 2048
        else:
            # Other models use maxTokens
            if max_tokens is not None:
                request_data["maxTokens"] = max_tokens
            elif max_completion_tokens is not None:
                request_data["maxTokens"] = max_completion_tokens
        if repetition_penalty is not None:
            request_data["repetitionPenalty"] = repetition_penalty
        if stop is not None:
            request_data["stop"] = stop
        if seed is not None:
            request_data["seed"] = seed
        if include_ai_filters is not None:
            request_data["includeAiFilters"] = include_ai_filters
        # Handle thinking parameter
        if thinking is not None:
            if isinstance(thinking, dict):
                request_data["thinking"] = thinking
            else:
                request_data["thinking"] = thinking.model_dump()
        elif response_format is not None and model == "HCX-007":
            # Structured output requires thinking.effort = "none"
            request_data["thinking"] = {"effort": "none"}
        if tools is not None:
            request_data["tools"] = [
                tool if isinstance(tool, dict) else tool.model_dump(by_alias=True)
                for tool in tools
            ]
        if tool_choice is not None:
            if isinstance(tool_choice, str):
                request_data["toolChoice"] = tool_choice
            else:
                request_data["toolChoice"] = tool_choice
        if response_format is not None:
            if isinstance(response_format, dict):
                request_data["responseFormat"] = response_format
            else:
                request_data["responseFormat"] = response_format.model_dump(by_alias=True)
        
        # Create request object for validation
        request = ChatCompletionRequest(**request_data)
        
        if stream:
            return self._acreate_stream(model, request, request_id)
        else:
            return await self._acreate_sync(model, request, request_id)
    
    async def _acreate_sync(
        self,
        model: str,
        request: ChatCompletionRequest,
        request_id: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Create an asynchronous chat completion."""
        url = build_url(self.base_url, f"/v3/chat-completions/{model}")
        headers = get_headers(self.api_key, request_id)
        
        try:
            response = await self.async_client.post(
                url,
                headers=headers,
                json=request.model_dump(by_alias=True, exclude_none=True),
            )
            response.raise_for_status()
            
            data = response.json()
            api_response = APIResponse(**data)
            
            if api_response.status.code != "20000":
                raise_for_status_code(
                    int(api_response.status.code),
                    data
                )
            
            return ChatCompletionResponse(**api_response.result)
            
        except httpx.ConnectError as e:
            raise APIConnectionError(f"Failed to connect to API: {e}")
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"Request timed out: {e}")
        except httpx.HTTPStatusError as e:
            raise_for_status_code(e.response.status_code, e.response.json())
    
    async def _acreate_stream(
        self,
        model: str,
        request: ChatCompletionRequest,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Create an asynchronous streaming chat completion."""
        url = build_url(self.base_url, f"/v3/chat-completions/{model}")
        headers = get_headers(self.api_key, request_id, accept="text/event-stream")
        
        try:
            response = await self.async_client.post(
                url,
                headers=headers,
                json=request.model_dump(by_alias=True, exclude_none=True),
                timeout=None,  # No timeout for streaming
            )
            response.raise_for_status()
            
            streaming_response = create_streaming_response(response, is_async=True)
            async for chunk in streaming_response:
                yield chunk
            
        except httpx.ConnectError as e:
            raise APIConnectionError(f"Failed to connect to API: {e}")
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"Request timed out: {e}")
        except httpx.HTTPStatusError as e:
            raise_for_status_code(e.response.status_code, e.response.json())
    
    def close(self):
        """Close HTTP clients."""
        if self._client:
            self._client.close()
            self._client = None
        # Note: async client should be closed with aclose() in async context
    
    async def aclose(self):
        """Close HTTP clients asynchronously."""
        if self._client:
            self._client.close()
            self._client = None
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._client:
            try:
                self._client.close()
            except:
                pass