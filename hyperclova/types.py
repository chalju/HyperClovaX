"""Type definitions for HyperCLOVA API."""

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
)
from typing_extensions import NotRequired


# Model names
ModelName = Literal["HCX-005", "HCX-007", "HCX-DASH-002"]

# Role types
Role = Literal["system", "user", "assistant", "tool"]

# Thinking effort levels
ThinkingEffort = Literal["none", "low", "medium", "high"]

# Tool types
ToolType = Literal["function"]

# Tool choice types
ToolChoiceType = Union[Literal["auto", "none"], Dict[str, Any]]

# Finish reasons
FinishReason = Literal["length", "stop", "tool_calls"]

# Content types
ContentType = Literal["text", "image_url"]

# Response format type
ResponseFormatType = Literal["json"]

# AI Filter types
AIFilterGroupName = Literal["curse", "unsafeContents"]
AIFilterName = Literal["discrimination", "insult", "sexualHarassment"]
AIFilterScore = Literal["-1", "0", "1", "2"]
AIFilterResult = Literal["OK", "ERROR"]


class TextContent(TypedDict):
    """Text content in a message."""
    type: Literal["text"]
    text: str


class ImageUrlContent(TypedDict):
    """Image URL content in a message."""
    type: Literal["image_url"]
    imageUrl: NotRequired[Dict[str, str]]
    dataUri: NotRequired[Dict[str, str]]


# Message content can be string or array of content items
MessageContent = Union[str, List[Union[TextContent, ImageUrlContent]]]


class Message(TypedDict):
    """Chat message."""
    role: Role
    content: MessageContent
    toolCalls: NotRequired[List[Dict[str, Any]]]
    toolCallId: NotRequired[str]


class ThinkingConfig(TypedDict):
    """Thinking configuration for HCX-007."""
    effort: ThinkingEffort


class FunctionParameters(TypedDict):
    """Function parameters schema."""
    type: str
    properties: Dict[str, Any]
    required: NotRequired[List[str]]


class FunctionDefinition(TypedDict):
    """Function definition for function calling."""
    name: str
    description: str
    parameters: FunctionParameters


class Tool(TypedDict):
    """Tool definition."""
    type: ToolType
    function: FunctionDefinition


class ToolChoice(TypedDict):
    """Tool choice configuration."""
    type: str
    function: NotRequired[Dict[str, str]]


class ResponseFormat(TypedDict):
    """Response format configuration."""
    type: ResponseFormatType
    schema: Dict[str, Any]


class ChatCompletionRequest(TypedDict):
    """Chat completion request."""
    messages: List[Message]
    topP: NotRequired[float]
    topK: NotRequired[int]
    maxTokens: NotRequired[int]
    maxCompletionTokens: NotRequired[int]
    temperature: NotRequired[float]
    repetitionPenalty: NotRequired[float]
    stop: NotRequired[List[str]]
    seed: NotRequired[int]
    includeAiFilters: NotRequired[bool]
    thinking: NotRequired[ThinkingConfig]
    tools: NotRequired[List[Tool]]
    toolChoice: NotRequired[ToolChoiceType]
    responseFormat: NotRequired[ResponseFormat]


class Usage(TypedDict):
    """Token usage information."""
    promptTokens: int
    completionTokens: int
    totalTokens: int
    completionTokensDetails: NotRequired[Dict[str, int]]


class AIFilter(TypedDict):
    """AI filter result."""
    groupName: AIFilterGroupName
    name: AIFilterName
    score: AIFilterScore
    result: NotRequired[AIFilterResult]


class ToolCall(TypedDict):
    """Tool call information."""
    id: str
    type: ToolType
    function: Dict[str, Any]


class ResponseMessage(TypedDict):
    """Response message."""
    role: Role
    content: str
    thinkingContent: NotRequired[str]
    toolCalls: NotRequired[List[ToolCall]]


class ChatCompletionResult(TypedDict):
    """Chat completion result."""
    message: ResponseMessage
    finishReason: FinishReason
    created: int
    seed: int
    usage: Usage
    aiFilter: NotRequired[List[AIFilter]]


class EmbeddingRequest(TypedDict):
    """Embedding request."""
    text: str


class EmbeddingResult(TypedDict):
    """Embedding result."""
    embedding: List[float]
    inputTokens: int


class StreamingMessage(TypedDict):
    """Streaming message."""
    role: Role
    content: NotRequired[str]
    thinkingContent: NotRequired[str]
    toolCalls: NotRequired[List[Dict[str, Any]]]


class StreamingTokenEvent(TypedDict):
    """Streaming token event data."""
    message: StreamingMessage
    finishReason: Optional[FinishReason]
    created: int
    seed: int
    usage: Optional[Usage]


class StreamingResultEvent(TypedDict):
    """Streaming result event data."""
    message: ResponseMessage
    finishReason: FinishReason
    created: int
    seed: int
    usage: Usage
    aiFilter: NotRequired[List[AIFilter]]


class ErrorStatus(TypedDict):
    """Error status."""
    code: str
    message: str


class ErrorEvent(TypedDict):
    """Error event data."""
    status: ErrorStatus