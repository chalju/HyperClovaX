"""Pydantic models for HyperCLOVA API requests and responses."""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime

from .types import (
    ModelName,
    Role,
    ThinkingEffort,
    FinishReason,
    ContentType,
    AIFilterGroupName,
    AIFilterName,
    AIFilterScore,
    AIFilterResult,
    ToolType,
)


class TextContentItem(BaseModel):
    """Text content item."""
    type: Literal["text"]
    text: str


class ImageUrlItem(BaseModel):
    """Image URL item."""
    url: str


class DataUriItem(BaseModel):
    """Data URI item for base64 encoded images."""
    data: str


class ImageContentItem(BaseModel):
    """Image content item."""
    type: Literal["image_url"]
    imageUrl: Optional[ImageUrlItem] = Field(None, alias="image_url")
    dataUri: Optional[DataUriItem] = Field(None, alias="data_uri")
    
    model_config = ConfigDict(populate_by_name=True)
    
    @field_validator('imageUrl', 'dataUri')
    def validate_image_source(cls, v, info):
        """Ensure at least one image source is provided."""
        if info.field_name == 'dataUri' and v is None and info.data.get('imageUrl') is None:
            raise ValueError("Either imageUrl or dataUri must be provided")
        return v


ContentItem = Union[TextContentItem, ImageContentItem]


class Message(BaseModel):
    """Chat message."""
    role: Role
    content: Union[str, List[ContentItem]]
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, alias="toolCalls")
    tool_call_id: Optional[str] = Field(None, alias="toolCallId")
    
    model_config = ConfigDict(populate_by_name=True)


class ThinkingConfig(BaseModel):
    """Thinking configuration for HCX-007."""
    effort: ThinkingEffort = "low"


class FunctionParameters(BaseModel):
    """Function parameters schema."""
    type: str = "object"
    properties: Dict[str, Any]
    required: Optional[List[str]] = None


class FunctionDefinition(BaseModel):
    """Function definition."""
    name: str
    description: str
    parameters: FunctionParameters


class Tool(BaseModel):
    """Tool definition."""
    type: ToolType = "function"
    function: FunctionDefinition


class ToolChoiceFunction(BaseModel):
    """Tool choice function specification."""
    name: str


class ToolChoice(BaseModel):
    """Tool choice configuration."""
    type: str = "function"
    function: ToolChoiceFunction


class ResponseFormatSchema(BaseModel):
    """Response format schema."""
    type: str
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None
    
    model_config = ConfigDict(extra='allow')


class ResponseFormat(BaseModel):
    """Response format configuration."""
    type: Literal["json"] = "json"
    schema_: ResponseFormatSchema = Field(alias="schema")
    
    model_config = ConfigDict(populate_by_name=True)


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""
    messages: List[Message]
    model: Optional[ModelName] = None
    top_p: Optional[float] = Field(None, alias="topP", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, alias="topK", ge=0, le=128)
    max_tokens: Optional[int] = Field(None, alias="maxTokens", ge=1)
    max_completion_tokens: Optional[int] = Field(None, alias="maxCompletionTokens", ge=1)
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(None, alias="repetitionPenalty", gt=0.0, le=2.0)
    stop: Optional[List[str]] = None
    seed: Optional[int] = Field(None, ge=0, le=4294967295)
    include_ai_filters: Optional[bool] = Field(None, alias="includeAiFilters")
    thinking: Optional[ThinkingConfig] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[Literal["auto", "none"], ToolChoice]] = Field(None, alias="toolChoice")
    response_format: Optional[ResponseFormat] = Field(None, alias="responseFormat")
    
    model_config = ConfigDict(populate_by_name=True)
    
    @field_validator('max_tokens', 'max_completion_tokens')
    def validate_token_params(cls, v, info):
        """Ensure only one of maxTokens or maxCompletionTokens is set."""
        if info.field_name == 'max_completion_tokens' and v is not None:
            if info.data.get('max_tokens') is not None:
                raise ValueError("Cannot use both maxTokens and maxCompletionTokens")
        return v


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(alias="promptTokens")
    completion_tokens: int = Field(alias="completionTokens")
    total_tokens: int = Field(alias="totalTokens")
    completion_tokens_details: Optional[Dict[str, int]] = Field(None, alias="completionTokensDetails")
    
    model_config = ConfigDict(populate_by_name=True)


class AIFilter(BaseModel):
    """AI filter result."""
    group_name: AIFilterGroupName = Field(alias="groupName")
    name: AIFilterName
    score: AIFilterScore
    result: Optional[AIFilterResult] = None
    
    model_config = ConfigDict(populate_by_name=True)


class ToolCallFunction(BaseModel):
    """Tool call function details."""
    name: str
    arguments: Dict[str, Any]


class ToolCall(BaseModel):
    """Tool call information."""
    id: str
    type: ToolType = "function"
    function: ToolCallFunction


class ChatCompletionMessage(BaseModel):
    """Chat completion response message."""
    role: Role
    content: str
    thinking_content: Optional[str] = Field(None, alias="thinkingContent")
    tool_calls: Optional[List[ToolCall]] = Field(None, alias="toolCalls")
    
    model_config = ConfigDict(populate_by_name=True)


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    message: ChatCompletionMessage
    finish_reason: Optional[FinishReason] = Field(None, alias="finishReason")  # Made optional
    created: Optional[int] = None  # Made optional
    seed: Optional[int] = None  # Made optional
    usage: Usage
    ai_filter: Optional[List[AIFilter]] = Field(None, alias="aiFilter")
    model: Optional[str] = None  # Added model field
    
    model_config = ConfigDict(populate_by_name=True)
    
    @property
    def created_datetime(self) -> datetime:
        """Convert millisecond timestamp to datetime."""
        return datetime.fromtimestamp(self.created / 1000)


class ChatCompletionChunk(BaseModel):
    """Chat completion streaming chunk."""
    message: ChatCompletionMessage
    finish_reason: Optional[FinishReason] = Field(None, alias="finishReason")
    created: int
    seed: int
    usage: Optional[Usage] = None
    
    model_config = ConfigDict(populate_by_name=True)


class EmbeddingRequest(BaseModel):
    """Embedding request."""
    text: str = Field(..., max_length=8192)


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    embedding: List[float]
    input_tokens: int = Field(alias="inputTokens")
    
    model_config = ConfigDict(populate_by_name=True)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return len(self.embedding)


class Status(BaseModel):
    """API response status."""
    code: str
    message: str


class APIResponse(BaseModel):
    """Base API response."""
    status: Status
    result: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(extra='allow')