"""HyperCLOVA Python Library.

A Python library for interacting with HyperCLOVA X APIs, providing chat completions
and embeddings functionality similar to OpenAI and LangChain libraries.

Example:
    ```python
    from hyperclova import HyperClova
    
    client = HyperClova(api_key="your-api-key")
    
    # Chat completion
    response = client.chat.completions.create(
        model="HCX-007",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello!"}
        ]
    )
    print(response.message.content)
    
    # Embeddings
    embedding = client.embeddings.create(text="Hello world")
    print(f"Vector dimension: {embedding.dimension}")
    ```
"""

__version__ = "0.1.0"

from .client import HyperClova
from .exceptions import (
    HyperClovaError,
    AuthenticationError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InvalidRequestError,
    ServerError,
    StreamingError,
    ModelNotSupportedError,
    TokenLimitExceededError,
)
from .types import (
    ModelName,
    Role,
    ThinkingEffort,
    FinishReason,
    ContentType,
)
from .models import (
    Message,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    EmbeddingRequest,
    EmbeddingResponse,
    Tool,
    FunctionDefinition,
    ResponseFormat,
    ThinkingConfig,
    Usage,
    AIFilter,
)

__all__ = [
    # Main client
    "HyperClova",
    
    # Exceptions
    "HyperClovaError",
    "AuthenticationError",
    "APIConnectionError", 
    "APITimeoutError",
    "RateLimitError",
    "InvalidRequestError",
    "ServerError",
    "StreamingError",
    "ModelNotSupportedError",
    "TokenLimitExceededError",
    
    # Types
    "ModelName",
    "Role",
    "ThinkingEffort",
    "FinishReason",
    "ContentType",
    
    # Models
    "Message",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "Tool",
    "FunctionDefinition",
    "ResponseFormat",
    "ThinkingConfig",
    "Usage",
    "AIFilter",
]