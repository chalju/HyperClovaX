"""Main HyperCLOVA client class."""

import os
import logging
from typing import Optional

from .completions import ChatCompletions
from .embeddings import Embeddings
from .utils import get_api_key, get_base_url
from .exceptions import AuthenticationError

logger = logging.getLogger(__name__)


class HyperClova:
    """Main client for HyperCLOVA API.
    
    This client provides access to all HyperCLOVA API endpoints.
    
    Example:
        ```python
        from hyperclova import HyperClova
        
        # Initialize with API key
        client = HyperClova(api_key="your-api-key")
        
        # Or use environment variable
        os.environ["HYPERCLOVA_API_KEY"] = "your-api-key"
        client = HyperClova()
        
        # Create a chat completion
        response = client.chat.completions.create(
            model="HCX-007",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello!"}
            ]
        )
        print(response.message.content)
        
        # Create embeddings
        embedding = client.embeddings.create(text="Hello world")
        print(f"Embedding dimension: {len(embedding.embedding)}")
        ```
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize HyperCLOVA client.
        
        Args:
            api_key: API key for authentication. If not provided, will look for
                    HYPERCLOVA_API_KEY environment variable.
            base_url: Base URL for API. If not provided, will use default or
                     HYPERCLOVA_BASE_URL environment variable.
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        # Get API key
        self.api_key = api_key or get_api_key()
        if not self.api_key:
            raise AuthenticationError(
                "No API key provided. Please set HYPERCLOVA_API_KEY environment "
                "variable or pass api_key parameter."
            )
        
        # Get base URL
        self.base_url = base_url or get_base_url()
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize sub-clients
        self._chat = None
        self._embeddings = None
        
        logger.info(f"Initialized HyperCLOVA client with base URL: {self.base_url}")
    
    @property
    def chat(self):
        """Get chat namespace with completions."""
        if self._chat is None:
            self._chat = ChatNamespace(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._chat
    
    @property
    def embeddings(self) -> Embeddings:
        """Get embeddings client."""
        if self._embeddings is None:
            self._embeddings = Embeddings(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._embeddings
    
    def close(self):
        """Close all HTTP clients."""
        if self._chat:
            self._chat.completions.close()
        if self._embeddings:
            self._embeddings.close()
    
    async def aclose(self):
        """Close all HTTP clients asynchronously."""
        if self._chat:
            await self._chat.completions.aclose()
        if self._embeddings:
            await self._embeddings.aclose()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class ChatNamespace:
    """Chat API namespace."""
    
    def __init__(self, api_key: str, base_url: str, timeout: float, max_retries: int):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._completions = None
    
    @property
    def completions(self) -> ChatCompletions:
        """Get chat completions client."""
        if self._completions is None:
            self._completions = ChatCompletions(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._completions