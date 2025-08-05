"""Embeddings API implementation for HyperCLOVA."""

import logging
from typing import Optional, List, Union

import httpx

from .models import EmbeddingRequest, EmbeddingResponse, APIResponse
from .exceptions import raise_for_status_code, APIConnectionError, APITimeoutError
from .utils import build_url, get_headers, retry_with_backoff

logger = logging.getLogger(__name__)


class Embeddings:
    """Embeddings API client."""
    
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
    
    def create(
        self,
        text: str,
        request_id: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Create embeddings for text.
        
        Args:
            text: Text to embed (max 8,192 tokens)
            request_id: Optional request ID for tracking
            
        Returns:
            EmbeddingResponse with 1024-dimensional embedding vector
        """
        request = EmbeddingRequest(text=text)
        
        def _make_request():
            url = build_url(self.base_url, "/v1/api-tools/embedding/v2")
            headers = get_headers(self.api_key, request_id)
            
            try:
                response = self.client.post(
                    url,
                    headers=headers,
                    json=request.model_dump(),
                )
                response.raise_for_status()
                
                data = response.json()
                api_response = APIResponse(**data)
                
                if api_response.status.code != "20000":
                    raise_for_status_code(
                        int(api_response.status.code),
                        data
                    )
                
                return EmbeddingResponse(**api_response.result)
                
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
    
    async def acreate(
        self,
        text: str,
        request_id: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Create embeddings for text asynchronously.
        
        Args:
            text: Text to embed (max 8,192 tokens)
            request_id: Optional request ID for tracking
            
        Returns:
            EmbeddingResponse with 1024-dimensional embedding vector
        """
        request = EmbeddingRequest(text=text)
        
        url = build_url(self.base_url, "/v1/api-tools/embedding/v2")
        headers = get_headers(self.api_key, request_id)
        
        try:
            response = await self.async_client.post(
                url,
                headers=headers,
                json=request.model_dump(),
            )
            response.raise_for_status()
            
            data = response.json()
            api_response = APIResponse(**data)
            
            if api_response.status.code != "20000":
                raise_for_status_code(
                    int(api_response.status.code),
                    data
                )
            
            return EmbeddingResponse(**api_response.result)
            
        except httpx.ConnectError as e:
            raise APIConnectionError(f"Failed to connect to API: {e}")
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"Request timed out: {e}")
        except httpx.HTTPStatusError as e:
            raise_for_status_code(e.response.status_code, e.response.json())
    
    def create_batch(
        self,
        texts: List[str],
        request_id: Optional[str] = None,
    ) -> List[EmbeddingResponse]:
        """Create embeddings for multiple texts.
        
        Note: This method makes multiple API calls as the API
        doesn't support batch embedding natively.
        
        Args:
            texts: List of texts to embed
            request_id: Optional request ID prefix
            
        Returns:
            List of EmbeddingResponse objects
        """
        results = []
        for i, text in enumerate(texts):
            req_id = f"{request_id}-{i}" if request_id else None
            result = self.create(text, req_id)
            results.append(result)
        return results
    
    async def acreate_batch(
        self,
        texts: List[str],
        request_id: Optional[str] = None,
    ) -> List[EmbeddingResponse]:
        """Create embeddings for multiple texts asynchronously.
        
        Note: This method makes multiple API calls as the API
        doesn't support batch embedding natively.
        
        Args:
            texts: List of texts to embed
            request_id: Optional request ID prefix
            
        Returns:
            List of EmbeddingResponse objects
        """
        import asyncio
        
        tasks = []
        for i, text in enumerate(texts):
            req_id = f"{request_id}-{i}" if request_id else None
            task = self.acreate(text, req_id)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
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