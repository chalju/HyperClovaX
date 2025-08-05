"""Server-Sent Events (SSE) streaming support for HyperCLOVA API."""

import json
import logging
from typing import AsyncIterator, Iterator, Optional, Dict, Any, Union
from dataclasses import dataclass

import httpx

from .exceptions import StreamingError
from .models import ChatCompletionChunk

logger = logging.getLogger(__name__)


@dataclass
class SSEEvent:
    """Server-Sent Event."""
    id: Optional[str] = None
    event: Optional[str] = None
    data: Optional[str] = None
    retry: Optional[int] = None


class SSEParser:
    """Parser for Server-Sent Events."""
    
    def __init__(self):
        self.buffer = ""
        
    def parse(self, chunk: str) -> Iterator[SSEEvent]:
        """Parse SSE chunk and yield events."""
        self.buffer += chunk
        
        while "\n\n" in self.buffer:
            event_str, self.buffer = self.buffer.split("\n\n", 1)
            event = self._parse_event(event_str)
            if event:
                yield event
    
    def _parse_event(self, event_str: str) -> Optional[SSEEvent]:
        """Parse a single SSE event."""
        if not event_str.strip():
            return None
        
        event = SSEEvent()
        
        for line in event_str.split("\n"):
            if not line:
                continue
                
            if ":" not in line:
                continue
            
            field, value = line.split(":", 1)
            value = value.lstrip()
            
            if field == "id":
                event.id = value
            elif field == "event":
                event.event = value
            elif field == "data":
                event.data = value
            elif field == "retry":
                try:
                    event.retry = int(value)
                except ValueError:
                    pass
        
        return event if event.data else None


class StreamingResponse:
    """Wrapper for streaming response."""
    
    def __init__(self, response: httpx.Response):
        self.response = response
        self.parser = SSEParser()
        
    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        """Iterate over streaming chunks."""
        try:
            for line in self.response.iter_lines():
                if not line:
                    continue
                
                for event in self.parser.parse(line + "\n\n"):
                    chunk = self._process_event(event)
                    if chunk:
                        yield chunk
        except Exception as e:
            raise StreamingError(f"Error during streaming: {e}")
        finally:
            self.response.close()
    
    def _process_event(self, event: SSEEvent) -> Optional[ChatCompletionChunk]:
        """Process SSE event into ChatCompletionChunk."""
        if not event.data:
            return None
        
        try:
            data = json.loads(event.data)
            
            # Handle error events
            if event.event == "error":
                raise StreamingError(
                    f"Streaming error: {data.get('status', {}).get('message', 'Unknown error')}"
                )
            
            # Handle result event (final event)
            if event.event == "result":
                # Convert result format to chunk format
                return ChatCompletionChunk(**data)
            
            # Handle token event
            if event.event == "token":
                return ChatCompletionChunk(**data)
            
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse SSE data: {event.data}")
            raise StreamingError(f"Invalid JSON in SSE: {e}")


class AsyncStreamingResponse:
    """Async wrapper for streaming response."""
    
    def __init__(self, response: httpx.Response):
        self.response = response
        self.parser = SSEParser()
    
    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        """Iterate over streaming chunks asynchronously."""
        try:
            async for line in self.response.aiter_lines():
                if not line:
                    continue
                
                for event in self.parser.parse(line + "\n\n"):
                    chunk = self._process_event(event)
                    if chunk:
                        yield chunk
        except Exception as e:
            raise StreamingError(f"Error during streaming: {e}")
        finally:
            await self.response.aclose()
    
    def _process_event(self, event: SSEEvent) -> Optional[ChatCompletionChunk]:
        """Process SSE event into ChatCompletionChunk."""
        if not event.data:
            return None
        
        try:
            data = json.loads(event.data)
            
            # Handle error events
            if event.event == "error":
                raise StreamingError(
                    f"Streaming error: {data.get('status', {}).get('message', 'Unknown error')}"
                )
            
            # Handle result event (final event)
            if event.event == "result":
                return ChatCompletionChunk(**data)
            
            # Handle token event
            if event.event == "token":
                return ChatCompletionChunk(**data)
            
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse SSE data: {event.data}")
            raise StreamingError(f"Invalid JSON in SSE: {e}")


def create_streaming_response(
    response: httpx.Response,
    is_async: bool = False
) -> Union[StreamingResponse, AsyncStreamingResponse]:
    """Create appropriate streaming response wrapper."""
    if is_async:
        return AsyncStreamingResponse(response)
    return StreamingResponse(response)