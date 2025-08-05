"""Custom exceptions for HyperCLOVA API."""

from typing import Optional, Dict, Any


class HyperClovaError(Exception):
    """Base exception for HyperCLOVA API errors."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.response = response


class AuthenticationError(HyperClovaError):
    """Raised when authentication fails."""
    pass


class APIConnectionError(HyperClovaError):
    """Raised when unable to connect to the API."""
    pass


class APITimeoutError(HyperClovaError):
    """Raised when a request times out."""
    pass


class RateLimitError(HyperClovaError):
    """Raised when rate limit is exceeded."""
    pass


class InvalidRequestError(HyperClovaError):
    """Raised when the request is invalid."""
    pass


class ServerError(HyperClovaError):
    """Raised when the server returns a 5xx error."""
    pass


class StreamingError(HyperClovaError):
    """Raised when there's an error during streaming."""
    pass


class ModelNotSupportedError(HyperClovaError):
    """Raised when the model doesn't support a requested feature."""
    pass


class TokenLimitExceededError(HyperClovaError):
    """Raised when token limit is exceeded."""
    pass


def raise_for_status_code(status_code: int, response: Dict[str, Any]) -> None:
    """Raise appropriate exception based on status code."""
    message = response.get("status", {}).get("message", "Unknown error")
    code = response.get("status", {}).get("code", str(status_code))
    
    if status_code == 401:
        raise AuthenticationError(
            message="Authentication failed. Invalid API key.",
            code=code,
            status_code=status_code,
            response=response
        )
    elif status_code == 403:
        raise AuthenticationError(
            message="Access forbidden. Check your API key permissions.",
            code=code,
            status_code=status_code,
            response=response
        )
    elif status_code == 404:
        raise InvalidRequestError(
            message="Resource not found.",
            code=code,
            status_code=status_code,
            response=response
        )
    elif status_code == 429:
        raise RateLimitError(
            message="Rate limit exceeded.",
            code=code,
            status_code=status_code,
            response=response
        )
    elif 400 <= status_code < 500:
        raise InvalidRequestError(
            message=message,
            code=code,
            status_code=status_code,
            response=response
        )
    elif 500 <= status_code < 600:
        raise ServerError(
            message=f"Server error: {message}",
            code=code,
            status_code=status_code,
            response=response
        )
    else:
        raise HyperClovaError(
            message=f"Unexpected status code: {status_code}",
            code=code,
            status_code=status_code,
            response=response
        )