# HyperCLOVA Python Library

A Python library for interacting with HyperCLOVA X APIs, providing chat completions and embeddings functionality similar to OpenAI and LangChain libraries.

## Features

- 🤖 **Chat Completions**: Support for HCX-005, HCX-007, and HCX-DASH-002 models
- 🧠 **Thinking Mode**: Advanced reasoning capabilities with HCX-007
- 🛠️ **Function Calling**: Tool integration for dynamic functionality
- 📋 **Structured Output**: JSON schema-based responses (HCX-007 only)
- 👁️ **Vision Support**: Multimodal capabilities with HCX-005
- 🔢 **Embeddings**: 1024-dimensional text embeddings
- 🌊 **Streaming**: Real-time response streaming with SSE
- ⚡ **Async Support**: Full async/await compatibility
- 🔐 **Type Safety**: Complete type hints with Pydantic models

## Installation

```bash
# Install from GitHub with uv
uv pip install git+https://github.com/chalju/HyperClovaX.git

# Or clone and install locally with uv
git clone https://github.com/chalju/HyperClovaX.git
cd HyperClovaX
uv pip install -e .

# Or install with standard pip
pip install git+https://github.com/chalju/HyperClovaX.git

# For development
git clone https://github.com/chalju/HyperClovaX.git
cd HyperClovaX
uv pip install -e ".[dev]"
```

## Quick Start

```python
from hyperclova import HyperClova

# Initialize client (uses HYPERCLOVA_API_KEY env var by default)
client = HyperClova(api_key="your-api-key")

# Basic chat completion
response = client.chat.completions.create(
    model="HCX-007",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello! What can you do?"}
    ],
    temperature=0.7,
    max_tokens=1024
)
print(response.message.content)
```

## Examples

### Thinking Mode (HCX-007)

```python
response = client.chat.completions.create(
    model="HCX-007",
    messages=[
        {"role": "user", "content": "Solve this complex problem: If a train travels..."}
    ],
    thinking={"effort": "medium"},  # none, low, medium, high
    max_completion_tokens=10240
)

print(f"Thinking process: {response.message.thinking_content}")
print(f"Final answer: {response.message.content}")
```

### Function Calling

```python
# Define a tool
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}

response = client.chat.completions.create(
    model="HCX-005",
    messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
    tools=[weather_tool],
    tool_choice="auto"
)

if response.message.tool_calls:
    tool_call = response.message.tool_calls[0]
    print(f"Function: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

### Structured Output (HCX-007)

```python
response = client.chat.completions.create(
    model="HCX-007",
    messages=[
        {"role": "user", "content": "Extract person info from: John Doe, 30, engineer"}
    ],
    response_format={
        "type": "json",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "job": {"type": "string"}
            },
            "required": ["name", "age", "job"]
        }
    }
)

import json
data = json.loads(response.message.content)
print(data)  # {"name": "John Doe", "age": 30, "job": "engineer"}
```

### Multimodal (HCX-005)

```python
# With image URL
response = client.chat.completions.create(
    model="HCX-005",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }]
)

# With base64 image
response = client.chat.completions.create(
    model="HCX-005",
    messages=[{
        "role": "user", 
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "data_uri": {"data": "base64_encoded_image_data"}}
        ]
    }]
)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="HCX-007",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True
)

for chunk in stream:
    if chunk.message.content:
        print(chunk.message.content, end="", flush=True)
```

### Embeddings

```python
# Single embedding
response = client.embeddings.create(
    text="HyperCLOVA X is powerful"
)
print(f"Dimension: {response.dimension}")  # 1024
print(f"Vector: {response.embedding[:5]}...")

# Batch embeddings
texts = ["First text", "Second text", "Third text"]
embeddings = client.embeddings.create_batch(texts)
```

### Async Support

```python
import asyncio

async def async_example():
    # Async completion
    response = await client.chat.completions.acreate(
        model="HCX-007",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # Async streaming
    stream = await client.chat.completions.acreate(
        model="HCX-007",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.message.content:
            print(chunk.message.content, end="")

asyncio.run(async_example())
```

## Models

| Model | Description | Max Tokens | Features |
|-------|-------------|------------|----------|
| HCX-005 | Vision model | 128K total, 4096 output | Multimodal, Function calling |
| HCX-007 | Reasoning model | 128K total, 32768 output | Thinking mode, Structured output, Function calling |
| HCX-DASH-002 | Lightweight model | 32K total, 4096 output | Fast responses |

**Note**: 
- HCX-007 requires `maxCompletionTokens` parameter instead of `maxTokens`
- Structured output (`responseFormat`) with HCX-007 requires `thinking.effort = "none"` (automatically handled by the library)
- When using structured output, thinking mode cannot be used with other effort levels

## API Configuration

### Environment Variables

- `HYPERCLOVA_API_KEY`: Your API key
- `HYPERCLOVA_BASE_URL`: API base URL (default: https://clovastudio.stream.ntruss.com)

### Client Configuration

```python
client = HyperClova(
    api_key="your-api-key",
    base_url="https://custom.api.url",  # Optional
    timeout=30.0,  # Request timeout in seconds
    max_retries=3  # Max retry attempts
)
```

## Error Handling

```python
from hyperclova import (
    HyperClovaError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError
)

try:
    response = client.chat.completions.create(...)
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded")
except InvalidRequestError as e:
    print(f"Invalid request: {e}")
except HyperClovaError as e:
    print(f"API error: {e}")
```

## LangChain Integration

This library is designed to be compatible with LangChain patterns:

```python
# Example LangChain-style usage
from hyperclova import HyperClova

llm = HyperClova(api_key="your-key")

# Use with LangChain (implementation example)
response = llm.chat.completions.create(
    model="HCX-007",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.