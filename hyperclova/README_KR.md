# HyperCLOVA Python 라이브러리

HyperCLOVA X API와 상호작용하기 위한 Python 라이브러리로, OpenAI 및 LangChain 라이브러리와 유사한 채팅 완성 및 임베딩 기능을 제공합니다.

## 주요 기능

- 🤖 **채팅 완성**: HCX-005, HCX-007, HCX-DASH-002 모델 지원
- 🧠 **사고 모드**: HCX-007의 고급 추론 기능
- 🛠️ **함수 호출**: 동적 기능을 위한 도구 통합
- 📋 **구조화된 출력**: JSON 스키마 기반 응답 (HCX-007 전용)
- 👁️ **비전 지원**: HCX-005의 멀티모달 기능
- 🔢 **임베딩**: 1024차원 텍스트 임베딩
- 🌊 **스트리밍**: SSE를 활용한 실시간 응답 스트리밍
- ⚡ **비동기 지원**: 완전한 async/await 호환성
- 🔐 **타입 안전성**: Pydantic 모델을 사용한 완전한 타입 힌트

## 설치

```bash
# GitHub에서 uv로 설치
uv pip install git+https://github.com/chalju/HyperClovaX.git

# 또는 로컬에서 클론 후 uv로 설치
git clone https://github.com/chalju/HyperClovaX.git
cd HyperClovaX
uv pip install -e .

# 또는 표준 pip로 설치
pip install git+https://github.com/chalju/HyperClovaX.git

# 개발용
git clone https://github.com/chalju/HyperClovaX.git
cd HyperClovaX
uv pip install -e ".[dev]"
```

## 빠른 시작

```python
from hyperclova import HyperClova

# 클라이언트 초기화 (기본적으로 HYPERCLOVA_API_KEY 환경 변수 사용)
client = HyperClova(api_key="your-api-key")

# 기본 채팅 완성
response = client.chat.completions.create(
    model="HCX-007",
    messages=[
        {"role": "system", "content": "당신은 도움이 되는 어시스턴트입니다"},
        {"role": "user", "content": "안녕하세요! 무엇을 할 수 있나요?"}
    ],
    temperature=0.7,
    max_tokens=1024
)
print(response.message.content)
```

## 예제

### 사고 모드 (HCX-007)

```python
response = client.chat.completions.create(
    model="HCX-007",
    messages=[
        {"role": "user", "content": "이 복잡한 문제를 해결해주세요: 기차가 시속 80km로..."}
    ],
    thinking={"effort": "medium"},  # none, low, medium, high
    max_completion_tokens=10240
)

print(f"사고 과정: {response.message.thinking_content}")
print(f"최종 답변: {response.message.content}")
```

### 함수 호출

```python
# 도구 정의
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "날씨 정보 가져오기",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "도시 이름"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}

response = client.chat.completions.create(
    model="HCX-005",
    messages=[{"role": "user", "content": "서울의 날씨는 어때?"}],
    tools=[weather_tool],
    tool_choice="auto"
)

if response.message.tool_calls:
    tool_call = response.message.tool_calls[0]
    print(f"함수: {tool_call.function.name}")
    print(f"인자: {tool_call.function.arguments}")
```

### 구조화된 출력 (HCX-007)

```python
response = client.chat.completions.create(
    model="HCX-007",
    messages=[
        {"role": "user", "content": "다음에서 사람 정보를 추출하세요: 홍길동, 30세, 엔지니어"}
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
print(data)  # {"name": "홍길동", "age": 30, "job": "엔지니어"}
```

### 멀티모달 (HCX-005)

```python
# 이미지 URL 사용
response = client.chat.completions.create(
    model="HCX-005",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "이 이미지에 무엇이 있나요?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }]
)

# base64 이미지 사용
response = client.chat.completions.create(
    model="HCX-005",
    messages=[{
        "role": "user", 
        "content": [
            {"type": "text", "text": "이 이미지를 설명해주세요"},
            {"type": "image_url", "data_uri": {"data": "base64_encoded_image_data"}}
        ]
    }]
)
```

### 스트리밍

```python
stream = client.chat.completions.create(
    model="HCX-007",
    messages=[{"role": "user", "content": "이야기를 써주세요"}],
    stream=True
)

for chunk in stream:
    if chunk.message.content:
        print(chunk.message.content, end="", flush=True)
```

### 임베딩

```python
# 단일 임베딩
response = client.embeddings.create(
    text="HyperCLOVA X는 강력합니다"
)
print(f"차원: {response.dimension}")  # 1024
print(f"벡터: {response.embedding[:5]}...")

# 배치 임베딩
texts = ["첫 번째 텍스트", "두 번째 텍스트", "세 번째 텍스트"]
embeddings = client.embeddings.create_batch(texts)
```

### 비동기 지원

```python
import asyncio

async def async_example():
    # 비동기 완성
    response = await client.chat.completions.acreate(
        model="HCX-007",
        messages=[{"role": "user", "content": "안녕하세요!"}]
    )
    
    # 비동기 스트리밍
    stream = await client.chat.completions.acreate(
        model="HCX-007",
        messages=[{"role": "user", "content": "5까지 세어주세요"}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.message.content:
            print(chunk.message.content, end="")

asyncio.run(async_example())
```

## 모델

| 모델 | 설명 | 최대 토큰 | 기능 |
|------|------|-----------|------|
| HCX-005 | 비전 모델 | 총 128K, 출력 4096 | 멀티모달, 함수 호출 |
| HCX-007 | 추론 모델 | 총 128K, 출력 32768 | 사고 모드, 구조화된 출력, 함수 호출 |
| HCX-DASH-002 | 경량 모델 | 총 32K, 출력 4096 | 빠른 응답 |

**참고**: 
- HCX-007은 `maxTokens` 대신 `maxCompletionTokens` 매개변수가 필요합니다
- HCX-007의 구조화된 출력 (`responseFormat`)은 `thinking.effort = "none"`이 필요합니다 (라이브러리에서 자동 처리)
- 구조화된 출력 사용 시, 사고 모드는 다른 effort 레벨과 함께 사용할 수 없습니다

## API 설정

### 환경 변수

- `HYPERCLOVA_API_KEY`: API 키
- `HYPERCLOVA_BASE_URL`: API 기본 URL (기본값: https://clovastudio.stream.ntruss.com)

### 클라이언트 설정

```python
client = HyperClova(
    api_key="your-api-key",
    base_url="https://custom.api.url",  # 선택사항
    timeout=30.0,  # 요청 타임아웃(초)
    max_retries=3  # 최대 재시도 횟수
)
```

## 오류 처리

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
    print("잘못된 API 키")
except RateLimitError:
    print("요청 한도 초과")
except InvalidRequestError as e:
    print(f"잘못된 요청: {e}")
except HyperClovaError as e:
    print(f"API 오류: {e}")
```

## LangChain 통합

이 라이브러리는 LangChain 패턴과 호환되도록 설계되었습니다:

```python
# LangChain 스타일 사용 예제
from hyperclova import HyperClova

llm = HyperClova(api_key="your-key")

# LangChain과 함께 사용 (구현 예제)
response = llm.chat.completions.create(
    model="HCX-007",
    messages=[{"role": "user", "content": "안녕하세요"}]
)
```

## 라이선스

MIT 라이선스

## 기여하기

기여를 환영합니다! Pull Request를 자유롭게 제출해 주세요.