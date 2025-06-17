# QA Benchmark Agents

이 프로젝트는 다양한 QA 벤치마크에 대해 CoT (Chain of Thought), ReAct (Reasoning and Acting), Reflexion (Self-reflection) 추론 방법론을 사용하는 에이전트들을 제공합니다.

## 지원되는 추론 방법론

### 1. Chain of Thought (CoT)
- **특징**: 단계별 논리적 추론
- **사용법**: 복잡한 문제를 작은 단위로 분해하여 체계적으로 해결
- **적합한 작업**: 수학 문제, 논리적 추론, 단계별 분석이 필요한 작업

### 2. ReAct (Reasoning and Acting)
- **특징**: 생각-행동-관찰 사이클
- **사용법**: 추론과 행동을 번갈아가며 실행
- **적합한 작업**: 정보 검색이 필요한 작업, 동적 문제 해결

### 3. Reflexion (Self-reflection)
- **특징**: 자기 반성 및 개선
- **사용법**: 초기 답안을 생성한 후 자체 평가 및 개선
- **적합한 작업**: 복잡한 추론, 정확성이 중요한 작업

## 지원되는 QA 벤치마크

- **HotpotQA**: 멀티홉 추론 질문 답변
- **SQuAD**: 독해 기반 질문 답변  
- **Natural Questions**: 자연어 질문 답변
- **MS MARCO**: 검색 기반 질문 답변
- **CommonsenseQA**: 상식 기반 객관식 문제
- **DROP**: 수치 추론 문제
- **BoolQ**: 예/아니오 질문
- **QuAC**: 대화형 질문 답변
- **ARC**: 과학 상식 문제
- **TriviaQA**: 일반 상식 문제

## 설치 및 설정

```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# 환경 변수 설정 (OpenAI API 키)
export OPENAI_API_KEY="your-api-key-here"
```

## 사용법

### 1. 기본 사용법

```python
from graphs.agent.experiments.hotpotqa_agent import (
    create_cot_qa_agent,
    create_react_qa_agent,
    create_reflexion_qa_agent,
    create_qa_agent
)

# CoT Agent 생성
cot_agent = create_cot_qa_agent(benchmark="hotpotqa")

# ReAct Agent 생성  
react_agent = create_react_qa_agent(benchmark="squad")

# Reflexion Agent 생성
reflexion_agent = create_reflexion_qa_agent(benchmark="naturalqa")

# 통합 함수로 생성
agent = create_qa_agent("react", benchmark="hotpotqa")
```

### 2. 질문 답변 실행

```python
# 질문 준비
question = "What is the capital of the country where the Eiffel Tower is located?"

# 에이전트 실행
result = agent.invoke({
    "messages": [{"role": "user", "content": question}]
})

# 답변 출력
print(result["messages"][-1].content)
```

### 3. 대화형 실행

```python
from graphs.agent.experiments.hotpotqa_agent import run_qa_agent_interactive

# 대화형 모드로 실행
agent = create_react_qa_agent(benchmark="hotpotqa")
run_qa_agent_interactive(agent)
```

### 4. 커스텀 도구 추가

```python
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """웹에서 정보를 검색합니다."""
    # 실제 웹 검색 구현
    return f"웹 검색 결과: {query}"

# 커스텀 도구와 함께 에이전트 생성
agent = create_react_qa_agent(
    benchmark="naturalqa",
    tools=[web_search]
)
```

## 테스트 실행

```bash
# 테스트 스크립트 실행
python graphs/agent/experiments/test_qa_agents.py
```

## 프롬프트 커스터마이징

`graphs/agent/prompts/qa_prompt.yaml` 파일을 수정하여 각 벤치마크와 추론 방법론에 맞는 프롬프트를 커스터마이즈할 수 있습니다.

```yaml
prompts:
  hotpotqa:
    cot:
      template: |
        Your custom CoT prompt here...
    react:
      template: |
        Your custom ReAct prompt here...
    reflexion:
      template: |
        Your custom Reflexion prompt here...
```

## 예제

### 멀티홉 추론 예제 (HotpotQA)

```python
agent = create_cot_qa_agent(benchmark="hotpotqa")

question = """
Who directed the movie that won the Academy Award for Best Picture in 2010, 
and what other famous movie did this director make in 2008?
"""

result = agent.invoke({
    "messages": [{"role": "user", "content": question}]
})

print(result["messages"][-1].content)
```

### 수치 추론 예제 (DROP)

```python
agent = create_react_qa_agent(benchmark="drop")

question = """
Based on the following passage: "The company had 1,250 employees in 2019, 
increased by 15% in 2020, and then decreased by 8% in 2021." 
How many employees did the company have in 2021?
"""

result = agent.invoke({
    "messages": [{"role": "user", "content": question}]
})

print(result["messages"][-1].content)
```

## 아키텍처

```
QA Agent
├── Prompt Template (from YAML)
├── Language Model (GPT-4, Claude, etc.)
├── Tools
│   ├── search_documents
│   ├── extract_information  
│   ├── verify_facts
│   └── calculate_numbers
└── LangGraph Framework
```

## 기여하기

1. 새로운 벤치마크 지원을 위해 `qa_prompt.yaml`에 프롬프트 추가
2. 새로운 도구 개발 (`@tool` 데코레이터 사용)
3. 새로운 추론 방법론 구현
4. 성능 평가 및 벤치마크 결과 공유

## 라이센스

이 프로젝트는 MIT 라이센스 하에 제공됩니다.
