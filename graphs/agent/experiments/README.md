# Math Benchmark Agents

이 프로젝트는 수학 문제 해결을 위한 세 가지 다른 추론 방법론을 구현한 에이전트들을 제공합니다:

- **CoT (Chain of Thought)**: 단계별 추론을 통한 문제 해결
- **ReAct (Reasoning and Acting)**: 사고-행동-관찰 사이클을 통한 문제 해결  
- **Reflexion**: 자기 성찰을 통한 개선된 문제 해결

## 🗂️ 파일 구조

```
graphs/agent/experiments/
├── math_agents.py              # 메인 에이전트 생성 함수들
├── math_agent.py              # 업데이트된 클래스 기반 래퍼
├── math_agent_evaluator.py    # 에이전트 평가 유틸리티
├── demo_math_agents.py        # 데모 스크립트
├── simple_test.py             # 간단한 테스트 스크립트
└── README.md                  # 이 파일

graphs/agent/prompts/
└── math_prompt.yaml           # 각 방법론별 프롬프트 템플릿
```

## 🛠️ 주요 기능

### 1. 에이전트 생성 함수들

```python
from math_agents import (
    create_cot_math_agent,
    create_react_math_agent,
    create_reflexion_math_agent,
    create_math_agent
)

# 특정 타입의 에이전트 생성
cot_agent = create_cot_math_agent(model="openai:gpt-4o-mini", benchmark="gsm8k")
react_agent = create_react_math_agent(model="openai:gpt-4o-mini", benchmark="gsm8k")
reflexion_agent = create_reflexion_math_agent(model="openai:gpt-4o-mini", benchmark="gsm8k")

# 통합 함수로 에이전트 생성
agent = create_math_agent(
    agent_type="cot",  # "cot", "react", "reflexion"
    model="openai:gpt-4o-mini",
    benchmark="gsm8k"  # "basic", "gsm8k", "competition", "aime"
)
```

### 2. 수학 도구들

에이전트들이 사용할 수 있는 수학 계산 도구들:

- `add_numbers(a, b)`: 덧셈
- `subtract_numbers(a, b)`: 뺄셈
- `multiply_numbers(a, b)`: 곱셈
- `divide_numbers(a, b)`: 나눗셈
- `calculate_power(base, exponent)`: 거듭제곱
- `calculate_sqrt(number)`: 제곱근
- `calculate_factorial(n)`: 팩토리얼

### 3. 벤치마크 타입

- **basic**: 기본 수학 문제
- **gsm8k**: 초등학교 수준의 문장제 문제
- **competition**: 수학 경시대회 수준의 문제
- **aime**: AIME(American Invitational Mathematics Examination) 수준

## 🚀 사용 방법

### 기본 사용법

```python
# 에이전트 생성
agent = create_cot_math_agent(benchmark="gsm8k")

# 문제 해결
problem = "Sarah has 15 apples. She gives 3 to her friend and buys 8 more. How many apples does she have?"
result = agent.invoke({"messages": [{"role": "user", "content": problem}]})
print(result["messages"][-1].content)
```

### 클래스 기반 사용법

```python
from math_agent import MathAgentRunner

# 러너 생성
runner = MathAgentRunner(
    agent_type="react",
    benchmark="gsm8k",
    model="openai:gpt-4o-mini"
)

# 인터랙티브 모드 실행
runner.run_interactive()

# 또는 단일 문제 해결
result = runner.solve_problem("What is 15 + 23 * 4?")
print(result)
```

### 데모 실행

```bash
# 현재 디렉토리에서
cd graphs/agent/experiments

# 간단한 테스트
python simple_test.py

# 전체 데모 (모든 방법론 비교)
python demo_math_agents.py

# 특정 데모 실행
python demo_math_agents.py arithmetic    # 사칙연산 데모
python demo_math_agents.py word         # 문장제 데모
python demo_math_agents.py competition  # 경시 문제 데모

# 인터랙티브 모드
python demo_math_agents.py interactive
```

## 📊 에이전트 평가

```python
from math_agent_evaluator import MathAgentEvaluator

# 평가기 생성
evaluator = MathAgentEvaluator()

# 문제들과 정답
problems = [
    ("What is 5 + 3?", "8"),
    ("Sarah has 10 apples and buys 5 more. How many does she have?", "15")
]

# 평가 실행
results = evaluator.evaluate_benchmark(problems, "gsm8k")
report = evaluator.generate_report(results)
evaluator.print_report(report)
```

## 🎯 각 방법론의 특징

### Chain of Thought (CoT)
- **특징**: 단계별로 명확한 추론 과정을 거쳐 문제를 해결
- **장점**: 논리적이고 체계적인 접근, 추론 과정 이해 용이
- **사용 상황**: 복잡한 다단계 계산이 필요한 문제

### ReAct (Reasoning and Acting)  
- **특징**: 생각→행동→관찰의 사이클을 반복하며 문제 해결
- **장점**: 도구 사용이 자연스럽고, 중간 결과를 활용한 적응적 추론
- **사용 상황**: 계산 도구 사용이 빈번한 문제, 동적 문제 해결

### Reflexion
- **특징**: 초기 해답을 자기 성찰을 통해 개선하며 문제 해결
- **장점**: 오류 자가 수정 능력, 해답의 정확성 향상
- **사용 상황**: 높은 정확성이 요구되는 문제, 복잡한 추론이 필요한 문제

## 🧪 테스트 결과 예시

```
🧮 Testing Math Agents
==================================================
Problem: What is 5 + 3 * 2?
Expected answer: 11 (order of operations: 3*2=6, then 5+6=11)

🔍 Testing CoT Agent...
✅ CoT Result:
**Step 1: Problem Comprehension**
The problem is asking to evaluate the expression 5 + 3 * 2. According to the order of operations...
**Final Answer:** 11

🔍 Testing ReAct Agent...  
✅ ReAct Result:
**Thought 1**: I need to evaluate this mathematical expression...
**Final Result**: The answer to the problem 5 + 3 × 2 is 11.

🔍 Testing Reflexion Agent...
✅ Reflexion Result:
**Initial Mathematical Attempt**: Let me first solve this math problem...
**Final Answer:** 11
```

## ⚙️ 설정 및 요구사항

### 환경 변수
`.env` 파일에 OpenAI API 키를 설정해야 합니다:
```
OPENAI_API_KEY=your_api_key_here
```

### 의존성
- `langgraph`
- `langchain_core` 
- `pydantic`
- `python-dotenv`
- `pyyaml`

## 🔧 커스터마이징

### 새로운 벤치마크 추가
`math_prompt.yaml` 파일에 새로운 벤치마크와 프롬프트를 추가할 수 있습니다.

### 새로운 도구 추가
`math_agents.py`의 `MATH_TOOLS` 리스트에 새로운 `@tool` 함수를 추가할 수 있습니다.

### 모델 변경
모든 함수에서 `model` 매개변수를 통해 다른 언어 모델을 사용할 수 있습니다.

## 🤝 기여하기

1. 새로운 추론 방법론 구현
2. 추가 벤치마크 지원
3. 평가 메트릭 개선
4. 사용자 경험 개선

## 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.
