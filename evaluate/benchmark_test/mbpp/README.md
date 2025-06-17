# MBPP 벤치마크 평가

MBPP (Mostly Basic Python Programming) 데이터셋을 사용하여 CoT, ReAct, Reflexion 에이전트들의 성능을 평가합니다.

## 파일 구조

```
evaluate/benchmark_test/mbpp/
├── evaluator.py          # 메인 평가 클래스
├── run_evaluation.py     # 실행 스크립트
├── demo.py              # 데모 (데이터셋 없이 테스트)
└── README.md            # 이 파일
```

## 설치

```bash
# 필요한 패키지 설치
pip install datasets pandas langchain langchain-openai langgraph pyyaml

# 환경변수 설정
export OPENAI_API_KEY="your-openai-api-key"
```

## 사용법

### 1. 빠른 데모 (2개 문제)

```bash
cd evaluate/benchmark_test/mbpp
python demo.py
```

### 2. 실제 MBPP 데이터셋으로 평가

```bash
# 빠른 테스트 (2개 문제)
python run_evaluation.py --mode quick

# 전체 테스트 (10개 문제)  
python run_evaluation.py --mode full

# 커스텀 설정
python run_evaluation.py --problems 5 --model gpt-4o-mini
```

### 3. Python 코드로 직접 실행

```python
import asyncio
from evaluator import MBPPEvaluator

async def main():
    # 평가기 생성
    evaluator = MBPPEvaluator(model="gpt-4o-mini", max_problems=5)
    
    # 평가 실행
    results = await evaluator.evaluate_all_agents(max_concurrent=1)
    
    # 결과 출력
    evaluator.print_summary()
    evaluator.save_results()

asyncio.run(main())
```

## 평가 지표

- **성공률 (Success Rate)**: 에이전트가 오류 없이 솔루션을 생성한 비율
- **테스트 통과율 (Test Pass Rate)**: 생성된 코드가 모든 테스트 케이스를 통과한 비율  
- **평균 실행 시간**: 문제당 평균 해결 시간
- **테스트 성공률**: 개별 테스트 케이스 통과 비율

## 출력 파일

- `mbpp_evaluation_results.csv`: 요약 결과 (CSV)
- `mbpp_evaluation_results_detailed.json`: 상세 결과 (JSON)

## 결과 예시

```
=== MBPP 벤치마크 평가 결과 요약 ===
총 문제 수: 5
총 평가 수: 15
평가 추론 방식: cot, react, reflexion

전체 성공률: 86.67%
전체 테스트 통과율: 73.33%
전체 평균 실행 시간: 45.23초
전체 평균 테스트 성공률: 78.50%

--- 추론 방식별 성과 ---

COT:
  성공률: 80.00%
  테스트 통과율: 60.00%
  평균 실행 시간: 35.40초
  평균 테스트 성공률: 72.00%

REACT:
  성공률: 100.00%
  테스트 통과율: 80.00%
  평균 실행 시간: 52.10초
  평균 테스트 성공률: 85.00%

REFLEXION:
  성공률: 80.00%
  테스트 통과율: 80.00%
  평균 실행 시간: 48.20초
  평균 테스트 성공률: 78.50%
```

## 주의사항

1. **API 한도**: OpenAI API 사용량에 주의하세요
2. **실행 시간**: 각 문제당 1-2분 소요될 수 있습니다
3. **동시 실행**: `max_concurrent` 값을 조정하여 병렬 처리 수를 제어하세요
4. **타임아웃**: 복잡한 문제는 타임아웃이 발생할 수 있습니다

## 에이전트 종류

### CoT (Chain of Thought)
- 단계별 논리적 추론
- 문제를 작은 단위로 분해하여 해결
- 명확한 설명과 함께 진행

### ReAct (Reasoning + Acting)  
- Thought-Action-Observation 사이클
- 추론과 행동을 반복하여 해결
- 도구 사용을 통한 검증

### Reflexion
- 자기 반성적 접근
- 초기 솔루션을 비판적으로 분석
- 개선된 솔루션으로 발전

## 문제 해결

### 일반적인 오류들

1. **ImportError**: 패키지 설치 확인
2. **API Error**: OpenAI API 키 확인
3. **TimeoutError**: 타임아웃 값 증가
4. **Memory Error**: 동시 실행 수 감소

### 디버깅

```python
# 단일 문제만 테스트
evaluator = MBPPEvaluator(max_problems=1)
result = await evaluator.evaluate_single_problem(
    problem, "cot", timeout=300
)
print(result.generated_solution)
```
