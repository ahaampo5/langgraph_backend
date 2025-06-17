# Math Benchmark Evaluation

이 디렉토리는 EleutherAI/hendrycks_math 데이터셋을 사용하여 CoT, ReAct, Reflexion 수학 에이전트들을 평가하는 코드를 포함합니다.

## 📋 데이터셋 정보

**EleutherAI/hendrycks_math** 데이터셋은 다음과 같은 수학 영역을 포함합니다:

- `algebra`: 대수학
- `counting_and_probability`: 조합론 및 확률
- `geometry`: 기하학  
- `intermediate_algebra`: 중급 대수학
- `number_theory`: 정수론
- `prealgebra`: 기초 대수학
- `precalculus`: 미적분학 예비과정

각 데이터 포인트는 다음 컬럼을 포함합니다:
- `problem`: 수학 문제 텍스트
- `level`: 난이도 (1-5, 5가 가장 어려움)
- `type`: 수학 영역 (예: "Algebra")
- `solution`: 정답 풀이 과정

## 🗂️ 파일 구조

```
evaluate/benchmark_test/math/
├── evaluate.py           # 메인 평가 스크립트
├── demo_evaluation.py    # 간단한 데모 스크립트
├── test_evaluation.py    # 테스트 스크립트
└── README.md            # 이 파일
```

## 🚀 사용 방법

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install datasets transformers

# 작업 디렉토리로 이동
cd /Users/admin/Desktop/workspace/my_github/langgraph_service
```

### 2. 간단한 테스트

```bash
# 평가 시스템 테스트
cd evaluate/benchmark_test/math
python test_evaluation.py
```

### 3. 데모 실행

```bash
# 몇 개의 문제로 간단한 데모
python demo_evaluation.py
```

### 4. 전체 평가 실행

```bash
# 모든 카테고리에 대한 전체 평가
python evaluate.py
```

## 📊 평가 결과

평가 스크립트는 다음과 같은 결과를 제공합니다:

### 전체 성능
- 각 에이전트별 정확도
- 평균 실행 시간
- 오류율

### 카테고리별 성능
- 수학 영역별 에이전트 성능 비교
- 각 영역에서의 정확도

### 난이도별 성능  
- 레벨 1-5별 성능 분석
- 난이도에 따른 에이전트 성능 변화

### 예시 출력

```
📊 MATH BENCHMARK EVALUATION REPORT
================================================================================
🎯 OVERALL RESULTS
----------------------------------------
COT Agent:
  📈 Accuracy: 67.50%
  ✅ Correct: 27/40
  ⏱️  Avg Time: 8.45s
  ❌ Errors: 2

REACT Agent:
  📈 Accuracy: 72.50%
  ✅ Correct: 29/40
  ⏱️  Avg Time: 12.30s
  ❌ Errors: 1

REFLEXION Agent:
  📈 Accuracy: 75.00%
  ✅ Correct: 30/40
  ⏱️  Avg Time: 15.60s
  ❌ Errors: 0

🏆 Best Overall: REFLEXION Agent (75.00%)

📚 RESULTS BY CATEGORY
----------------------------------------
Algebra:
  COT: 70.00% (7/10)
  REACT: 80.00% (8/10)
  REFLEXION: 80.00% (8/10)

Geometry:
  COT: 60.00% (6/10)
  REACT: 70.00% (7/10)
  REFLEXION: 75.00% (7.5/10)
```

## ⚙️ 설정 옵션

### evaluate.py 매개변수

```python
# 카테고리별 최대 문제 수 조정
max_problems_per_category = 20  # 기본값: 20

# 평가할 에이전트 타입 선택
agent_types = ["cot", "react", "reflexion"]  # 또는 일부만 선택

# 모델 변경
model = "openai:gpt-4o-mini"  # 또는 다른 모델
```

### 사용자 정의 평가

```python
from evaluate import MathBenchmarkEvaluator

# 평가기 생성
evaluator = MathBenchmarkEvaluator(model="openai:gpt-4o-mini")

# 특정 카테고리만 평가
results = evaluator.evaluate_category("algebra", max_problems=10)

# 결과 분석
report = evaluator.generate_report({"algebra": results})
evaluator.print_report(report)
```

## 🔧 문제 해결

### 일반적인 오류

1. **ImportError: math_agents**
   ```bash
   # 올바른 디렉토리에서 실행하세요
   cd /Users/admin/Desktop/workspace/my_github/langgraph_service
   ```

2. **Dataset loading error**
   ```bash
   # datasets 패키지 설치
   pip install datasets transformers
   ```

3. **API 키 오류**
   ```bash
   # .env 파일에 OpenAI API 키 설정
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

### 성능 최적화

- `max_problems_per_category`를 줄여서 더 빠른 테스트
- 특정 에이전트만 평가하여 시간 단축
- 더 작은 모델 사용 (예: gpt-3.5-turbo)

## 📈 결과 분석

### 예상되는 성능 패턴

1. **Reflexion > ReAct > CoT**: 일반적으로 자기 반성이 포함된 Reflexion이 가장 좋은 성능
2. **레벨별 성능 저하**: 난이도가 높아질수록 모든 에이전트의 성능 저하
3. **영역별 차이**: 기하학이 대수학보다 어려울 수 있음

### 결과 해석

- **정확도**: 문제를 올바르게 해결한 비율
- **실행 시간**: 평균 문제 해결 시간 (Reflexion이 가장 오래 걸림)
- **오류율**: 실행 중 발생한 오류 비율

## 🔄 확장 방법

### 새로운 평가 메트릭 추가

```python
def calculate_partial_credit(agent_answer: str, solution: str) -> float:
    """부분 점수 계산"""
    # 구현...
    pass
```

### 새로운 벤치마크 추가

```python
def evaluate_custom_dataset(self, dataset_name: str):
    """사용자 정의 데이터셋 평가"""
    # 구현...
    pass
```

### 결과 시각화

```python
import matplotlib.pyplot as plt

def plot_results(report):
    """결과 시각화"""
    # 구현...
    pass
```

## 📝 참고 자료

- [EleutherAI/hendrycks_math Dataset](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Math Agents Implementation](../../../graphs/agent/experiments/)
