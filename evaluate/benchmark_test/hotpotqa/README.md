# HotpotQA Benchmark Evaluation System

이 시스템은 HotpotQA 데이터셋을 사용하여 Chain of Thought (CoT), ReAct, Reflexion 방법론의 QA 성능을 평가합니다.

## 📁 파일 구조

```
evaluate/benchmark_test/hotpotqa/
├── evaluator.py          # 메인 평가기 (전체 기능)
├── simple_evaluator.py   # 간단한 평가기 (import 문제 해결)
├── run_benchmark.py      # 다양한 시나리오 실행 스크립트
├── test_evaluator.py     # 평가기 테스트 스크립트
├── analyze_results.py    # 결과 분석 (차트 포함)
├── simple_analyze.py     # 간단한 결과 분석
├── results/              # 평가 결과 저장 디렉토리
└── README.md            # 이 파일
```

## 🚀 빠른 시작

### 1. 테스트 명령어 (3개 질문)
```bash
cd /Users/admin/Desktop/workspace/my_github/langgraph_service
python evaluate/benchmark_test/hotpotqa/simple_evaluator.py --max-samples 3 --seed 42 --timestamp 20250617_143022
```

## 📊 평가 방법론

### Chain of Thought (CoT)
- **특징**: 단계별 논리적 추론
- **프롬프트**: 6단계 체계적 접근
  1. Question Analysis (질문 분석)
  2. Information Requirements (정보 요구사항)
  3. Step-by-step Reasoning (단계별 추론)
  4. Evidence Integration (증거 통합)
  5. Final Answer (최종 답변)
  6. Evidence Verification (증거 검증)

### ReAct (Reasoning and Acting)
- **특징**: Thought-Action-Observation 사이클
- **프롬프트**: 반복적 추론-행동-관찰 패턴
- **적합성**: 정보 검색이 필요한 복잡한 질문

### Reflexion (Self-reflection)
- **특징**: 자기 반성 및 답안 개선
- **프롬프트**: 초기 답안 → 자기 반성 → 개선된 답안
- **적합성**: 높은 정확성이 요구되는 작업

## 📋 평가 메트릭

### 기본 메트릭
- **Overall Accuracy**: 전체 정확도
- **Bridge Accuracy**: 브리지 질문 정확도 (멀티홉 추론)
- **Comparison Accuracy**: 비교 질문 정확도 (Yes/No)
- **Average Response Time**: 평균 응답 시간

### 질문 타입
- **Bridge Questions**: 여러 정보를 연결하는 멀티홉 추론 질문
- **Comparison Questions**: 두 엔티티를 비교하는 Yes/No 질문

## 🎯 사용 예시

### 기본 평가 실행
```bash
# 10개 질문으로 빠른 테스트
python evaluate/benchmark_test/hotpotqa/simple_evaluator.py --max-samples 10

# 특정 모델 사용
python evaluate/benchmark_test/hotpotqa/simple_evaluator.py --max-samples 20 --model "anthropic:claude-3-sonnet-latest"
```

### 결과 분석
```bash
# 간단한 텍스트 분석
python evaluate/benchmark_test/hotpotqa/simple_analyze.py

# 차트 포함 상세 분석 (matplotlib 필요)
python evaluate/benchmark_test/hotpotqa/analyze_results.py
```

### 대화형 벤치마크 실행
```bash
python evaluate/benchmark_test/hotpotqa/run_benchmark.py --mode interactive
```

## 📈 결과 해석

### 성능 지표
- **Accuracy > 0.8**: 우수한 성능
- **Accuracy 0.6-0.8**: 양호한 성능
- **Accuracy < 0.6**: 개선 필요

### 응답 시간
- **< 5초**: 실시간 애플리케이션에 적합
- **5-10초**: 일반적인 용도에 적합
- **> 10초**: 배치 처리에만 적합

### 질문 타입별 성능
- **Bridge Questions**: 복잡한 멀티홉 추론 능력 측정
- **Comparison Questions**: 간단한 비교 추론 능력 측정

## 📁 결과 파일

### 자동 생성 파일
```
results/
├── hotpotqa_summary_YYYYMMDD_HHMMSS.json      # 요약 통계
├── hotpotqa_detailed_results_YYYYMMDD_HHMMSS.json  # 상세 결과
├── simple_analysis_YYYYMMDD_HHMMSS.txt        # 텍스트 분석 보고서
└── performance_comparison_YYYYMMDD_HHMMSS.png # 성능 비교 차트
```

### 요약 통계 형식
```json
{
  "cot": {
    "total_questions": 10,
    "correct_answers": 7,
    "accuracy": 0.7,
    "avg_response_time": 8.5,
    "bridge_accuracy": 0.6,
    "comparison_accuracy": 0.9,
    "bridge_count": 6,
    "comparison_count": 4
  }
}
```

## ⚙️ 설정 옵션

### 환경 변수
```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"  # Anthropic 모델 사용시
```

### 명령줄 옵션
```bash
--max-samples N     # 평가할 질문 수 (기본: 10)
--model MODEL_NAME  # 사용할 모델 (기본: openai:gpt-4o-mini)
--workers N         # 병렬 처리 워커 수 (기본: 1)
```

## 🔧 문제해결

### 일반적인 오류

#### Import 오류
```bash
# 해결법: simple_evaluator.py 사용
python evaluate/benchmark_test/hotpotqa/simple_evaluator.py
```

#### 모델 API 오류
```bash
# API 키 확인
echo $OPENAI_API_KEY

# 다른 모델 시도
python simple_evaluator.py --model "anthropic:claude-3-haiku-latest"
```

#### 메모리 부족
```bash
# 샘플 수 줄이기
python simple_evaluator.py --max-samples 5 --workers 1
```

### 성능 최적화

#### 병렬 처리
```bash
# 더 많은 워커 사용 (신중하게)
python simple_evaluator.py --workers 3
```

#### 배치 크기 조정
```bash
# 작은 배치로 여러 번 실행
python simple_evaluator.py --max-samples 20
python simple_evaluator.py --max-samples 20  # 추가 실행
```

## 📊 예시 결과

### 성능 비교 예시
```
Method       Accuracy   Bridge     Comparison   Avg Time    
----------------------------------------------------------------------
COT          0.650      0.600      0.750        9.20s
REACT        0.700      0.680      0.800        7.50s
REFLEXION    0.750      0.720      0.850        8.80s
```

### 해석
- **REFLEXION**: 가장 높은 정확도 (75%)
- **REACT**: 가장 빠른 응답 시간 (7.5초)
- **COT**: 균형잡힌 성능

## 🚀 고급 사용법

### 커스텀 데이터셋
```python
# 커스텀 데이터 경로 지정
evaluator = SimpleHotpotQAEvaluator(
    data_path="/path/to/custom/dataset.json",
    max_samples=100
)
```

### 프롬프트 커스터마이징
```python
# graphs/agent/prompts/qa_prompt.yaml 파일 수정
# 각 방법론별 프롬프트 템플릿 변경 가능
```

### 결과 분석 자동화
```bash
# 평가 후 자동 분석
python simple_evaluator.py --max-samples 50 && python simple_analyze.py
```

## 📝 참고 사항

### 데이터셋 정보
- **파일**: `raw_data/hotpot_dev_v1_simplified.json`
- **총 질문 수**: 37,027개
- **질문 타입**: Bridge (멀티홉), Comparison (비교)
- **형식**: `{"question": "...", "answer": "...", "type": "..."}`

### 평가 기준
- **정확한 매칭**: 정규화된 답변이 정확히 일치
- **부분 매칭**: 예상 답변이 예측 답변에 포함
- **Yes/No 특별 처리**: 비교 질문의 경우 yes/no 패턴 인식

### 성능 고려사항
- 각 질문당 여러 API 호출 발생 (도구 사용으로 인해)
- 대규모 평가시 API 비용 고려 필요
- 응답 시간은 네트워크 상황에 따라 변동 가능

## 🤝 기여하기

### 새로운 평가 메트릭 추가
1. `evaluator.py`의 `_calculate_stats` 메서드 수정
2. 결과 출력 부분에 새 메트릭 추가

### 새로운 분석 기능 추가
1. `simple_analyze.py`에 새 분석 메서드 추가
2. 텍스트 보고서 형식 확장

### 버그 리포트
- 평가 결과가 예상과 다른 경우
- Import 오류나 런타임 오류 발생시
- 성능 문제나 메모리 사용량 이슈

---

**💡 팁**: 첫 사용시에는 `--max-samples 3`으로 시작해서 시스템이 정상 작동하는지 확인한 후 점진적으로 샘플 수를 늘려보세요!
