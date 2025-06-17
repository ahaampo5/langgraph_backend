"""
HotpotQA Benchmark Evaluator
CoT, ReAct, Reflexion 방법론을 사용한 QA 에이전트 평가
"""

import json
import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 루트를 Python path에 추가
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from graphs.agent.experiments.hotpotqa_agent import (
    create_cot_qa_agent,
    create_react_qa_agent, 
    create_reflexion_qa_agent
)


@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    reasoning_type: str
    question: str
    expected_answer: str
    predicted_answer: str
    is_correct: bool
    response_time: float
    question_type: str
    error: Optional[str] = None


@dataclass
class BenchmarkStats:
    """벤치마크 통계 데이터 클래스"""
    reasoning_type: str
    total_questions: int
    correct_answers: int
    accuracy: float
    avg_response_time: float
    bridge_accuracy: float
    comparison_accuracy: float
    bridge_count: int
    comparison_count: int


class HotpotQAEvaluator:
    """HotpotQA 벤치마크 평가기"""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        model: str = "openai:gpt-4o-mini",
        max_samples: Optional[int] = None,
        parallel_workers: int = 3
    ):
        """
        Args:
            data_path: HotpotQA 데이터 파일 경로
            model: 사용할 언어 모델
            max_samples: 평가할 최대 샘플 수 (None이면 전체)
            parallel_workers: 병렬 처리 워커 수
        """
        self.data_path = data_path or str(project_root / "raw_data" / "hotpot_dev_v1_simplified.json")
        self.model = model
        self.max_samples = max_samples
        self.parallel_workers = parallel_workers
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 결과 저장 디렉토리
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 에이전트 초기화
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """QA 에이전트들 초기화"""
        try:
            self.agents = {
                "cot": create_cot_qa_agent(model=self.model, benchmark="hotpotqa"),
                "react": create_react_qa_agent(model=self.model, benchmark="hotpotqa"),
                "reflexion": create_reflexion_qa_agent(model=self.model, benchmark="hotpotqa")
            }
            self.logger.info("All QA agents initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing agents: {e}")
            raise
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """HotpotQA 데이터셋 로드"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if self.max_samples:
                data = data[:self.max_samples]
            
            self.logger.info(f"Loaded {len(data)} questions from HotpotQA dataset")
            return data
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def normalize_answer(self, answer: str) -> str:
        """답변 정규화 (대소문자, 공백, 구두점 처리)"""
        import re
        import string
        
        # 소문자 변환
        answer = answer.lower()
        
        # 구두점 제거
        answer = re.sub(r'[{}]'.format(re.escape(string.punctuation)), ' ', answer)
        
        # 연속된 공백을 하나로
        answer = re.sub(r'\s+', ' ', answer)
        
        # 앞뒤 공백 제거
        answer = answer.strip()
        
        return answer
    
    def evaluate_answer(self, predicted: str, expected: str) -> bool:
        """답변 정확성 평가"""
        pred_normalized = self.normalize_answer(predicted)
        exp_normalized = self.normalize_answer(expected)
        
        # 정확한 매칭
        if pred_normalized == exp_normalized:
            return True
        
        # 부분 매칭 (예상 답변이 예측에 포함되어 있는지)
        if exp_normalized in pred_normalized:
            return True
        
        # Yes/No 질문 특별 처리
        if exp_normalized in ["yes", "no"]:
            if exp_normalized in pred_normalized:
                return True
        
        return False
    
    def extract_final_answer(self, response: str, question_type: str) -> str:
        """응답에서 최종 답변 추출"""
        import re
        
        # 응답이 문자열이 아닌 경우 처리
        if not isinstance(response, str):
            return str(response)
        
        # 일반적인 답변 패턴들
        answer_patterns = [
            r"(?i)final answer[:\s]*(.+?)(?:\n|$)",
            r"(?i)answer[:\s]*(.+?)(?:\n|$)", 
            r"(?i)the answer is[:\s]*(.+?)(?:\n|$)",
            r"(?i)conclusion[:\s]*(.+?)(?:\n|$)"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response)
            if match:
                answer = match.group(1).strip()
                # 문장 끝 구두점 제거
                answer = re.sub(r'[.!?]+$', '', answer)
                return answer
        
        # Yes/No 질문의 경우
        if question_type == "comparison":
            yes_patterns = r"(?i)\b(yes|true|correct|same)\b"
            no_patterns = r"(?i)\b(no|false|incorrect|different)\b"
            
            if re.search(yes_patterns, response):
                return "yes"
            elif re.search(no_patterns, response):
                return "no"
        
        # 패턴을 찾지 못한 경우 마지막 문장 반환
        sentences = response.split('\n')
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if sentence and not sentence.startswith('**') and len(sentence) > 3:
                return sentence
        
        return response.strip()
    
    def evaluate_single_question(
        self, 
        question_data: Dict[str, Any], 
        reasoning_type: str
    ) -> EvaluationResult:
        """단일 질문 평가"""
        question = question_data["question"]
        expected_answer = question_data["answer"]
        question_type = question_data["type"]
        
        try:
            start_time = time.time()
            
            # 에이전트 실행
            agent = self.agents[reasoning_type]
            result = agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            
            response_time = time.time() - start_time
            
            # 응답에서 답변 추출
            if result and "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    response_content = last_message.content
                else:
                    response_content = str(last_message)
            else:
                response_content = "No response"
            
            predicted_answer = self.extract_final_answer(response_content, question_type)
            
            # 정확성 평가
            is_correct = self.evaluate_answer(predicted_answer, expected_answer)
            
            return EvaluationResult(
                reasoning_type=reasoning_type,
                question=question,
                expected_answer=expected_answer,
                predicted_answer=predicted_answer,
                is_correct=is_correct,
                response_time=response_time,
                question_type=question_type
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating question: {e}")
            return EvaluationResult(
                reasoning_type=reasoning_type,
                question=question,
                expected_answer=expected_answer,
                predicted_answer="ERROR",
                is_correct=False,
                response_time=0.0,
                question_type=question_type,
                error=str(e)
            )
    
    def evaluate_reasoning_type(
        self, 
        dataset: List[Dict[str, Any]], 
        reasoning_type: str
    ) -> Tuple[List[EvaluationResult], BenchmarkStats]:
        """특정 추론 방법론으로 전체 데이터셋 평가"""
        self.logger.info(f"Starting evaluation with {reasoning_type.upper()} reasoning...")
        
        results = []
        
        # 병렬 처리로 평가 실행
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {
                executor.submit(self.evaluate_single_question, question_data, reasoning_type): i
                for i, question_data in enumerate(dataset)
            }
            
            for future in as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Completed {i + 1}/{len(dataset)} questions")
                        
                except Exception as e:
                    self.logger.error(f"Error in question {i}: {e}")
        
        # 통계 계산
        stats = self._calculate_stats(results, reasoning_type)
        
        return results, stats
    
    def _calculate_stats(
        self, 
        results: List[EvaluationResult], 
        reasoning_type: str
    ) -> BenchmarkStats:
        """평가 결과 통계 계산"""
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r.is_correct)
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
        # 응답 시간 평균
        valid_times = [r.response_time for r in results if r.response_time > 0]
        avg_response_time = sum(valid_times) / len(valid_times) if valid_times else 0.0
        
        # 질문 타입별 통계
        bridge_results = [r for r in results if r.question_type == "bridge"]
        comparison_results = [r for r in results if r.question_type == "comparison"]
        
        bridge_accuracy = (
            sum(1 for r in bridge_results if r.is_correct) / len(bridge_results)
            if bridge_results else 0.0
        )
        comparison_accuracy = (
            sum(1 for r in comparison_results if r.is_correct) / len(comparison_results)
            if comparison_results else 0.0
        )
        
        return BenchmarkStats(
            reasoning_type=reasoning_type,
            total_questions=total_questions,
            correct_answers=correct_answers,
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            bridge_accuracy=bridge_accuracy,
            comparison_accuracy=comparison_accuracy,
            bridge_count=len(bridge_results),
            comparison_count=len(comparison_results)
        )
    
    def save_results(
        self, 
        all_results: Dict[str, List[EvaluationResult]], 
        all_stats: Dict[str, BenchmarkStats]
    ):
        """평가 결과 저장"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 상세 결과 저장
        detailed_results = {}
        for reasoning_type, results in all_results.items():
            detailed_results[reasoning_type] = [
                {
                    "question": r.question,
                    "expected_answer": r.expected_answer,
                    "predicted_answer": r.predicted_answer,
                    "is_correct": r.is_correct,
                    "response_time": r.response_time,
                    "question_type": r.question_type,
                    "error": r.error
                }
                for r in results
            ]
        
        detailed_file = self.results_dir / f"hotpotqa_detailed_results_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # 통계 요약 저장
        summary_stats = {}
        for reasoning_type, stats in all_stats.items():
            summary_stats[reasoning_type] = {
                "total_questions": stats.total_questions,
                "correct_answers": stats.correct_answers,
                "accuracy": stats.accuracy,
                "avg_response_time": stats.avg_response_time,
                "bridge_accuracy": stats.bridge_accuracy,
                "comparison_accuracy": stats.comparison_accuracy,
                "bridge_count": stats.bridge_count,
                "comparison_count": stats.comparison_count
            }
        
        summary_file = self.results_dir / f"hotpotqa_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {detailed_file} and {summary_file}")
    
    def run_evaluation(self) -> Dict[str, BenchmarkStats]:
        """전체 평가 실행"""
        self.logger.info("Starting HotpotQA benchmark evaluation...")
        
        # 데이터셋 로드
        dataset = self.load_dataset()
        
        # 각 추론 방법론별 평가
        all_results = {}
        all_stats = {}
        
        for reasoning_type in ["cot", "react", "reflexion"]:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Evaluating {reasoning_type.upper()} reasoning")
            self.logger.info(f"{'='*50}")
            
            results, stats = self.evaluate_reasoning_type(dataset, reasoning_type)
            all_results[reasoning_type] = results
            all_stats[reasoning_type] = stats
            
            # 중간 결과 출력
            self._print_stats(stats)
        
        # 결과 저장
        self.save_results(all_results, all_stats)
        
        # 최종 비교 출력
        self._print_comparison(all_stats)
        
        return all_stats
    
    def _print_stats(self, stats: BenchmarkStats):
        """통계 출력"""
        print(f"\n{stats.reasoning_type.upper()} Results:")
        print(f"  Total Questions: {stats.total_questions}")
        print(f"  Correct Answers: {stats.correct_answers}")
        print(f"  Overall Accuracy: {stats.accuracy:.3f}")
        print(f"  Bridge Questions Accuracy: {stats.bridge_accuracy:.3f} ({stats.bridge_count} questions)")
        print(f"  Comparison Questions Accuracy: {stats.comparison_accuracy:.3f} ({stats.comparison_count} questions)")
        print(f"  Average Response Time: {stats.avg_response_time:.2f}s")
    
    def _print_comparison(self, all_stats: Dict[str, BenchmarkStats]):
        """모든 방법론 비교 출력"""
        print(f"\n{'='*60}")
        print("FINAL COMPARISON")
        print(f"{'='*60}")
        
        # 표 형태로 출력
        print(f"{'Method':<12} {'Accuracy':<10} {'Bridge':<10} {'Comparison':<12} {'Avg Time':<10}")
        print("-" * 60)
        
        for reasoning_type, stats in all_stats.items():
            print(f"{reasoning_type.upper():<12} {stats.accuracy:<10.3f} {stats.bridge_accuracy:<10.3f} "
                  f"{stats.comparison_accuracy:<12.3f} {stats.avg_response_time:<10.2f}s")
        
        # 최고 성능 방법론 찾기
        best_overall = max(all_stats.items(), key=lambda x: x[1].accuracy)
        best_bridge = max(all_stats.items(), key=lambda x: x[1].bridge_accuracy)
        best_comparison = max(all_stats.items(), key=lambda x: x[1].comparison_accuracy)
        fastest = min(all_stats.items(), key=lambda x: x[1].avg_response_time)
        
        print(f"\nBest Performance:")
        print(f"  Overall Accuracy: {best_overall[0].upper()} ({best_overall[1].accuracy:.3f})")
        print(f"  Bridge Questions: {best_bridge[0].upper()} ({best_bridge[1].bridge_accuracy:.3f})")
        print(f"  Comparison Questions: {best_comparison[0].upper()} ({best_comparison[1].comparison_accuracy:.3f})")
        print(f"  Fastest Response: {fastest[0].upper()} ({fastest[1].avg_response_time:.2f}s)")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HotpotQA Benchmark Evaluation")
    parser.add_argument("--data-path", type=str, help="Path to HotpotQA data file")
    parser.add_argument("--model", type=str, default="openai:gpt-4o-mini", help="Model to use")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    evaluator = HotpotQAEvaluator(
        data_path=args.data_path,
        model=args.model,
        max_samples=args.max_samples,
        parallel_workers=args.workers
    )
    
    try:
        evaluator.run_evaluation()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()