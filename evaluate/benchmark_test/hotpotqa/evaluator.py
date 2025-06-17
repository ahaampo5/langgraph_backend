#!/usr/bin/env python3
"""
HotpotQA 평가기 - 완전 기능 버전
CoT, ReAct, Reflexion 등 모든 추론 방법론을 지원하는 통합 평가기
중간 저장/재개 기능과 병렬 처리 옵션 포함
"""

import json
import os
import sys
import time
import logging
import importlib.util
import random
import re
import string
import argparse
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from langchain_core.messages import AIMessage

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
    f1_score: float = 0.0
    total_calls: int = 0
    total_completion_tokens: int = 0
    total_prompt_tokens: int = 0
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
    avg_f1_score: float = 0.0
    bridge_avg_f1_score: float = 0.0
    comparison_avg_f1_score: float = 0.0
    total_calls: int = 0
    avg_calls_per_question: float = 0.0
    total_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    total_tokens: int = 0
    avg_completion_tokens_per_question: float = 0.0
    avg_prompt_tokens_per_question: float = 0.0
    avg_total_tokens_per_question: float = 0.0


class HotpotQAEvaluator:
    """HotpotQA 벤치마크 평가기 - 완전 기능 통합 버전"""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        model: str = "openai:gpt-4o-mini",
        max_samples: Optional[int] = None,
        parallel_workers: int = 1,  # 안정성을 위해 기본값 1, 필요시 증가 가능
        timestamp: Optional[str] = None,
        seed: int = 42
    ):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.data_path = data_path or str(self.project_root / "raw_data" / "hotpot_dev_fullwiki_v1.json")
        self.model = model
        self.max_samples = max_samples
        self.parallel_workers = parallel_workers
        self.seed = seed
        
        # timestamp 설정 (지정되지 않으면 현재 시간 사용)
        self.timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
        
        # seed 고정
        random.seed(self.seed)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 결과 저장 디렉토리
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 중간 결과 저장을 위한 파일 경로들
        self._setup_result_files()
        
        # 에이전트 모듈 로드
        self._load_agent_module()
        self._initialize_agents()
    
    def _load_agent_module(self):
        """에이전트 모듈 직접 로드"""
        agent_path = self.project_root / "graphs" / "agent" / "experiments" / "qa_agent.py"
        spec = importlib.util.spec_from_file_location("qa_agent", agent_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load agent module from {agent_path}")
        
        self.agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.agent_module)
        self.logger.info("Agent module loaded successfully")
    
    def _initialize_agents(self):
        """QA 에이전트들 초기화"""
        try:
            self.agents = {
                "io": self.agent_module.create_io_qa_agent(model=self.model, benchmark="hotpotqa"),
                "cot": self.agent_module.create_cot_qa_agent(model=self.model, benchmark="hotpotqa"),
                "react": self.agent_module.create_react_qa_agent(model=self.model, benchmark="hotpotqa"),
                "reflexion": self.agent_module.create_reflexion_qa_agent(model=self.model, benchmark="hotpotqa")
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
            
            # seed를 사용하여 동일한 샘플링 보장
            if self.max_samples and len(data) > self.max_samples:
                random.seed(self.seed)  # 샘플링 전에 다시 seed 설정
                data = random.sample(data, self.max_samples)
                # 샘플링 후 순서를 일정하게 유지
                data.sort(key=lambda x: x.get('_id', ''))
            
            self.logger.info(f"Loaded {len(data)} questions from HotpotQA dataset")
            return data
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def normalize_answer(self, answer: str) -> str:
        """답변 정규화 (대소문자, 공백, 구두점 처리)"""
        answer = re.sub(r"\b(a|an|the)\b", " ", answer)
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
        
        if pred_normalized == exp_normalized:
            return True
        
        if exp_normalized in pred_normalized:
            return True
        
        if exp_normalized in ["yes", "no"]:
            if exp_normalized in pred_normalized:
                return True
        
        return False

    def evaluate_f1_score(self, predicted: str, expected: str) -> Tuple[float, str]:
        """F1 점수 계산"""
        prediction_tokens = self.normalize_answer(predicted).split()
        ground_truth_tokens = self.normalize_answer(expected).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0, predicted
        precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0.0
        recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1, predicted
    
    def extract_final_answer(self, response: str, question_type: str) -> str:
        """응답에서 최종 답변 추출"""
        import re
        
        if not isinstance(response, str):
            return str(response)
        
        # 답변 패턴 찾기
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
                answer = re.sub(r'[.!?]+$', '', answer)
                return answer
        
        # Yes/No 질문 처리
        if question_type == "comparison":
            yes_patterns = r"(?i)\b(yes|true|correct|same)\b"
            no_patterns = r"(?i)\b(no|false|incorrect|different)\b"
            
            if re.search(yes_patterns, response):
                return "yes"
            elif re.search(no_patterns, response):
                return "no"
        
        # 마지막 문장 반환
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
        paragraphs = [item[1] for item in question_data["context"] if isinstance(item[1], list)]
        context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
        inputs = f"Context: {context_str}\n\nQuestion: {question}\n\nAnswer:"

        total_calls = 0
        total_completion_tokens = 0
        total_prompt_tokens = 0
        try:
            start_time = time.time()
            
            # 에이전트 실행
            agent = self.agents[reasoning_type]
            result = agent.invoke({
                "messages": [{"role": "user", "content": inputs}]
            })

            
            for m in result.get("messages", []):
                if isinstance(m, AIMessage):
                    total_calls += 1
                    response_metadata = m.response_metadata
                    token_usage = response_metadata.get("token_usage", {})
                    completion_tokens = token_usage.get("completion_tokens", 0)
                    prompt_tokens = token_usage.get("prompt_tokens", 0)
                    total_completion_tokens += completion_tokens
                    total_prompt_tokens += prompt_tokens

            response_time = time.time() - start_time
            
            # 응답 추출
            if result and "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    response_content = last_message.content
                else:
                    response_content = str(last_message)
            else:
                response_content = "No response"
            
            predicted_answer = self.extract_final_answer(response_content, question_type)
            is_correct = self.evaluate_answer(predicted_answer, expected_answer)
            f1_score, _ = self.evaluate_f1_score(predicted_answer, expected_answer)
            
            return EvaluationResult(
                reasoning_type=reasoning_type,
                question=question,
                expected_answer=expected_answer,
                predicted_answer=predicted_answer,
                is_correct=is_correct,
                response_time=response_time,
                question_type=question_type,
                f1_score=f1_score,
                total_calls=total_calls,
                total_completion_tokens=total_completion_tokens,
                total_prompt_tokens=total_prompt_tokens
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
                f1_score=0.0,
                total_calls=total_calls,
                total_completion_tokens=total_completion_tokens,
                total_prompt_tokens=total_prompt_tokens,
                error=str(e)
            )
    
    def evaluate_reasoning_type(
        self, 
        dataset: List[Dict[str, Any]], 
        reasoning_type: str
    ) -> Tuple[List[EvaluationResult], BenchmarkStats]:
        """특정 추론 방법론 평가"""
        self.logger.info(f"Starting evaluation with {reasoning_type.upper()} reasoning...")
        
        results = []
        
        # 병렬 처리 사용 여부 결정
        if self.parallel_workers > 1:
            results = self._evaluate_parallel(dataset, reasoning_type)
        else:
            results = self._evaluate_sequential(dataset, reasoning_type)
        
        # 통계 계산
        stats = self._calculate_stats(results, reasoning_type)
        return results, stats
    
    def _evaluate_sequential(
        self, 
        dataset: List[Dict[str, Any]], 
        reasoning_type: str
    ) -> List[EvaluationResult]:
        """순차 평가 (안전하고 재개 가능)"""
        results = []
        
        for i, question_data in enumerate(dataset):
            # 이미 완료된 질문인지 확인
            if self._is_question_completed(reasoning_type, i):
                result = self._load_completed_result(reasoning_type, i)
                results.append(result)
                self.logger.info(f"Loaded cached result for question {i + 1}/{len(dataset)}")
                continue
            
            # 새로운 질문 평가
            result = self.evaluate_single_question(question_data, reasoning_type)
            results.append(result)
            
            # 중간 저장
            self._save_progress(reasoning_type, i, result)
            
            if (i + 1) % 5 == 0:
                self.logger.info(f"Completed {i + 1}/{len(dataset)} questions")
                # 중간 통계 저장
                stats = self._calculate_stats(results, reasoning_type)
                self._save_intermediate_results(reasoning_type, results, stats)
        
        return results
    
    def _evaluate_parallel(
        self, 
        dataset: List[Dict[str, Any]], 
        reasoning_type: str
    ) -> List[EvaluationResult]:
        """병렬 평가 (빠르지만 재개 어려움)"""
        results: List[Optional[EvaluationResult]] = [None] * len(dataset)
        completed_count = 0
        
        # 이미 완료된 작업들 먼저 로드
        for i, question_data in enumerate(dataset):
            if self._is_question_completed(reasoning_type, i):
                results[i] = self._load_completed_result(reasoning_type, i)
                completed_count += 1
        
        # 남은 작업들을 병렬로 처리
        remaining_indices = [i for i in range(len(dataset)) if results[i] is None]
        
        if remaining_indices:
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                future_to_index = {
                    executor.submit(self.evaluate_single_question, dataset[i], reasoning_type): i
                    for i in remaining_indices
                }
                
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                        completed_count += 1
                        
                        # 중간 저장
                        self._save_progress(reasoning_type, index, result)
                        
                        if completed_count % 5 == 0:
                            self.logger.info(f"Completed {completed_count}/{len(dataset)} questions")
                    except Exception as e:
                        self.logger.error(f"Error in parallel evaluation for question {index}: {e}")
                        results[index] = EvaluationResult(
                            reasoning_type=reasoning_type,
                            question=dataset[index]["question"],
                            expected_answer=dataset[index]["answer"],
                            predicted_answer="ERROR",
                            is_correct=False,
                            response_time=0.0,
                            question_type=dataset[index]["type"],
                            f1_score=0.0,
                            error=str(e)
                        )
        
        return [r for r in results if r is not None]
    
    def _calculate_stats(
        self, 
        results: List[EvaluationResult], 
        reasoning_type: str
    ) -> BenchmarkStats:
        """통계 계산"""
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r.is_correct)
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
        valid_times = [r.response_time for r in results if r.response_time > 0]
        avg_response_time = sum(valid_times) / len(valid_times) if valid_times else 0.0
        
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
        
        # F1 점수 통계 계산
        avg_f1_score = sum(r.f1_score for r in results) / total_questions if total_questions > 0 else 0.0
        bridge_avg_f1_score = (
            sum(r.f1_score for r in bridge_results) / len(bridge_results)
            if bridge_results else 0.0
        )
        comparison_avg_f1_score = (
            sum(r.f1_score for r in comparison_results) / len(comparison_results)
            if comparison_results else 0.0
        )
        
        # 토큰 관련 통계 계산
        total_calls = sum(r.total_calls for r in results)
        total_completion_tokens = sum(r.total_completion_tokens for r in results)
        total_prompt_tokens = sum(r.total_prompt_tokens for r in results)
        total_tokens = total_completion_tokens + total_prompt_tokens
        
        avg_calls_per_question = total_calls / total_questions if total_questions > 0 else 0.0
        avg_completion_tokens_per_question = total_completion_tokens / total_questions if total_questions > 0 else 0.0
        avg_prompt_tokens_per_question = total_prompt_tokens / total_questions if total_questions > 0 else 0.0
        avg_total_tokens_per_question = total_tokens / total_questions if total_questions > 0 else 0.0
        
        return BenchmarkStats(
            reasoning_type=reasoning_type,
            total_questions=total_questions,
            correct_answers=correct_answers,
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            bridge_accuracy=bridge_accuracy,
            comparison_accuracy=comparison_accuracy,
            bridge_count=len(bridge_results),
            comparison_count=len(comparison_results),
            avg_f1_score=avg_f1_score,
            bridge_avg_f1_score=bridge_avg_f1_score,
            comparison_avg_f1_score=comparison_avg_f1_score,
            total_calls=total_calls,
            avg_calls_per_question=avg_calls_per_question,
            total_completion_tokens=total_completion_tokens,
            total_prompt_tokens=total_prompt_tokens,
            total_tokens=total_tokens,
            avg_completion_tokens_per_question=avg_completion_tokens_per_question,
            avg_prompt_tokens_per_question=avg_prompt_tokens_per_question,
            avg_total_tokens_per_question=avg_total_tokens_per_question
        )
    
    def _setup_result_files(self):
        """중간 결과 저장을 위한 파일 경로 설정"""
        self.detailed_file = self.results_dir / f"hotpotqa_detailed_results_{self.timestamp}.json"
        self.summary_file = self.results_dir / f"hotpotqa_summary_{self.timestamp}.json"
        self.progress_file = self.results_dir / f"hotpotqa_progress_{self.timestamp}.json"
        
        # 진행 상황 파일이 존재하면 로드
        self.completed_questions = {}
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.completed_questions = json.load(f)
                self.logger.info(f"Loaded progress from {self.progress_file}")
                self.logger.info(f"Previously completed: {sum(len(v) for v in self.completed_questions.values())} questions")
            except Exception as e:
                self.logger.warning(f"Could not load progress file: {e}")
                self.completed_questions = {}
    
    def _save_progress(self, reasoning_type: str, question_idx: int, result: EvaluationResult):
        """진행 상황을 중간 저장"""
        if reasoning_type not in self.completed_questions:
            self.completed_questions[reasoning_type] = {}
        
        self.completed_questions[reasoning_type][str(question_idx)] = {
            "question": result.question,
            "expected_answer": result.expected_answer,
            "predicted_answer": result.predicted_answer,
            "is_correct": result.is_correct,
            "response_time": result.response_time,
            "question_type": result.question_type,
            "f1_score": result.f1_score,
            "total_calls": result.total_calls,
            "total_completion_tokens": result.total_completion_tokens,
            "total_prompt_tokens": result.total_prompt_tokens,
            "error": result.error
        }
        
        # 진행 상황 저장
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.completed_questions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"Could not save progress: {e}")
    
    def _is_question_completed(self, reasoning_type: str, question_idx: int) -> bool:
        """특정 질문이 이미 완료되었는지 확인"""
        return (reasoning_type in self.completed_questions and 
                str(question_idx) in self.completed_questions[reasoning_type])
    
    def _load_completed_result(self, reasoning_type: str, question_idx: int) -> EvaluationResult:
        """완료된 결과 로드"""
        data = self.completed_questions[reasoning_type][str(question_idx)]
        return EvaluationResult(
            reasoning_type=reasoning_type,
            question=data["question"],
            expected_answer=data["expected_answer"],
            predicted_answer=data["predicted_answer"],
            is_correct=data["is_correct"],
            response_time=data["response_time"],
            question_type=data["question_type"],
            f1_score=data.get("f1_score", 0.0),
            total_calls=data.get("total_calls", 0),
            total_completion_tokens=data.get("total_completion_tokens", 0),
            total_prompt_tokens=data.get("total_prompt_tokens", 0),
            error=data.get("error")
        )

    def save_results(
        self, 
        all_results: Dict[str, List[EvaluationResult]], 
        all_stats: Dict[str, BenchmarkStats]
    ):
        """결과 저장"""
        # self.timestamp 사용 (초기화에서 설정됨)
        
        # 상세 결과
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
                    "f1_score": r.f1_score,
                    "total_calls": r.total_calls,
                    "total_completion_tokens": r.total_completion_tokens,
                    "total_prompt_tokens": r.total_prompt_tokens,
                    "error": r.error
                }
                for r in results
            ]
        
        # 이미 설정된 파일 경로 사용
        with open(self.detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # 통계 요약
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
                "comparison_count": stats.comparison_count,
                "avg_f1_score": stats.avg_f1_score,
                "bridge_avg_f1_score": stats.bridge_avg_f1_score,
                "comparison_avg_f1_score": stats.comparison_avg_f1_score,
                "total_calls": stats.total_calls,
                "avg_calls_per_question": stats.avg_calls_per_question,
                "total_completion_tokens": stats.total_completion_tokens,
                "total_prompt_tokens": stats.total_prompt_tokens,
                "total_tokens": stats.total_tokens,
                "avg_completion_tokens_per_question": stats.avg_completion_tokens_per_question,
                "avg_prompt_tokens_per_question": stats.avg_prompt_tokens_per_question,
                "avg_total_tokens_per_question": stats.avg_total_tokens_per_question
            }
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {self.detailed_file} and {self.summary_file}")
    
    def _save_intermediate_results(
        self, 
        reasoning_type: str, 
        results: List[EvaluationResult], 
        stats: BenchmarkStats
    ):
        """중간 결과 저장"""
        try:
            # 현재까지의 상세 결과
            detailed_results = {
                reasoning_type: [
                    {
                        "question": r.question,
                        "expected_answer": r.expected_answer,
                        "predicted_answer": r.predicted_answer,
                        "is_correct": r.is_correct,
                        "response_time": r.response_time,
                        "question_type": r.question_type,
                        "f1_score": r.f1_score,
                        "total_calls": r.total_calls,
                        "total_completion_tokens": r.total_completion_tokens,
                        "total_prompt_tokens": r.total_prompt_tokens,
                        "error": r.error
                    }
                    for r in results
                ]
            }
            
            # 기존 파일이 있으면 로드하여 합치기
            if self.detailed_file.exists():
                with open(self.detailed_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                existing_results.update(detailed_results)
                detailed_results = existing_results
            
            # 상세 결과 저장
            with open(self.detailed_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
            # 통계 요약 저장
            summary_stats = {
                reasoning_type: {
                    "total_questions": stats.total_questions,
                    "correct_answers": stats.correct_answers,
                    "accuracy": stats.accuracy,
                    "avg_response_time": stats.avg_response_time,
                    "bridge_accuracy": stats.bridge_accuracy,
                    "comparison_accuracy": stats.comparison_accuracy,
                    "bridge_count": stats.bridge_count,
                    "comparison_count": stats.comparison_count,
                    "avg_f1_score": stats.avg_f1_score,
                    "bridge_avg_f1_score": stats.bridge_avg_f1_score,
                    "comparison_avg_f1_score": stats.comparison_avg_f1_score,
                    "total_calls": stats.total_calls,
                    "avg_calls_per_question": stats.avg_calls_per_question,
                    "total_completion_tokens": stats.total_completion_tokens,
                    "total_prompt_tokens": stats.total_prompt_tokens,
                    "total_tokens": stats.total_tokens,
                    "avg_completion_tokens_per_question": stats.avg_completion_tokens_per_question,
                    "avg_prompt_tokens_per_question": stats.avg_prompt_tokens_per_question,
                    "avg_total_tokens_per_question": stats.avg_total_tokens_per_question
                }
            }
            
            # 기존 요약이 있으면 로드하여 합치기
            if self.summary_file.exists():
                with open(self.summary_file, 'r', encoding='utf-8') as f:
                    existing_summary = json.load(f)
                existing_summary.update(summary_stats)
                summary_stats = existing_summary
            
            # 요약 저장
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_stats, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.warning(f"Could not save intermediate results: {e}")

    def run_evaluation(self) -> Dict[str, BenchmarkStats]:
        """전체 평가 실행"""
        print("🚀 Starting HotpotQA benchmark evaluation...")
        print(f"📁 Results will be saved with timestamp: {self.timestamp}")
        
        # 이전 진행 상황이 있는 경우 출력
        if self.completed_questions:
            total_completed = sum(len(v) for v in self.completed_questions.values())
            print(f"📊 Found {total_completed} previously completed questions")
            for reasoning_type, questions in self.completed_questions.items():
                print(f"   - {reasoning_type}: {len(questions)} questions")
        
        dataset = self.load_dataset()
        all_results = {}
        all_stats = {}
        
        for reasoning_type in ["io", "cot"]: #, "react", "reflexion"]:
            print(f"\n{'='*50}")
            print(f"Evaluating {reasoning_type.upper()} reasoning")
            print(f"{'='*50}")
            
            results, stats = self.evaluate_reasoning_type(dataset, reasoning_type)
            all_results[reasoning_type] = results
            all_stats[reasoning_type] = stats
            
            self._print_stats(stats)
            # 각 reasoning type 완료 후 중간 저장
            self._save_intermediate_results(reasoning_type, results, stats)
        
        self.save_results(all_results, all_stats)
        self._print_comparison(all_stats)
        
        return all_stats
    
    def _print_stats(self, stats: BenchmarkStats):
        """통계 출력"""
        print(f"\n{stats.reasoning_type.upper()} Results:")
        print(f"  Total Questions: {stats.total_questions}")
        print(f"  Correct Answers: {stats.correct_answers}")
        print(f"  Overall Accuracy: {stats.accuracy:.3f}")
        print(f"  Average F1 Score: {stats.avg_f1_score:.3f}")
        print(f"  Bridge Questions Accuracy: {stats.bridge_accuracy:.3f} ({stats.bridge_count} questions)")
        print(f"  Bridge Questions F1 Score: {stats.bridge_avg_f1_score:.3f}")
        print(f"  Comparison Questions Accuracy: {stats.comparison_accuracy:.3f} ({stats.comparison_count} questions)")
        print(f"  Comparison Questions F1 Score: {stats.comparison_avg_f1_score:.3f}")
        print(f"  Average Response Time: {stats.avg_response_time:.2f}s")
        print(f"  Token Usage:")
        print(f"    Total API Calls: {stats.total_calls}")
        print(f"    Avg Calls per Question: {stats.avg_calls_per_question:.1f}")
        print(f"    Total Tokens: {stats.total_tokens:,} (Prompt: {stats.total_prompt_tokens:,}, Completion: {stats.total_completion_tokens:,})")
        print(f"    Avg Tokens per Question: {stats.avg_total_tokens_per_question:.0f} (Prompt: {stats.avg_prompt_tokens_per_question:.0f}, Completion: {stats.avg_completion_tokens_per_question:.0f})")
    
    def _print_comparison(self, all_stats: Dict[str, BenchmarkStats]):
        """모든 방법론 비교 출력"""
        print(f"\n{'='*60}")
        print("FINAL COMPARISON")
        print(f"{'='*60}")
        
        # 표 형태로 출력 - 성능 비교
        print(f"{'Method':<12} {'Accuracy':<10} {'F1 Score':<10} {'Bridge Acc':<12} {'Comparison Acc':<15} {'Avg Time':<10}")
        print("-" * 80)
        
        for reasoning_type, stats in all_stats.items():
            print(f"{reasoning_type.upper():<12} {stats.accuracy:<10.3f} {stats.avg_f1_score:<10.3f} "
                  f"{stats.bridge_accuracy:<12.3f} {stats.comparison_accuracy:<15.3f} {stats.avg_response_time:<10.2f}s")
        
        # F1 점수 상세 비교
        print(f"\n{'='*80}")
        print("F1 SCORE DETAILED COMPARISON")
        print(f"{'='*80}")
        print(f"{'Method':<12} {'Overall F1':<12} {'Bridge F1':<12} {'Comparison F1':<15}")
        print("-" * 60)
        
        for reasoning_type, stats in all_stats.items():
            print(f"{reasoning_type.upper():<12} {stats.avg_f1_score:<12.3f} "
                  f"{stats.bridge_avg_f1_score:<12.3f} {stats.comparison_avg_f1_score:<15.3f}")
        
        # 토큰 사용량 비교
        print(f"\n{'='*80}")
        print("TOKEN USAGE COMPARISON")
        print(f"{'='*80}")
        print(f"{'Method':<12} {'Avg Calls':<10} {'Avg Total':<12} {'Avg Prompt':<12} {'Avg Completion':<15}")
        print("-" * 80)
        
        for reasoning_type, stats in all_stats.items():
            print(f"{reasoning_type.upper():<12} {stats.avg_calls_per_question:<10.1f} "
                  f"{stats.avg_total_tokens_per_question:<12.0f} {stats.avg_prompt_tokens_per_question:<12.0f} "
                  f"{stats.avg_completion_tokens_per_question:<15.0f}")
        
        # 최고 성능 방법론 찾기
        if all_stats:
            best_overall = max(all_stats.items(), key=lambda x: x[1].accuracy)
            best_f1 = max(all_stats.items(), key=lambda x: x[1].avg_f1_score)
            best_bridge = max(all_stats.items(), key=lambda x: x[1].bridge_accuracy)
            best_comparison = max(all_stats.items(), key=lambda x: x[1].comparison_accuracy)
            fastest = min(all_stats.items(), key=lambda x: x[1].avg_response_time)
            most_efficient = min(all_stats.items(), key=lambda x: x[1].avg_total_tokens_per_question)
            
            print(f"\nBest Performance:")
            print(f"  Overall Accuracy: {best_overall[0].upper()} ({best_overall[1].accuracy:.3f})")
            print(f"  Overall F1 Score: {best_f1[0].upper()} ({best_f1[1].avg_f1_score:.3f})")
            print(f"  Bridge Questions: {best_bridge[0].upper()} ({best_bridge[1].bridge_accuracy:.3f})")
            print(f"  Comparison Questions: {best_comparison[0].upper()} ({best_comparison[1].comparison_accuracy:.3f})")
            print(f"  Fastest Response: {fastest[0].upper()} ({fastest[1].avg_response_time:.2f}s)")
            print(f"  Most Token Efficient: {most_efficient[0].upper()} ({most_efficient[1].avg_total_tokens_per_question:.0f} tokens/question)")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple HotpotQA Benchmark Evaluation")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum number of samples")
    parser.add_argument("--model", type=str, default="openai:gpt-4o-mini", help="Model to use")
    parser.add_argument("--timestamp", type=str, help="Timestamp for continuing previous evaluation (format: YYYYMMDD_HHMMSS)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    
    args = parser.parse_args()
    
    print(f"🧪 HotpotQA Evaluation with {args.max_samples} questions")
    if args.timestamp:
        print(f"📁 Using timestamp: {args.timestamp} (continuing previous evaluation)")
    print(f"🎲 Using seed: {args.seed}")
    print("=" * 60)
    
    evaluator = HotpotQAEvaluator(
        max_samples=args.max_samples,
        model=args.model,
        timestamp=args.timestamp,
        seed=args.seed
    )
    
    try:
        evaluator.run_evaluation()
        print("\n🎉 Evaluation completed successfully!")
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        print("💾 Progress has been saved. You can continue with the same timestamp.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("💾 Progress has been saved. You can continue with the same timestamp.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
