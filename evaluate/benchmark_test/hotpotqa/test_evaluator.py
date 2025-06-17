#!/usr/bin/env python3
"""
HotpotQA 평가기 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from evaluate.benchmark_test.hotpotqa.evaluator import HotpotQAEvaluator


def test_evaluator_creation():
    """평가기 생성 테스트"""
    print("=== Testing Evaluator Creation ===")
    
    try:
        evaluator = HotpotQAEvaluator(
            max_samples=5,  # 5개 샘플만 테스트
            parallel_workers=1  # 단일 워커로 테스트
        )
        print("✓ Evaluator created successfully")
        return evaluator
    except Exception as e:
        print(f"❌ Error creating evaluator: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_dataset_loading(evaluator):
    """데이터셋 로딩 테스트"""
    print("\n=== Testing Dataset Loading ===")
    
    try:
        dataset = evaluator.load_dataset()
        print(f"✓ Dataset loaded successfully: {len(dataset)} questions")
        
        # 첫 번째 질문 출력
        if dataset:
            first_q = dataset[0]
            print(f"  Sample question: {first_q['question']}")
            print(f"  Answer: {first_q['answer']}")
            print(f"  Type: {first_q['type']}")
        
        return dataset
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_single_evaluation(evaluator, dataset):
    """단일 질문 평가 테스트"""
    print("\n=== Testing Single Question Evaluation ===")
    
    if not dataset:
        print("❌ No dataset available for testing")
        return False
    
    try:
        # 첫 번째 질문으로 테스트
        question_data = dataset[0]
        print(f"Testing question: {question_data['question']}")
        
        # CoT 방법론으로 테스트
        result = evaluator.evaluate_single_question(question_data, "cot")
        
        print(f"✓ Single evaluation completed")
        print(f"  Expected: {result.expected_answer}")
        print(f"  Predicted: {result.predicted_answer}")
        print(f"  Correct: {result.is_correct}")
        print(f"  Time: {result.response_time:.2f}s")
        
        return True
    except Exception as e:
        print(f"❌ Error in single evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_answer_extraction(evaluator):
    """답변 추출 테스트"""
    print("\n=== Testing Answer Extraction ===")
    
    test_responses = [
        ("Final Answer: Paris", "bridge"),
        ("The answer is yes", "comparison"),
        ("Based on my analysis, the answer is New York", "bridge"),
        ("No, they are not the same", "comparison")
    ]
    
    try:
        for response, q_type in test_responses:
            extracted = evaluator.extract_final_answer(response, q_type)
            print(f"  '{response}' -> '{extracted}'")
        
        print("✓ Answer extraction test completed")
        return True
    except Exception as e:
        print(f"❌ Error in answer extraction: {e}")
        return False


def test_mini_evaluation(evaluator):
    """소규모 평가 테스트 (3개 질문만)"""
    print("\n=== Testing Mini Evaluation ===")
    
    try:
        # 매우 작은 샘플로 전체 파이프라인 테스트
        evaluator.max_samples = 3
        dataset = evaluator.load_dataset()
        
        if not dataset:
            print("❌ No dataset for mini evaluation")
            return False
        
        # CoT 방법론으로만 테스트
        results, stats = evaluator.evaluate_reasoning_type(dataset, "cot")
        
        print(f"✓ Mini evaluation completed")
        print(f"  Evaluated {len(results)} questions")
        print(f"  Accuracy: {stats.accuracy:.3f}")
        print(f"  Average time: {stats.avg_response_time:.2f}s")
        
        return True
    except Exception as e:
        print(f"❌ Error in mini evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""
    print("🧪 Starting HotpotQA Evaluator Test Suite")
    print("=" * 60)
    
    # 테스트 실행
    success_count = 0
    total_tests = 5
    
    # 1. 평가기 생성 테스트
    evaluator = test_evaluator_creation()
    if evaluator:
        success_count += 1
    else:
        print("❌ Cannot proceed without evaluator")
        return
    
    # 2. 데이터셋 로딩 테스트
    dataset = test_dataset_loading(evaluator)
    if dataset:
        success_count += 1
    
    # 3. 답변 추출 테스트
    if test_answer_extraction(evaluator):
        success_count += 1
    
    # 4. 단일 평가 테스트
    if test_single_evaluation(evaluator, dataset):
        success_count += 1
    
    # 5. 소규모 평가 테스트
    if test_mini_evaluation(evaluator):
        success_count += 1
    
    # 결과 출력
    print("\n" + "=" * 60)
    print(f"Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! The evaluator is ready to use.")
        print("\nTo run a full evaluation:")
        print("python evaluate/benchmark_test/hotpotqa/evaluator.py --max-samples 50")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("\n👋 Test suite completed.")


if __name__ == "__main__":
    main()
