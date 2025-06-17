#!/usr/bin/env python3
"""
HotpotQA 벤치마크 실행 스크립트
다양한 시나리오로 평가를 실행할 수 있는 편의 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from evaluate.benchmark_test.hotpotqa.evaluator import HotpotQAEvaluator


def run_quick_test():
    """빠른 테스트 (10개 질문)"""
    print("🚀 Running Quick Test (10 questions)")
    print("=" * 50)
    
    evaluator = HotpotQAEvaluator(
        max_samples=10,
        parallel_workers=1
    )
    
    return evaluator.run_evaluation()


def run_small_benchmark():
    """소규모 벤치마크 (50개 질문)"""
    print("🚀 Running Small Benchmark (50 questions)")
    print("=" * 50)
    
    evaluator = HotpotQAEvaluator(
        max_samples=50,
        parallel_workers=2
    )
    
    return evaluator.run_evaluation()


def run_medium_benchmark():
    """중간 규모 벤치마크 (200개 질문)"""
    print("🚀 Running Medium Benchmark (200 questions)")
    print("=" * 50)
    
    evaluator = HotpotQAEvaluator(
        max_samples=200,
        parallel_workers=3
    )
    
    return evaluator.run_evaluation()


def run_full_benchmark():
    """전체 벤치마크 (모든 질문)"""
    print("🚀 Running Full Benchmark (all questions)")
    print("=" * 50)
    print("⚠️  This may take several hours to complete!")
    
    confirm = input("Are you sure you want to run the full benchmark? (y/N): ")
    if confirm.lower() != 'y':
        print("Full benchmark cancelled.")
        return None
    
    evaluator = HotpotQAEvaluator(
        max_samples=None,  # 전체 데이터셋
        parallel_workers=3
    )
    
    return evaluator.run_evaluation()


def run_single_method_test(method: str, num_questions: int = 20):
    """단일 방법론 테스트"""
    print(f"🚀 Running {method.upper()} Method Test ({num_questions} questions)")
    print("=" * 50)
    
    evaluator = HotpotQAEvaluator(
        max_samples=num_questions,
        parallel_workers=1
    )
    
    # 데이터셋 로드
    dataset = evaluator.load_dataset()
    
    # 단일 방법론 평가
    results, stats = evaluator.evaluate_reasoning_type(dataset, method)
    
    # 결과 출력
    evaluator._print_stats(stats)
    
    return {method: stats}


def compare_methods_quick():
    """방법론 간 빠른 비교 (각각 10개 질문)"""
    print("🚀 Quick Methods Comparison (10 questions each)")
    print("=" * 50)
    
    results = {}
    
    for method in ["cot", "react", "reflexion"]:
        print(f"\nTesting {method.upper()}...")
        method_result = run_single_method_test(method, 10)
        results.update(method_result)
    
    # 비교 결과 출력
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    print(f"{'Method':<12} {'Accuracy':<10} {'Avg Time':<10}")
    print("-" * 35)
    
    for method, stats in results.items():
        print(f"{method.upper():<12} {stats.accuracy:<10.3f} {stats.avg_response_time:<10.2f}s")
    
    return results


def interactive_menu():
    """대화형 메뉴"""
    while True:
        print("\n🧪 HotpotQA Benchmark Evaluation")
        print("=" * 40)
        print("1. Quick Test (10 questions)")
        print("2. Small Benchmark (50 questions)")
        print("3. Medium Benchmark (200 questions)")  
        print("4. Full Benchmark (all questions)")
        print("5. Quick Methods Comparison")
        print("6. Test Single Method")
        print("7. Exit")
        
        try:
            choice = input("\nSelect an option (1-7): ").strip()
            
            if choice == "1":
                run_quick_test()
            elif choice == "2":
                run_small_benchmark()
            elif choice == "3":
                run_medium_benchmark()
            elif choice == "4":
                run_full_benchmark()
            elif choice == "5":
                compare_methods_quick()
            elif choice == "6":
                print("\nAvailable methods: cot, react, reflexion")
                method = input("Enter method name: ").strip().lower()
                if method in ["cot", "react", "reflexion"]:
                    num_q = input("Number of questions (default 20): ").strip()
                    num_questions = int(num_q) if num_q.isdigit() else 20
                    run_single_method_test(method, num_questions)
                else:
                    print("Invalid method name!")
            elif choice == "7":
                print("Goodbye!")
                break
            else:
                print("Invalid option. Please choose 1-7.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HotpotQA Benchmark Runner")
    parser.add_argument("--mode", choices=["quick", "small", "medium", "full", "compare", "interactive"], 
                       default="interactive", help="Evaluation mode")
    parser.add_argument("--method", choices=["cot", "react", "reflexion"], 
                       help="Single method to test")
    parser.add_argument("--questions", type=int, default=20, 
                       help="Number of questions for single method test")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "quick":
            run_quick_test()
        elif args.mode == "small":
            run_small_benchmark()
        elif args.mode == "medium":
            run_medium_benchmark()
        elif args.mode == "full":
            run_full_benchmark()
        elif args.mode == "compare":
            compare_methods_quick()
        elif args.mode == "interactive":
            interactive_menu()
        
        if args.method:
            run_single_method_test(args.method, args.questions)
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
