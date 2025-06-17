"""
MBPP 평가 실행 스크립트
간단한 테스트를 위한 래퍼
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from evaluate.benchmark_test.mbpp.evaluator import MBPPEvaluator


async def quick_test():
    """빠른 테스트 (2개 문제만)"""
    print("MBPP 빠른 테스트 시작 (2개 문제)")
    
    evaluator = MBPPEvaluator(model="gpt-4o-mini", max_problems=2)
    results = await evaluator.evaluate_all_agents(max_concurrent=1)
    
    evaluator.print_summary()
    evaluator.save_results("mbpp_quick_test.csv")
    
    print("빠른 테스트 완료!")


async def full_test():
    """전체 테스트 (10개 문제)"""
    print("MBPP 전체 테스트 시작 (10개 문제)")
    
    evaluator = MBPPEvaluator(model="gpt-4o-mini", max_problems=10)
    results = await evaluator.evaluate_all_agents(max_concurrent=2)
    
    evaluator.print_summary()
    evaluator.save_results("mbpp_full_test.csv")
    
    print("전체 테스트 완료!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="MBPP 벤치마크 평가")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="테스트 모드 선택")
    parser.add_argument("--problems", type=int, default=None,
                       help="평가할 문제 수")
    parser.add_argument("--model", default="gpt-4o-mini",
                       help="사용할 모델")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        asyncio.run(quick_test())
    elif args.mode == "full":
        asyncio.run(full_test())
    else:
        # 커스텀 설정
        async def custom_test():
            problems = args.problems or 5
            print(f"MBPP 커스텀 테스트 시작 ({problems}개 문제)")
            
            evaluator = MBPPEvaluator(model=args.model, max_problems=problems)
            results = await evaluator.evaluate_all_agents(max_concurrent=1)
            
            evaluator.print_summary()
            evaluator.save_results(f"mbpp_custom_{problems}problems.csv")
            
            print("커스텀 테스트 완료!")
        
        asyncio.run(custom_test())


if __name__ == "__main__":
    main()
