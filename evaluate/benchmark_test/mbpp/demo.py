"""
MBPP 평가 데모 - 단일 문제 테스트
실제 데이터셋 없이도 테스트할 수 있는 데모
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Literal, cast

# 프로젝트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from graphs.agent.experiments.code_benchmark_agents import create_benchmark_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage


# 테스트용 MBPP 문제들
TEST_PROBLEMS = [
    {
        "task_id": "1",
        "text": "Write a function to find the similar elements from the given two tuple lists.",
        "test_cases": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)", 
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"
        ]
    },
    {
        "task_id": "2", 
        "text": "Write a python function to remove first and last occurrence of a given character from the string.",
        "test_cases": [
            "assert remove_Occ(\"hello\",\"l\") == \"heo\"",
            "assert remove_Occ(\"abcda\",\"a\") == \"bcd\"",
            "assert remove_Occ(\"PHP\",\"P\") == \"H\""
        ]
    }
]


def create_problem_prompt(problem):
    """문제를 프롬프트로 변환"""
    prompt = f"""다음 Python 프로그래밍 문제를 해결해주세요:

문제: {problem['text']}

테스트 케이스들:
"""
    for i, test in enumerate(problem['test_cases'], 1):
        prompt += f"{i}. {test}\n"
    
    prompt += """
완전한 함수를 작성해주세요. 함수명과 매개변수는 테스트 케이스에서 유추할 수 있습니다.
모든 테스트 케이스가 통과하도록 구현해주세요."""
    
    return prompt


async def test_single_problem():
    """단일 문제로 모든 에이전트 테스트"""
    problem = TEST_PROBLEMS[0]
    prompt = create_problem_prompt(problem)
    
    print("=" * 60)
    print(f"문제 {problem['task_id']} 테스트")
    print("=" * 60)
    print(f"문제: {problem['text']}")
    print(f"테스트 케이스: {len(problem['test_cases'])}개")
    print()
    
    reasoning_types: List[Literal["cot", "react", "reflexion"]] = ["cot", "react", "reflexion"]
    
    for reasoning_type in reasoning_types:
        print(f"\n--- {reasoning_type.upper()} Agent 테스트 ---")
        
        try:
            # 에이전트 생성
            memory = MemorySaver()
            agent = create_benchmark_agent(
                reasoning_type=reasoning_type,
                model="gpt-4o-mini",
                benchmark_type="mbpp",
                checkpointer=memory
            )
            
            # 문제 해결
            config = {"configurable": {"thread_id": f"demo_{reasoning_type}"}}
            
            start_time = asyncio.get_event_loop().time()
            response = await agent.ainvoke({
                "messages": [HumanMessage(content=prompt)]
            }, config=config)
            end_time = asyncio.get_event_loop().time()
            
            execution_time = end_time - start_time
            solution = response["messages"][-1].content
            
            print(f"실행 시간: {execution_time:.2f}초")
            print(f"솔루션 길이: {len(solution)}자")
            print("솔루션 미리보기:")
            print("-" * 40)
            # 솔루션의 처음 300자만 출력
            preview = solution[:300] + "..." if len(solution) > 300 else solution
            print(preview)
            print("-" * 40)
            
        except Exception as e:
            print(f"에러 발생: {e}")


async def test_all_problems():
    """모든 테스트 문제로 CoT 에이전트만 테스트"""
    print("\n" + "=" * 60)
    print("모든 문제 CoT Agent 테스트")
    print("=" * 60)
    
    memory = MemorySaver()
    agent = create_benchmark_agent(
        reasoning_type="cot",
        model="gpt-4o-mini",
        benchmark_type="mbpp",
        checkpointer=memory
    )
    
    for problem in TEST_PROBLEMS:
        print(f"\n--- 문제 {problem['task_id']} ---")
        print(f"문제: {problem['text']}")
        
        try:
            prompt = create_problem_prompt(problem)
            config = {"configurable": {"thread_id": f"all_test_{problem['task_id']}"}}
            
            start_time = asyncio.get_event_loop().time()
            response = await agent.ainvoke({
                "messages": [HumanMessage(content=prompt)]
            }, config=config)
            end_time = asyncio.get_event_loop().time()
            
            execution_time = end_time - start_time
            solution = response["messages"][-1].content
            
            print(f"실행 시간: {execution_time:.2f}초")
            print(f"솔루션 생성됨 (길이: {len(solution)}자)")
            
        except Exception as e:
            print(f"에러 발생: {e}")


async def main():
    """메인 실행"""
    print("MBPP 에이전트 데모 시작")
    
    # 단일 문제 테스트
    await test_single_problem()
    
    # 모든 문제 테스트
    await test_all_problems()
    
    print("\n데모 완료!")


if __name__ == "__main__":
    # OpenAI API 키 확인
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("export OPENAI_API_KEY='your-api-key' 로 설정해주세요.")
    else:
        asyncio.run(main())
