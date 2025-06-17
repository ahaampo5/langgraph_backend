"""
MBPP (Mostly Basic Python Programming) 벤치마크용 에이전트
LangGraph의 create_react_agent 기반으로 구현
"""

from typing import List, Optional, Union
from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike
from langgraph.types import Checkpointer
from langgraph.checkpoint.memory import MemorySaver

from code_benchmark_agents import (
    create_benchmark_agent,
    execute_python_code,
    run_test_cases,
    analyze_code_complexity
)


def create_mbpp_cot_agent(
    model: Union[str, LanguageModelLike] = "gpt-4o-mini",
    tools: Optional[List[BaseTool]] = None,
    checkpointer: Optional[Checkpointer] = None,
    **kwargs
):
    """MBPP 문제 해결용 Chain of Thought 에이전트"""
    
    if tools is None:
        tools = [execute_python_code, run_test_cases, analyze_code_complexity]
    
    return create_benchmark_agent(
        reasoning_type="cot",
        model=model,
        benchmark_type="mbpp",
        tools=tools,
        checkpointer=checkpointer,
        **kwargs
    )


def create_mbpp_react_agent(
    model: Union[str, LanguageModelLike] = "gpt-4o-mini",
    tools: Optional[List[BaseTool]] = None,
    checkpointer: Optional[Checkpointer] = None,
    **kwargs
):
    """MBPP 문제 해결용 ReAct 에이전트"""
    
    if tools is None:
        tools = [execute_python_code, run_test_cases, analyze_code_complexity]
    
    return create_benchmark_agent(
        reasoning_type="react",
        model=model,
        benchmark_type="mbpp",
        tools=tools,
        checkpointer=checkpointer,
        **kwargs
    )


def create_mbpp_reflexion_agent(
    model: Union[str, LanguageModelLike] = "gpt-4o-mini",
    tools: Optional[List[BaseTool]] = None,
    checkpointer: Optional[Checkpointer] = None,
    **kwargs
):
    """MBPP 문제 해결용 Reflexion 에이전트"""
    
    if tools is None:
        tools = [execute_python_code, run_test_cases, analyze_code_complexity]
    
    return create_benchmark_agent(
        reasoning_type="reflexion",
        model=model,
        benchmark_type="mbpp",
        tools=tools,
        checkpointer=checkpointer,
        **kwargs
    )


# 예제 사용법
if __name__ == "__main__":
    # 메모리 체크포인터
    memory = MemorySaver()
    
    # MBPP 문제 예제
    mbpp_problem = """
Write a function to find the similar elements from the given two tuple lists.

Test cases:
assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) 
assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)
"""
    
    # CoT 에이전트로 문제 해결
    print("MBPP CoT Agent 테스트")
    cot_agent = create_mbpp_cot_agent(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "mbpp_cot_test"}}
    response = cot_agent.invoke({
        "messages": [{"role": "user", "content": mbpp_problem}]
    }, config=config)
    
    print("CoT 결과:")
    print(response["messages"][-1].content)
    
    print("\n" + "="*50)
    
    # ReAct 에이전트로 문제 해결
    print("MBPP ReAct Agent 테스트")
    react_agent = create_mbpp_react_agent(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "mbpp_react_test"}}
    response = react_agent.invoke({
        "messages": [{"role": "user", "content": mbpp_problem}]
    }, config=config)
    
    print("ReAct 결과:")
    print(response["messages"][-1].content)
    
    print("\n" + "="*50)
    
    # Reflexion 에이전트로 문제 해결
    print("MBPP Reflexion Agent 테스트")
    reflexion_agent = create_mbpp_reflexion_agent(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "mbpp_reflexion_test"}}
    response = reflexion_agent.invoke({
        "messages": [{"role": "user", "content": mbpp_problem}]
    }, config=config)
    
    print("Reflexion 결과:")
    print(response["messages"][-1].content)