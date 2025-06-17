"""
Code Benchmark Agents: CoT, ReAct, Reflexion
LangGraph의 create_react_agent를 기반으로 한 코드 벤치마크용 에이전트들
"""

from typing import Annotated, Any, List, Dict, Optional, Literal, Union, Sequence
from typing_extensions import TypedDict
import yaml
import os
from pathlib import Path

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Checkpointer

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, AnyMessage
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import LanguageModelLike
from langchain_openai import ChatOpenAI


class CodeBenchmarkState(TypedDict):
    """코드 벤치마크 에이전트의 상태"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    benchmark_type: str  # humaneval, mbpp, leetcode, etc.
    reasoning_type: str  # cot, react, reflexion
    problem: Optional[str]
    solution: Optional[str]
    reflection: Optional[str]
    test_results: Optional[str]


# 코드 실행 도구
@tool
def execute_python_code(code: str) -> str:
    """Python 코드를 실행하고 결과를 반환합니다."""
    import subprocess
    import tempfile
    import os
    
    try:
        # 임시 파일에 코드 저장
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # 코드 실행
        result = subprocess.run(
            ['python', temp_file], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        # 임시 파일 삭제
        os.unlink(temp_file)
        
        if result.returncode == 0:
            return f"실행 성공:\n{result.stdout}"
        else:
            return f"실행 오류:\n{result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "실행 시간 초과 (10초)"
    except Exception as e:
        return f"실행 중 오류 발생: {str(e)}"


@tool
def run_test_cases(function_code: str, test_cases: str) -> str:
    """함수 코드와 테스트 케이스를 실행하여 결과를 반환합니다."""
    full_code = f"{function_code}\n\n{test_cases}"
    return execute_python_code(full_code)


@tool
def analyze_code_complexity(code: str) -> str:
    """코드의 시간/공간 복잡도를 분석합니다."""
    # 간단한 복잡도 분석 로직
    lines = code.split('\n')
    nested_loops = 0
    current_depth = 0
    
    for line in lines:
        stripped = line.strip()
        if 'for ' in stripped or 'while ' in stripped:
            current_depth += 1
            nested_loops = max(nested_loops, current_depth)
        elif stripped.startswith(('break', 'continue', 'return')):
            current_depth = max(0, current_depth - 1)
    
    if nested_loops == 0:
        complexity = "O(1) - 상수 시간"
    elif nested_loops == 1:
        complexity = "O(n) - 선형 시간"
    elif nested_loops == 2:
        complexity = "O(n²) - 제곱 시간"
    else:
        complexity = f"O(n^{nested_loops}) - {nested_loops}중 중첩"
    
    return f"예상 시간 복잡도: {complexity}"


def load_prompts() -> Dict[str, Any]:
    """프롬프트 YAML 파일을 로드합니다."""
    current_dir = Path(__file__).parent
    prompt_file = current_dir / "../prompts/code_prompt.yaml"
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # 기본 프롬프트 반환
        return {
            'prompts': {
                'humaneval': {
                    'cot': {'template': 'CoT 방식으로 문제를 해결하세요.'},
                    'react': {'template': 'ReAct 방식으로 문제를 해결하세요.'},
                    'reflexion': {'template': 'Reflexion 방식으로 문제를 해결하세요.'}
                }
            }
        }


def create_cot_agent(
    model: Union[str, LanguageModelLike],
    benchmark_type: str = "humaneval",
    tools: Optional[List[BaseTool]] = None,
    checkpointer: Optional[Checkpointer] = None,
    **kwargs
) -> Any:
    """Chain of Thought 에이전트를 생성합니다."""
    
    if tools is None:
        tools = [execute_python_code, run_test_cases, analyze_code_complexity]
    
    # 프롬프트 로드
    prompts = load_prompts()
    prompt_template = prompts.get('prompts', {}).get(benchmark_type, {}).get('cot', {}).get('template', '')
    
    # 시스템 프롬프트 구성
    system_prompt = f"""
{prompt_template}

당신은 코드 벤치마크 문제를 Chain of Thought 방식으로 해결하는 전문가입니다.

다음 단계를 따라 체계적으로 문제를 해결하세요:

1. **문제 이해**: 주어진 문제를 자세히 분석하고 요구사항을 파악합니다.
2. **단계별 추론**: 문제를 작은 단위로 나누어 단계별로 해결 방법을 생각합니다.
3. **알고리즘 설계**: 각 단계에 대한 구체적인 알고리즘을 설계합니다.
4. **코드 구현**: 설계한 알고리즘을 코드로 구현합니다.
5. **검증**: 구현한 코드를 테스트하고 검증합니다.

각 단계에서 명확한 설명과 함께 진행하세요.
"""

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
        checkpointer=checkpointer,
        **kwargs
    )


def create_react_reasoning_agent(
    model: Union[str, LanguageModelLike],
    benchmark_type: str = "humaneval",
    tools: Optional[List[BaseTool]] = None,
    checkpointer: Optional[Checkpointer] = None,
    **kwargs
) -> Any:
    """ReAct (Reasoning + Acting) 에이전트를 생성합니다."""
    
    if tools is None:
        tools = [execute_python_code, run_test_cases, analyze_code_complexity]
    
    # 프롬프트 로드
    prompts = load_prompts()
    prompt_template = prompts.get('prompts', {}).get(benchmark_type, {}).get('react', {}).get('template', '')
    
    # 시스템 프롬프트 구성
    system_prompt = f"""
{prompt_template}

당신은 ReAct (Reasoning + Acting) 방식으로 코드 벤치마크 문제를 해결하는 전문가입니다.

Thought-Action-Observation 사이클을 사용하여 문제를 해결하세요:

**Thought**: 현재 상황을 분석하고 다음에 무엇을 해야 할지 추론합니다.
**Action**: 구체적인 행동을 취합니다 (코드 분석, 구현, 테스트 등).
**Observation**: 행동의 결과를 관찰하고 다음 단계를 계획합니다.

이 과정을 반복하여 문제를 체계적으로 해결하세요.

사용 가능한 도구:
- execute_python_code: Python 코드 실행
- run_test_cases: 테스트 케이스 실행
- analyze_code_complexity: 코드 복잡도 분석

각 단계에서 명확한 Thought-Action-Observation을 보여주세요.
"""

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
        checkpointer=checkpointer,
        **kwargs
    )


def create_reflexion_agent(
    model: Union[str, LanguageModelLike],
    benchmark_type: str = "humaneval",
    tools: Optional[List[BaseTool]] = None,
    checkpointer: Optional[Checkpointer] = None,
    **kwargs
) -> Any:
    """Reflexion (자기 반성) 에이전트를 생성합니다."""
    
    if tools is None:
        tools = [execute_python_code, run_test_cases, analyze_code_complexity]
    
    # 프롬프트 로드
    prompts = load_prompts()
    prompt_template = prompts.get('prompts', {}).get(benchmark_type, {}).get('reflexion', {}).get('template', '')
    
    # 시스템 프롬프트 구성
    system_prompt = f"""
{prompt_template}

당신은 Reflexion 방식으로 코드 벤치마크 문제를 해결하는 전문가입니다.

자기 반성적 접근 방식을 사용하여 문제를 해결하세요:

1. **초기 시도**: 먼저 문제를 해결해 봅니다.
2. **자기 반성**: 초기 해결책을 비판적으로 분석합니다:
   - 모든 엣지 케이스를 다루었는가?
   - 논리가 모든 입력에 대해 올바른가?
   - 잠재적인 버그나 문제가 있는가?
   - 효율성이나 명확성을 개선할 수 있는가?
3. **문제점 식별**: 초기 시도에서 발견한 문제점들을 나열합니다.
4. **개선된 접근**: 반성을 바탕으로 개선된 접근 방식을 설명합니다.
5. **최종 구현**: 반성 결과를 반영한 개선된 솔루션을 구현합니다.
6. **최종 검증**: 개선된 솔루션을 검증합니다.
7. **학습 요약**: 이 반성적 과정에서 얻은 주요 통찰을 요약합니다.

각 단계에서 명확한 반성적 사고 과정을 보여주세요.
"""

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
        checkpointer=checkpointer,
        **kwargs
    )


def create_benchmark_agent(
    reasoning_type: Literal["cot", "react", "reflexion"],
    model: Union[str, LanguageModelLike] = "gpt-4o-mini",
    benchmark_type: str = "humaneval",
    tools: Optional[List[BaseTool]] = None,
    checkpointer: Optional[Checkpointer] = None,
    **kwargs
) -> Any:
    """
    지정된 추론 타입에 따라 적절한 벤치마크 에이전트를 생성합니다.
    
    Args:
        reasoning_type: 추론 방식 ("cot", "react", "reflexion")
        model: 사용할 언어 모델
        benchmark_type: 벤치마크 타입 (humaneval, mbpp, leetcode 등)
        tools: 사용할 도구 목록
        checkpointer: 체크포인터
        **kwargs: create_react_agent에 전달할 추가 인자
    
    Returns:
        CompiledGraph: 컴파일된 에이전트 그래프
    """
    
    agent_creators = {
        "cot": create_cot_agent,
        "react": create_react_reasoning_agent,
        "reflexion": create_reflexion_agent
    }
    
    if reasoning_type not in agent_creators:
        raise ValueError(f"지원되지 않는 추론 타입: {reasoning_type}")
    
    return agent_creators[reasoning_type](
        model=model,
        benchmark_type=benchmark_type,
        tools=tools,
        checkpointer=checkpointer,
        **kwargs
    )


# 예제 사용법
if __name__ == "__main__":
    # 메모리 체크포인터 생성
    memory = MemorySaver()
    
    # CoT 에이전트 생성
    cot_agent = create_cot_agent(
        model="gpt-4o-mini",
        benchmark_type="humaneval",
        checkpointer=memory
    )
    
    # ReAct 에이전트 생성
    react_agent = create_react_reasoning_agent(
        model="gpt-4o-mini",
        benchmark_type="humaneval",
        checkpointer=memory
    )
    
    # Reflexion 에이전트 생성
    reflexion_agent = create_reflexion_agent(
        model="gpt-4o-mini",
        benchmark_type="humaneval",
        checkpointer=memory
    )
    
    # 통합 함수로 에이전트 생성
    agent = create_benchmark_agent(
        reasoning_type="cot",
        model="gpt-4o-mini",
        benchmark_type="humaneval",
        checkpointer=memory
    )
    
    print("코드 벤치마크 에이전트들이 성공적으로 생성되었습니다!")
