from typing import Optional, List, Any, Dict, Union
import yaml
import os
from pathlib import Path
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph


def load_qa_prompts() -> Dict[str, Any]:
    """Load QA prompts from YAML file."""
    current_dir = Path(__file__).parent
    prompt_file = current_dir / "../prompts/qa_prompt.yaml"
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Prompt file not found at {prompt_file}")
        return {}


# 기본 도구들 정의
@tool
def search_documents(query: str) -> str:
    """Search for relevant documents or information based on a query."""
    # 실제 구현에서는 벡터 데이터베이스나 검색 엔진 연결
    return f"Found relevant information for: {query}"


@tool
def extract_information(text: str, target: str) -> str:
    """Extract specific information from given text."""
    # 실제 구현에서는 더 정교한 정보 추출 로직
    return f"Extracted {target} from text: {text[:100]}..."


@tool
def verify_facts(claim: str) -> str:
    """Verify the accuracy of a factual claim."""
    # 실제 구현에서는 팩트 체킹 API나 지식 베이스 연결
    return f"Verification result for claim: {claim}"


@tool
def calculate_numbers(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        # 안전한 계산을 위해 eval 대신 더 안전한 방법 사용 권장
        result = eval(expression)
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


def create_cot_qa_agent(
    model: str = "openai:gpt-4o-mini",
    benchmark: str = "hotpotqa",
    tools: Optional[List[BaseTool]] = [],
    **kwargs
) -> CompiledGraph:
    """
    Chain of Thought (CoT) QA Agent를 생성합니다.
    
    Args:
        model: 사용할 언어 모델
        benchmark: QA 벤치마크 타입 (hotpotqa, squad, naturalqa 등)
        tools: 추가 도구 리스트
        **kwargs: create_react_agent에 전달할 추가 인자
    
    Returns:
        CompiledGraph: 컴파일된 LangGraph 에이전트
    """
    prompts = load_qa_prompts()
    
    # 기본 도구 설정
    # default_tools = [search_documents, extract_information, verify_facts, calculate_numbers]
    # if tools:
    #     default_tools.extend(tools)
    
    # CoT 프롬프트 가져오기
    cot_template = ""
    if prompts and benchmark in prompts.get("prompts", {}):
        cot_template = prompts["prompts"][benchmark].get("cot", {}).get("system_template", "")
    
    if not cot_template:
        cot_template = """
        You are answering questions using Chain of Thought reasoning.
        
        **Step 1: Question Analysis**
        Let me understand what this question is asking...
        
        **Step 2: Information Requirements**
        To answer this question, I need to find:
        
        **Step 3: Step-by-step Reasoning**
        Let me work through this systematically:
        
        **Step 4: Evidence Integration**
        Combining all the information:
        
        **Step 5: Final Answer**
        Based on my reasoning:
        """
    
    system_prompt = SystemMessage(content=cot_template)
    
    return create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
        **kwargs
    )


def create_react_qa_agent(
    model: str = "openai:gpt-4o-mini",
    benchmark: str = "hotpotqa",
    tools: Optional[List[BaseTool]] = [],
    **kwargs
) -> CompiledGraph:
    """
    ReAct (Reasoning and Acting) QA Agent를 생성합니다.
    
    Args:
        model: 사용할 언어 모델
        benchmark: QA 벤치마크 타입
        tools: 추가 도구 리스트
        **kwargs: create_react_agent에 전달할 추가 인자
    
    Returns:
        CompiledGraph: 컴파일된 LangGraph 에이전트
    """
    prompts = load_qa_prompts()
    
    # 기본 도구 설정
    # default_tools = [search_documents, extract_information, verify_facts, calculate_numbers]
    # if tools:
    #     default_tools.extend(tools)
    
    # ReAct 프롬프트 가져오기
    react_template = ""
    if prompts and benchmark in prompts.get("prompts", {}):
        react_template = prompts["prompts"][benchmark].get("react", {}).get("template", "")
    
    if not react_template:
        react_template = """
        You are answering questions using ReAct methodology (Reasoning and Acting).
        
        **Thought 1**: I need to understand what this question is asking.
        
        **Action 1**: [ANALYZE_QUESTION]
        
        **Observation 1**: [What you learned about the question]
        
        **Thought 2**: Based on my analysis, I should search for relevant information.
        
        **Action 2**: [SEARCH_INFORMATION]
        
        **Observation 2**: [Information found]
        
        Continue this Thought-Action-Observation cycle until you have enough information to answer.
        
        **Final Answer**: [Complete answer based on your reasoning and observations]
        """
    
    system_prompt = SystemMessage(content=react_template)
    
    return create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
        **kwargs
    )


def create_reflexion_qa_agent(
    model: str = "openai:gpt-4o-mini",
    benchmark: str = "hotpotqa",
    tools: Optional[List[BaseTool]] = [],
    **kwargs
) -> CompiledGraph:
    """
    Reflexion (Self-reflection) QA Agent를 생성합니다.
    
    Args:
        model: 사용할 언어 모델
        benchmark: QA 벤치마크 타입
        tools: 추가 도구 리스트
        **kwargs: create_react_agent에 전달할 추가 인자
    
    Returns:
        CompiledGraph: 컴파일된 LangGraph 에이전트
    """
    prompts = load_qa_prompts()
    
    # 기본 도구 설정
    # default_tools = [search_documents, extract_information, verify_facts, calculate_numbers]
    # if tools:
    #     default_tools.extend(tools)
    
    # Reflexion 프롬프트 가져오기
    reflexion_template = ""
    if prompts and benchmark in prompts.get("prompts", {}):
        reflexion_template = prompts["prompts"][benchmark].get("reflexion", {}).get("template", "")
    
    if not reflexion_template:
        reflexion_template = """
        You are answering questions using Reflexion methodology (Self-reflection and improvement).
        
        **Initial Attempt**:
        Let me first try to answer this question:
        [Initial reasoning and answer]
        
        **Self-Reflection**:
        Let me critically examine my initial approach:
        - Did I understand the question correctly?
        - Is my reasoning sound?
        - Did I consider all relevant information?
        - Are there alternative approaches?
        - What could be improved?
        
        **Issues Identified**:
        [Specific problems with the initial attempt]
        
        **Improved Strategy**:
        Based on my reflection, here's a better approach:
        [Enhanced reasoning strategy]
        
        **Refined Solution**:
        [Improved answer with better reasoning]
        
        **Learning Summary**:
        Through this reflexive process, I learned:
        [Key insights and improvements]
        """
    
    system_prompt = SystemMessage(content=reflexion_template)
    
    return create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
        **kwargs
    )


def create_io_qa_agent(
    model: str = "openai:gpt-4o-mini",
    tools: Optional[List[BaseTool]] = [],
    benchmark: str = "hotpotqa",
    **kwargs
) -> CompiledGraph:
    """
    Input-Output (IO) QA Agent를 생성합니다.
    추가적인 프롬프트 없이 질문을 직접 입력하는 기본적인 방법입니다.
    
    Args:
        model: 사용할 언어 모델
        tools: 추가 도구 리스트
        benchmark: QA 벤치마크 타입 (hotpotqa, squad, naturalqa 등)
        **kwargs: create_react_agent에 전달할 추가 인자
    
    Returns:
        CompiledGraph: 컴파일된 LangGraph 에이전트
    """
    # # IO 방식은 도구 없이 직접 답변
    # from langchain_core.language_models import BaseChatModel
    # from langgraph.graph import StateGraph, START, END
    # from langgraph.graph.message import add_messages
    # from typing_extensions import Annotated, TypedDict
    # from langchain_core.messages import BaseMessage
    
    # # 모델 로딩
    # if model.startswith("openai:"):
    #     from langchain_openai import ChatOpenAI
    #     model_name = model.replace("openai:", "")
    #     llm = ChatOpenAI(model=model_name, temperature=0)
    # elif model.startswith("anthropic:"):
    #     from langchain_anthropic import ChatAnthropic
    #     model_name = model.replace("anthropic:", "")
    #     llm = ChatAnthropic(model_name=model_name, temperature=0, timeout=60, stop=None)
    # else:
    #     raise ValueError(f"Unsupported model: {model}")
    
    # # 상태 정의
    # class State(TypedDict):
    #     messages: Annotated[list[BaseMessage], add_messages]
    
    # # 간단한 응답 노드
    # def respond(state: State):
    #     return {"messages": [llm.invoke(state["messages"])]}
    
    # # 그래프 구성
    # workflow = StateGraph(State)
    # workflow.add_node("respond", respond)
    # workflow.add_edge(START, "respond")
    # workflow.add_edge("respond", END)
    
    # return workflow.compile()
    return create_react_agent(
        model=model,
        tools=tools,
        **kwargs
    )


def create_qa_agent(
    reasoning_type: str,
    model: str = "openai:gpt-4o-mini",
    benchmark: str = "hotpotqa",
    tools: Optional[List[BaseTool]] = None,
    **kwargs
) -> CompiledGraph:
    """
    지정된 추론 방법론에 따라 QA Agent를 생성합니다.
    
    Args:
        reasoning_type: 추론 방법론 ("cot", "react", "reflexion")
        model: 사용할 언어 모델
        benchmark: QA 벤치마크 타입
        tools: 추가 도구 리스트
        **kwargs: create_react_agent에 전달할 추가 인자
    
    Returns:
        CompiledGraph: 컴파일된 LangGraph 에이전트
    """
    if reasoning_type.lower() == "cot":
        return create_cot_qa_agent(model, benchmark, tools, **kwargs)
    elif reasoning_type.lower() == "react":
        return create_react_qa_agent(model, benchmark, tools, **kwargs)
    elif reasoning_type.lower() == "reflexion":
        return create_reflexion_qa_agent(model, benchmark, tools, **kwargs)
    else:
        raise ValueError(f"Unsupported reasoning type: {reasoning_type}. Choose from 'cot', 'react', 'reflexion'")


def run_qa_agent_interactive(agent: CompiledGraph):
    """QA Agent를 대화형으로 실행합니다."""
    print("QA Agent가 시작되었습니다. 'quit', 'exit', 'q'를 입력하면 종료됩니다.")
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            print("\nAssistant:")
            for event in agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                stream_mode="updates"
            ):
                for value in event.values():
                    if "messages" in value and value["messages"]:
                        last_message = value["messages"][-1]
                        if hasattr(last_message, 'content') and last_message.content:
                            print(last_message.content)
                            
        except EOFError:
            print("\nGoodbye!")
            break
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


# 사용 예시
if __name__ == "__main__":
    # CoT Agent 테스트
    print("=== Chain of Thought QA Agent ===")
    cot_agent = create_cot_qa_agent(benchmark="hotpotqa")
    
    # ReAct Agent 테스트  
    print("\n=== ReAct QA Agent ===")
    react_agent = create_react_qa_agent(benchmark="hotpotqa")
    
    # Reflexion Agent 테스트
    print("\n=== Reflexion QA Agent ===")
    reflexion_agent = create_reflexion_qa_agent(benchmark="hotpotqa")
    
    # IO Agent 테스트
    print("\n=== IO QA Agent ===")
    io_agent = create_io_qa_agent(benchmark="hotpotqa")
    
    # 통합 함수로 Agent 생성
    print("\n=== Using create_qa_agent function ===")
    unified_agent = create_qa_agent("react", benchmark="hotpotqa")
    
    # 대화형 실행 (원하는 agent 선택)
    print("\nStarting interactive session with ReAct agent...")
    run_qa_agent_interactive(react_agent)