# filepath: /Users/admin/Desktop/workspace/my_github/langgraph_backend/backend/graphs/agent/autoagent.py

"""
계획 기반 다단계 에이전트 (AutoAgent)
- 사용자 질문을 분석하여 계획을 수립
- 각 단계별로 LLM이 실행
- 웹 검색, 파일 시스템, MCP 도구 활용
- 메모리 관리 및 요약 기능
"""

from typing import Annotated, Any, List, Dict, Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

# MCP imports
from langchain_mcp_adapters.client import MultiServerMCPClient

import os
import json
import asyncio
from pathlib import Path


class PlanStep(BaseModel):
    """계획의 각 단계를 나타내는 클래스"""
    step_id: int
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    result: Optional[str] = None
    tools_needed: List[str] = []


class Plan(BaseModel):
    """전체 계획을 나타내는 클래스"""
    goal: str
    steps: List[PlanStep]
    current_step: int = 0
    status: Literal["planning", "executing", "completed", "failed"] = "planning"


class AutoAgentState(AgentState):
    """AutoAgent 에이전트의 상태를 관리하는 클래스"""
    messages: Annotated[list[AnyMessage], add_messages]
    plan: Optional[Plan] = None
    current_step_result: Optional[str] = None
    context: Dict[str, Any] = {}
    user_query: str = ""
    final_answer: Optional[str] = None


# 도구 정의
@tool
def web_search(query: str, max_results: int = 5) -> str:
    """웹에서 정보를 검색합니다."""
    try:
        tavily = TavilySearch(max_results=max_results)
        results = tavily.run(query)
        return f"웹 검색 결과 ('{query}'):\n{results}"
    except Exception as e:
        return f"웹 검색 중 오류 발생: {str(e)}"


@tool
def read_file(file_path: str) -> str:
    """파일의 내용을 읽습니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return f"파일 '{file_path}' 내용:\n{content}"
    except FileNotFoundError:
        return f"파일을 찾을 수 없습니다: {file_path}"
    except Exception as e:
        return f"파일 읽기 중 오류 발생: {str(e)}"


@tool 
def write_file(file_path: str, content: str) -> str:
    """파일에 내용을 씁니다."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"파일 '{file_path}'에 성공적으로 저장했습니다."
    except Exception as e:
        return f"파일 쓰기 중 오류 발생: {str(e)}"


@tool
def list_directory(directory_path: str) -> str:
    """디렉토리의 파일 목록을 조회합니다."""
    try:
        path = Path(directory_path)
        if not path.exists():
            return f"디렉토리가 존재하지 않습니다: {directory_path}"
        
        files = []
        for item in path.iterdir():
            if item.is_file():
                files.append(f"📄 {item.name}")
            elif item.is_dir():
                files.append(f"📁 {item.name}/")
        
        return f"디렉토리 '{directory_path}' 내용:\n" + "\n".join(files)
    except Exception as e:
        return f"디렉토리 조회 중 오류 발생: {str(e)}"


# 기본 도구 목록
TOOLS = [web_search, read_file, write_file, list_directory]


def create_planner_node(model: ChatOpenAI):
    """계획 수립 노드를 생성합니다."""
    
    def planner(state: AutoAgentState, config: RunnableConfig) -> dict:
        """사용자 질문을 분석하여 실행 계획을 수립합니다."""
        
        user_query = state["user_query"]
        if not user_query and state["messages"]:
            # 마지막 사용자 메시지에서 질문 추출
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break
        
        planning_prompt = f"""사용자의 질문을 분석하여 단계별 실행 계획을 수립해주세요.

사용자 질문: {user_query}

다음 도구들을 활용할 수 있습니다:
- web_search: 웹에서 정보 검색
- read_file: 파일 내용 읽기  
- write_file: 파일에 내용 쓰기
- list_directory: 디렉토리 파일 목록 조회

계획을 다음 JSON 형식으로 제공해주세요:
{{
    "goal": "사용자 질문에 대한 목표",
    "steps": [
        {{
            "step_id": 1,
            "description": "첫 번째 단계 설명",
            "tools_needed": ["필요한_도구1", "필요한_도구2"]
        }},
        {{
            "step_id": 2, 
            "description": "두 번째 단계 설명",
            "tools_needed": ["필요한_도구3"]
        }}
    ]
}}

각 단계는 명확하고 실행 가능해야 하며, 논리적 순서로 배열되어야 합니다."""
        
        messages = [SystemMessage(content=planning_prompt)]
        response = model.invoke(messages)
        
        try:
            # JSON 추출 (```json 마크다운 블록 처리)
            content = response.content
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            elif "{" in content:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                json_str = content[json_start:json_end]
            else:
                raise ValueError("JSON 형식을 찾을 수 없습니다.")
            
            plan_data = json.loads(json_str)
            
            # Plan 객체 생성
            steps = [PlanStep(**step) for step in plan_data["steps"]]
            plan = Plan(
                goal=plan_data["goal"],
                steps=steps,
                status="executing"
            )
            
            # 상태 업데이트
            ai_message = AIMessage(content=f"계획을 수립했습니다:\n목표: {plan.goal}\n단계 수: {len(plan.steps)}")
            
            return {
                "messages": [ai_message],
                "plan": plan,
                "user_query": user_query
            }
            
        except Exception as e:
            error_msg = f"계획 수립 중 오류 발생: {str(e)}"
            return {
                "messages": [AIMessage(content=error_msg)],
                "plan": None
            }
    
    return planner


def create_executor_node(model: ChatOpenAI, tools: List):
    """계획 실행 노드를 생성합니다."""
    
    def executor(state: AutoAgentState, config: RunnableConfig) -> dict:
        """현재 단계를 실행합니다."""
        
        plan = state["plan"]
        if not plan or plan.current_step >= len(plan.steps):
            return {"messages": [AIMessage(content="실행할 단계가 없습니다.")]}
        
        current_step = plan.steps[plan.current_step]
        current_step.status = "in_progress"
        
        # 실행 프롬프트
        execution_prompt = f"""다음 단계를 실행해주세요:

목표: {plan.goal}
현재 단계 ({current_step.step_id}/{len(plan.steps)}): {current_step.description}

사용 가능한 도구: {', '.join(current_step.tools_needed) if current_step.tools_needed else '모든 도구'}

이전 단계들의 결과를 참고하여 현재 단계를 완료해주세요.
필요한 경우 도구를 사용하고, 단계 완료 후 결과를 명확히 설명해주세요."""
        
        # 모델에 도구 바인딩
        model_with_tools = model.bind_tools(tools)
        
        messages = [SystemMessage(content=execution_prompt)] + state["messages"]
        response = model_with_tools.invoke(messages)
        
        # 도구 호출이 있는 경우 처리
        if response.tool_calls:
            tool_node = ToolNode(tools)
            
            # 도구 실행을 위한 메시지 준비
            tool_state = {"messages": messages + [response]}
            tool_result = tool_node.invoke(tool_state)
            tool_messages = tool_result["messages"]
            
            # 도구 결과를 포함한 최종 응답 생성
            final_messages = messages + [response] + tool_messages
            final_response = model.invoke(final_messages)
            
            current_step.result = final_response.content
            current_step.status = "completed"
            
            return {
                "messages": [response] + tool_messages + [final_response],
                "plan": plan,
                "current_step_result": final_response.content
            }
        else:
            current_step.result = response.content
            current_step.status = "completed"
            
            return {
                "messages": [response],
                "plan": plan,
                "current_step_result": response.content
            }
    
    return executor


def create_finalizer_node(model: ChatOpenAI):
    """최종 답변 생성 노드를 생성합니다."""
    
    def finalizer(state: AutoAgentState, config: RunnableConfig) -> dict:
        """모든 단계 완료 후 최종 답변을 생성합니다."""
        
        plan = state["plan"]
        if not plan:
            return {"messages": [AIMessage(content="계획이 없어 최종 답변을 생성할 수 없습니다.")]}
        
        # 모든 단계 결과 수집
        step_results = []
        for i, step in enumerate(plan.steps, 1):
            step_results.append(f"단계 {i}: {step.description}\n결과: {step.result}")
        
        finalization_prompt = f"""
사용자의 질문에 대한 최종 답변을 생성해주세요.

원래 질문: {state['user_query']}
목표: {plan.goal}

실행한 단계들과 결과:
{chr(10).join(step_results)}

위 결과들을 종합하여 사용자에게 도움이 되는 완전하고 정확한 답변을 제공해주세요.
"""
        
        messages = [SystemMessage(content=finalization_prompt)]
        response = model.invoke(messages)
        
        return {
            "messages": [response],
            "final_answer": response.content
        }
    
    return finalizer


def create_summarization_node(model: ChatOpenAI):
    """메시지 요약 노드를 생성합니다."""
    
    def summarizer(state: AutoAgentState, config: RunnableConfig) -> dict:
        """메시지가 너무 길어지면 요약합니다."""
        
        messages = state["messages"]
        
        # 토큰 수 확인 (대략적)
        total_tokens = count_tokens_approximately(messages)
        
        if total_tokens > 8000:  # 임계값
            # 최신 메시지들 유지하면서 이전 메시지들 요약
            recent_messages = messages[-10:]  # 최근 10개 메시지 유지
            old_messages = messages[:-10]
            
            if old_messages:
                summary_prompt = """다음 대화 내용을 간결하게 요약해주세요. 중요한 정보와 컨텍스트는 보존하되, 불필요한 세부사항은 제거해주세요.

대화 내용:""" + "\n".join([f"{msg.type}: {msg.content}" for msg in old_messages])
                
                summary_response = model.invoke([SystemMessage(content=summary_prompt)])
                summary_message = SystemMessage(
                    content=f"[이전 대화 요약] {summary_response.content}"
                )
                
                return {
                    "messages": [summary_message] + recent_messages
                }
        
        return {"messages": messages}
    
    return summarizer


def should_continue(state: AutoAgentState) -> str:
    """다음 단계로 진행할지 결정합니다."""
    
    plan = state["plan"]
    if not plan:
        return "end"
    
    if plan.current_step < len(plan.steps):
        # 다음 단계로 진행
        plan.current_step += 1
        if plan.current_step < len(plan.steps):
            return "execute"
        else:
            # 모든 단계 완료
            plan.status = "completed"
            return "finalize"
    
    return "end"


def needs_summarization(state: AutoAgentState) -> str:
    """메시지 요약이 필요한지 확인합니다."""
    
    total_tokens = count_tokens_approximately(state["messages"])
    if total_tokens > 8000:
        return "summarize"
    return "continue"


class AutoAgent:
    """AutoAgent 클래스"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", enable_mcp: bool = False):
        self.model = ChatOpenAI(model=model_name, temperature=0)
        self.tools = TOOLS.copy()
        self.enable_mcp = enable_mcp
        self.mcp_client = None
        
        # MCP 클라이언트 설정
        if enable_mcp:
            self._setup_mcp()
        
        # 메모리 설정
        self.checkpointer = MemorySaver()

        # 그래프 구성
        self.graph = self._create_graph()
    
    def _setup_mcp(self):
        """MCP 클라이언트를 설정합니다."""
        try:
            self.mcp_client = MultiServerMCPClient({
                "math": {
                    "command": "python",
                    "args": ["/Users/admin/Desktop/workspace/my_github/langgraph_backend/backend/mcp/math/math.py"],
                    "transport": "stdio",
                },
                "web_search": {
                    "url": "http://localhost:8931/mcp",
                    "transport": "streamable_http",
                },
                "playwright": {
                    "url": "http://localhost:8080/mcp",
                    "transport": "streamable_http",
                }
            })
        except Exception as e:
            print(f"MCP 설정 중 오류: {e}")
    
    async def _get_mcp_tools(self):
        """MCP 도구들을 가져옵니다."""
        if self.mcp_client:
            try:
                mcp_tools = await self.mcp_client.get_tools()
                return mcp_tools
            except Exception as e:
                print(f"MCP 도구 가져오기 실패: {e}")
        return []
    
    def _create_graph(self):
        """LangGraph를 구성합니다."""
        
        # 그래프 생성
        workflow = StateGraph(AutoAgentState)
        
        # 노드 추가
        workflow.add_node("planner", create_planner_node(self.model))
        workflow.add_node("executor", create_executor_node(self.model, self.tools))
        workflow.add_node("finalizer", create_finalizer_node(self.model))
        workflow.add_node("summarizer", create_summarization_node(self.model))
        
        # 시작점 설정
        workflow.add_edge(START, "planner")
        
        # 조건부 엣지 설정
        workflow.add_conditional_edges(
            "planner",
            should_continue,
            {
                "execute": "executor",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "executor", 
            should_continue,
            {
                "execute": "executor",
                "finalize": "finalizer", 
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "finalizer",
            needs_summarization,
            {
                "summarize": "summarizer",
                "continue": END
            }
        )
        
        workflow.add_edge("summarizer", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def ainvoke(self, user_input: str, thread_id: str = "default") -> dict:
        """비동기로 에이전트를 실행합니다."""
        
        # MCP 도구 추가
        if self.enable_mcp:
            mcp_tools = await self._get_mcp_tools()
            self.tools.extend(mcp_tools)
        
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input
        }
        
        result = await self.graph.ainvoke(initial_state, config)
        return result
    
    def invoke(self, user_input: str, thread_id: str = "default") -> dict:
        """동기로 에이전트를 실행합니다."""
        
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input
        }
        
        result = self.graph.invoke(initial_state, config)
        return result


# 사용 예시
async def main():
    """AutoAgent 에이전트 테스트"""
    
    # 에이전트 생성
    agent = AutoAgent(enable_mcp=True)
    
    # 테스트 질문들
    test_queries = [
        # "김치 레시피를 찾아서 단계별로 정리해주세요",
        # "현재 디렉토리의 파일들을 확인하고 README 파일을 작성해주세요",
        # "Python으로 간단한 계산기 프로그램을 만들어주세요"
        "구글 들어가서 LangGraph 검색하고 github에 들어가서 README.md 첫 줄 출력해줘"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"테스트 {i}: {query}")
        print('='*50)
        
        try:
            result = await agent.ainvoke(query, thread_id=f"test_{i}")
            
            print(f"최종 답변: {result.get('final_answer', '답변을 생성하지 못했습니다.')}")
            
            if result.get('plan'):
                print(f"\n실행된 계획:")
                plan = result['plan']
                print(f"목표: {plan.goal}")
                for step in plan.steps:
                    print(f"- 단계 {step.step_id}: {step.description} ({step.status})")
                    
        except Exception as e:
            print(f"오류 발생: {e}")


if __name__ == "__main__":
    asyncio.run(main())