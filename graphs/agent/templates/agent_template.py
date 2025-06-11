##### 1. Type #####
# state
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langgraph.prebuilt.chat_agent_executor import AgentState # 자유롭게 업데이트 가능
# message
from langchain_core.messages import ToolMessage, AIMessage, AnyMessage
# config
from langchain_core.runnables import RunnableConfig # runtime에서 사용되는 불변의 설정값을 저장하는 용도

##### 2. LLM #####
from langchain.chat_models import init_chat_model # disable_streaming=True # .with_fallbacks([init_chat_model(...)])
from langchain_openai import ChatOpenAI

# prebuilt agent
from langgraph.prebuilt import create_react_agent # parallel_tool_calls=False

##### 3. Tools #####
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition

# Human in the loop tools
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command, interrupt

##### 4. Prompt #####
def prompt(
    state: AgentState,
    config: RunnableConfig,
) -> list[AnyMessage]:
    user_name = config["configurable"].get("user_name")
    system_msg = f"You are a helpful assistant. User's name is {user_name}"
    return [{"role": "system", "content": system_msg}] + state["messages"]

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    prompt=prompt
)

##### Graph #####
from langgraph.graph import StateGraph, START, END

##### Memory #####
from langgraph.checkpoint.memory import MemorySaver
# In-memory checkpointer
from langgraph.checkpoint.memory import InMemorySaver

##### Config #####
config = {
    "recursion_limit": 5
}

#### Debugging #####
# graph.get_state()
# graph.update_state()
# graph.get_state_history()


llm = ChatOpenAI()
llm.invoke("Hello, world!")



##### Input format #####
# string - {"messages": "Hello"}
# Message dictionary - {"messages": {"role": "user", "content": "Hello"}}
# List of messages - {"messages": [{"role": "user", "content": "Hello"}]}
# With custom state - {"messages": [{"role": "user", "content": "Hello"}], "user_name": "Alice"}

##### State schema #####
class CustomState(AgentState):
    user_name: str

agent = create_react_agent(
    # Other agent parameters...
    state_schema=CustomState,
)

agent.invoke(
    {
    "messages": "hi!",
    "user_name": "Jane",
    },
    config=config
    )


##### Output format #####
# messages - (user input + assistant replies + tool invocations)
# structured_response - response_format=BaseModel
# state_schema - custom


##### 