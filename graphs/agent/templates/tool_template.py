##### Tools #####
from typing import Annotated
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt import create_react_agent

# 1. just using python function example
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
# tool decorator example
@tool("multiply_tool", 
      parse_docstring=True,
      return_direct=True) # LLM으로 Tool 결과를 바탕으로 재생성 하지 않음
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First operand
        b: Second operand
    """
    return a * b
# 2. hide tool arguments from the model
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig

def my_tool(
    # This will be populated by an LLM
    tool_arg: str,
    # access information that's dynamically updated inside the agent
    state: Annotated[AgentState, InjectedState],
    # access static data that is passed at agent invocation
    config: RunnableConfig,
) -> str:
    """My tool."""
    do_something_with_state(state["messages"])
    do_something_with_config(config)
    ...
# 3. Disable parallel tool calling
from langchain.chat_models import init_chat_model

def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

model = init_chat_model("anthropic:claude-3-5-sonnet-latest", temperature=0)
tools = [add, multiply]
agent = create_react_agent(
    # disable parallel tool calls
    model=model.bind_tools(tools, parallel_tool_calls=False),
    tools=tools
)

agent.invoke(
    {"messages": [{"role": "user", "content": "what's 3 + 5 and 4 * 7?"}]}
)

# 4. Force tool use example
from langchain_core.tools import tool

@tool(return_direct=True)
def greet(user_name: str) -> int:
    """Greet user."""
    return f"Hello {user_name}!"

tools = [greet]

agent = create_react_agent(
    model=model.bind_tools(tools, tool_choice={"type": "tool", "name": "greet"}),
    tools=tools
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Hi, I am Bob"}]}
)

# 5. Handle tool errors example
from langgraph.prebuilt import create_react_agent



def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    if a == 42:
        raise ValueError("The ultimate error")
    return a * b
tool_node = ToolNode(
    [multiply],
    handle_tool_errors="Can't use 42 as a first operand, you must switch operands!"
)

# 6. Run with error handling (default)
agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[multiply]
)
agent.invoke(
    {"messages": [{"role": "user", "content": "what's 42 x 7?"}]}
)

# 7. Prebuilt tools example
# prebuilt tools are listed here: https://python.langchain.com/docs/integrations/tools/
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model="openai:gpt-4o-mini", 
    tools=[{"type": "web_search_preview"}]
)
response = agent.invoke(
    {"messages": ["What was a positive news story from today?"]}
)

