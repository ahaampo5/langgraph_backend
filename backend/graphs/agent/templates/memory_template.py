##### Memory #####
from langgraph.checkpoint.memory import MemorySaver
# In-memory checkpointer
from langgraph.checkpoint.memory import InMemorySaver


# 1. short-term memory
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver() 


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    checkpointer=checkpointer 
)

# Run the agent
config = {
    "configurable": {
        "thread_id": "1"  
    }
}

sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config
)

# Continue the conversation using the same thread_id
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    config 
)

# 2. manage message history
# 2.1 summarization_node
from langchain_anthropic import ChatAnthropic
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import InMemorySaver
from typing import Any

model = ChatAnthropic(model="claude-3-7-sonnet-latest")

summarization_node = SummarizationNode( 
    token_counter=count_tokens_approximately,
    model=model,
    max_tokens=384,
    max_summary_tokens=128,
    output_messages_key="llm_input_messages",
)

class State(AgentState):
    # NOTE: we're adding this key to keep track of previous summary information
    # to make sure we're not summarizing on every LLM call
    context: dict[str, Any]  


checkpointer = InMemorySaver() 

agent = create_react_agent(
    model=model,
    tools=tools,
    pre_model_hook=summarization_node, 
    state_schema=State, 
    checkpointer=checkpointer,
)

# 2.2 trim message history
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately
)
from langgraph.prebuilt import create_react_agent

# This function will be called every time before the node that calls LLM
def pre_model_hook(state):
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=384,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"llm_input_messages": trimmed_messages}

checkpointer = InMemorySaver()
agent = create_react_agent(
    model,
    tools,
    pre_model_hook=pre_model_hook,
    checkpointer=checkpointer,
)

# 3. read/write state
# 3.1 read state
from typing import Annotated
from langgraph.prebuilt import InjectedState, create_react_agent

class CustomState(AgentState):
    user_id: str

def get_user_info(
    state: Annotated[CustomState, InjectedState]
) -> str:
    """Look up user info."""
    user_id = state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_user_info],
    state_schema=CustomState,
)

agent.invoke({
    "messages": "look up user information",
    "user_id": "user_123"
})
# 3.2 write state
from typing import Annotated
from langchain_core.tools import InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command

class CustomState(AgentState):
    user_name: str

def update_user_info(
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig
) -> Command:
    """Look up and update user info."""
    user_id = config["configurable"].get("user_id")
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        # update the message history
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=tool_call_id
            )
        ]
    })

def greet(
    state: Annotated[CustomState, InjectedState]
) -> str:
    """Use this to greet the user once you found their info."""
    user_name = state["user_name"]
    return f"Hello {user_name}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[update_user_info, greet],
    state_schema=CustomState
)

agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    config={"configurable": {"user_id": "user_123"}}
)

# 4. long-term memory
# 4.1 read
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

store = InMemoryStore() 

store.put(  
    ("users",),  
    "user_123",  
    {
        "name": "John Smith",
        "language": "English",
    } 
)

def get_user_info(config: RunnableConfig) -> str:
    """Look up user info."""
    # Same as that provided to `create_react_agent`
    store = get_store() 
    user_id = config["configurable"].get("user_id")
    user_info = store.get(("users",), user_id) 
    return str(user_info.value) if user_info else "Unknown user"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_user_info],
    store=store 
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "look up user information"}]},
    config={"configurable": {"user_id": "user_123"}}
)
# 4.2 write
from typing_extensions import TypedDict

from langgraph.config import get_store
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

store = InMemoryStore() 

class UserInfo(TypedDict): 
    name: str

def save_user_info(user_info: UserInfo, config: RunnableConfig) -> str: 
    """Save user info."""
    # Same as that provided to `create_react_agent`
    store = get_store() 
    user_id = config["configurable"].get("user_id")
    store.put(("users",), user_id, user_info) 
    return "Successfully saved user info."

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[save_user_info],
    store=store
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith"}]},
    config={"configurable": {"user_id": "user_123"}} 
)

# You can access the store directly to get the value
store.get(("users",), "user_123").value

# 5. semantic memory


# 6. Prebuilt long-term memory - langmem
# Import core components 
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# Set up storage 
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
) 

# Create an agent with memory capabilities 
agent = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    tools=[
        # Memory tools use LangGraph's BaseStore for persistence (4)
        create_manage_memory_tool(namespace=("memories",)),
        create_search_memory_tool(namespace=("memories",)),
    ],
    store=store,
)