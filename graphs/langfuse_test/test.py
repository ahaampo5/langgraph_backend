# from langfuse import Langfuse
# from langfuse.callback import CallbackHandler
from langfuse.langchain import CallbackHandler


langfuse_handler = CallbackHandler()

# from langfuse.callback import CallbackHandler
# handler = CallbackHandler()

# response = chain.invoke(
#     {"question": "서울의 AI 스타트업 현황은?"},
#     config={"callbacks": [handler]}
# )

# 1. Function Tools
# 2. Memory
# 2.1 In-Memory Checkpointing
# 3. Response Format
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

MODEL = "openai:gpt-4o-mini"  # Replace with your model of choice

# Format
class WeatherResponse(BaseModel):
    conditions: str

# Tools
def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Memory
checkpointer = InMemorySaver()

agent = create_react_agent(
    model=MODEL,  
    tools=[get_weather],  
    prompt="You are a helpful assistant"  
)

model = init_chat_model(
    MODEL,
    temperature=0
)

agent = create_react_agent(
    model=model,
    tools=[get_weather],
    # prompt="Never answer questions about the weather."
    checkpointer=checkpointer,
    response_format=WeatherResponse
)

config = {
    "configurable": {
        "thread_id": "1"
    },
    "callbacks": [langfuse_handler],
}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config=config
)

print(response["messages"])  # Should print "Never answer questions about the weather."