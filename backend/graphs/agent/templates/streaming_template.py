#### Streaming output ##### : Multi-Agent 에 효과적임
# For Agent progress - stream_mode="updates" ##
agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
)
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    stream_mode="updates"
):
    print(chunk)
    print("\n")
# For LLM tokens - stream_mode="messages" ##
agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
)
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    stream_mode="messages"
):
    print("Token", token)
    print("Metadata", metadata)
    print("\n")
# For Tool updates - stream_mode="custom" ##
from langgraph.config import get_stream_writer
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    # stream any arbitrary data
    writer(f"Looking up data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    stream_mode="custom"
):
    print(chunk)
    print("\n")
