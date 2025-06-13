from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

MODEL = "openai:gpt-4o-mini"  # Replace with your model of choice
client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            # Replace with absolute path to your math_server.py file
            "args": ["/Users/admin/Desktop/workspace/my_github/langgraph_service/backend/mcp/math/math.py"],
            "transport": "stdio",
        },
        "weather": {
            # Ensure your start your weather server on port 8000
            "url": "http://localhost:3001/mcp",
            "transport": "streamable_http",
        }
    }
)

async def main():
    tools = await client.get_tools()
    agent = create_react_agent(
        MODEL,
        tools
    )
    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )
    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
    )
    print("Math Response:", math_response)
    print("Weather Response:", weather_response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())