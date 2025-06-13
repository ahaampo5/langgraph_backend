from typing import Annotated
from pydantic import BaseModel
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import StateGraph, START
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage, AIMessage, AnyMessage
from langgraph.prebuilt import create_react_agent

class WeatherResponse(BaseModel):
    conditions: str

class MathAgent:
    def __init__(self, model: str = "openai:gpt-4o-mini", user_name: str = "Alice"):
        self.model = model
        self.user_name = user_name
        self.config = RunnableConfig(
            configurable={
                "user_name": self.user_name
            }
        )
        self.graph = self._build_graph()
    
    def add_number(self, a: float, b: float) -> float:  
        """Get Addition of two numbers."""
        return a + b
    
    def _prompt(self, state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
        user_name = config["configurable"].get("user_name")
        system_msg = f"You are a helpful assistant. User's name is {user_name}"
        print([{"role": "system", "content": system_msg}] + state["messages"])
        return [{"role": "system", "content": system_msg}] + state["messages"]
    
    def _build_graph(self):
        agent = create_react_agent(
            model=self.model,
            tools=[],#self.add_number],
            prompt=self._prompt
        )
        
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("agent", agent)
        graph_builder.add_edge(START, "agent")
        return graph_builder.compile()
    
    def stream_graph_updates(self, user_input: str):
        for event in self.graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=self.config
        ):
            for value in event.values():
                if isinstance(value["messages"][-1], ToolMessage):
                    print("Tool:", value["messages"][-1].tool_call.name)
                    print("Tool Args:", value["messages"][-1].tool_call.args)
                elif isinstance(value["messages"][-1], AIMessage):
                    print("Assistant:", value["messages"][-1].content)
    
    def run_interactive(self):
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                self.stream_graph_updates(user_input)
            except EOFError:
                # fallback if input() is not available
                user_input = "What do you know about LangGraph?"
                print("User: " + user_input)
                self.stream_graph_updates(user_input)
                break

# 사용 예시
if __name__ == "__main__":
    math_agent = MathAgent()
    math_agent.run_interactive()