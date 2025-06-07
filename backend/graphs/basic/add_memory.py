
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)

# from IPython.display import Image, display

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass


config = {
    "configurable": {
        "thread_id": "1"
    }
}
graph.invoke(
    {"messages": [{"role": "user", "content": "My name is Hallo"}]},
    config=config)