import inspect
from typing import (
    Union, Callable, Any, TypeVar, Sequence, Type, Optional, Literal, get_type_hints, cast
)

from langchain_core.language_models import (
    BaseChatModel,
    LanguageModelInput,
    LanguageModelLike
)

from langchain_core.messages import (
    AIMessage, AnyMessage, BaseMessage, SystemMessage, ToolMessage
)

from langchain_core.runnables import (
    Runnable, RunnableBinding, RunnableConfig, RunnableSequence
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict

from langgraph.errors import ErrorCode, create_error_message
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Send
from langgraph.utils.runnable import RunnableCallable, RunnableLike

StructuredResponse = Union[dict, BaseModel]
StructuredResponseSchema = Union[dict, type[BaseModel]]
F = TypeVar("F", bound=Callable[..., Any])

class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    is_last_step: IsLastStep

    remaining_steps: RemainingSteps


class AgentStatePydantic(BaseModel):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    remaining_steps: RemainingSteps = 25

class AgentStateWithStructuredResponse(AgentState):
    """The state of the agent with structured response."""

    structured_response: StructuredResponse


class AgentStateWithStructuredResponsePydantic(AgentStatePydantic):
    """The state of the agent with structured response."""

    structured_response: StructuredResponse


StateSchema = TypeVar("StateSchema", bound=Union[AgentState, AgentStatePydantic])
StateSchemaType = Type[StateSchema]

PROMPT_RUNNABLE_NAME = "Prompt"

Prompt = Union[
    SystemMessage,
    str,
    Callable[[StateSchema], LanguageModelInput],
    Runnable[StateSchema, LanguageModelInput],
]

def _get_state_value(state: StateSchema, key: str, default: Any = None) -> Any:
    """Get a value from the state."""
    return (
        state.get(key, default)
        if isinstance(state, dict)
        else getattr(state, key, default)
    )


def _get_prompt_runnable(prompt: Optional[Prompt]) -> Runnable:
    prompt_runnable: Runnable
    if prompt is None:
        prompt_runnable = RunnableCallable(
            lambda state: _get_state_value(state, "messages"), name=PROMPT_RUNNABLE_NAME
        )
    elif isinstance(prompt, str):
        _system_message: BaseMessage = SystemMessage(content=prompt)
        prompt_runnable = RunnableCallable(
            lambda state: [_system_message] + _get_state_value(state, "messages"),
            name=PROMPT_RUNNABLE_NAME
        )
    elif isinstance(prompt, SystemMessage):
        prompt_runnable = RunnableCallable(
            lambda state: [prompt] + _get_state_value(state, "messages"),
            name=PROMPT_RUNNABLE_NAME
        )
    elif inspect.iscoroutinefunction(prompt):
        prompt_runnable = RunnableCallable(
            None,
            prompt,
            name=PROMPT_RUNNABLE_NAME
        )
    elif callable(prompt):
        prompt_runnable = RunnableCallable(
            prompt,
            name=PROMPT_RUNNABLE_NAME
        )
    elif isinstance(prompt, Runnable):
        prompt_runnable = prompt
    else:
        raise ValueError(f"Got unexpected type for `prompt`: {type(prompt)}")
    
    return prompt_runnable # 왜 prompt를 runnable로 변환하는지?


def _should_bind_tools(
        model: LanguageModelLike, tools: Sequence[BaseTool]) -> bool:
    if isinstance(model, RunnableSequence):
        model = next(
            (
                step
                for step in model.steps
                if isinstance(step, (RunnableBinding, BaseChatModel))
            ),
            model
        )

    if not isinstance(model, RunnableBinding):
        return True
    
    if "tools" not in model.kwargs:
        return True
    
    bound_tools = model.kwargs["tools"]
    if len(tools) != len(bound_tools):
        raise ValueError(
            "Number of tools in the model.bind_tools() and tools passed to create_react_agent must match"
        )
    
    tool_names = set(tool.name for tool in tools)
    bound_tool_names = set()
    for bound_tool in bound_tools:
        # OpenAI-style tool
        if bound_tool.get("type") == "function":
            bound_tool_name = bound_tool["function"]["name"]
        # Anthropic-style tool
        elif bound_tool.get("name"):
            bound_tool_name = bound_tool["name"]
        else:
            continue

        bound_tool_names.add(bound_tool_name)

    if missing_tools := tool_names - bound_tool_names:
        raise ValueError(f"Missing tools '{missing_tools}' in the model.bind_tools()")
    
    return False


def _get_model(model: LanguageModelLike) -> BaseChatModel:
    """Get the underlying model from a RunnableBinding or return the model itself."""
    if isinstance(model, RunnableSequence):
        model = next(
            (
                step
                for step in model.steps
                if isinstance(step, (RunnableBinding, BaseChatModel))
            ),
            model
        )

    if isinstance(model, RunnableBinding):
        model = model.bound

    if not isinstance(model, BaseChatModel):
        raise TypeError(
            f"Expected `model` to be a ChatModel or RunnableBinding (e.g. model.bind_tools(...)), got {type(model)} instead."
        )
    
    return model


def _validate_chat_history(
    messages: Sequence[BaseMessage],
) -> None:
    """Validate that all tool calls in AIMessages have a corresponding ToolMessage."""
    all_tool_calls = [
        tool_call
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]
    tool_call_ids_with_results = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    tool_calls_without_results = [
        tool_call
        for tool_call in all_tool_calls
        if tool_call["id"] not in tool_call_ids_with_results
    ]
    if not tool_calls_without_results:
        return
    
    error_message = create_error_message(
        message="Found AIMessages with tool_calls that do not have a corresponding ToolMessage. "
        f"Here are the first few of those tool calls: {tool_calls_without_results[:3]}.\n\n"
        "Every tool call (LLM requesting to call a tool) in the message history MUST have a corresponding ToolMessage "
        "(result of a tool invocation to return to the LLM) - this is required by most LLM providers.",
        error_code=ErrorCode.INVALID_CHAT_HISTORY,
    )
    raise ValueError(error_message)


def create_reasoning_agent(
    model: Union[str, LanguageModelLike],
    tools: Union[Sequence[Union[BaseTool, Callable, dict[str, Any]]], ToolNode],
    *,
    prompt: Optional[Prompt] = None,
    reasoning_type: Literal["cot", "cot-sc", "react", "reflecxion"] = "react",
    response_format: Optional[
        Union[StructuredResponseSchema, tuple[str, StructuredResponseSchema]]
    ] = None,
    pre_model_hook: Optional[RunnableLike] = None,
    post_model_hook: Optional[RunnableLike] = None,
    state_schema: Optional[StateSchemaType] = None,
    config_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer],
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
    version: Literal["v1", "v2"] = "v2",
    name: Optional[str] = None,
) -> CompiledGraph:
    """
    Creates an agent graph that calls tools in a loop until a stopping condition is met.
    For more details on using `create_reasoning_agent`, see below.

    Args:
    model: The `LangChain` chat model that supports tool calling.
    tools: A list of tools or a ToolNode instance.
        If an empty list is provided, the agent will consist of a single node without tool calling.
    prompt: An optional prompt for the LLM. Can take a few different forms:
    """
    if version not in ("v1", "v2"):
        raise ValueError(
            f"Invalid version {version}. Supported versions are 'v1' and 'v2'."
        )
    
    if state_schema is not None:
        required_keys = {"messages", "remaining_steps"}
        if response_format is not None:
            required_keys.add("structured_response")

        schema_keys = set(get_type_hints(state_schema))
        if missing_keys := required_keys - set(schema_keys):
            raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")
        
    if state_schema is None:
        state_schema = (
            AgentStateWithStructuredResponse
            if response_format is not None
            else AgentState
        )

    llm_builtin_tools: list[dict] = []
    if isinstance(tools, ToolNode):
        tool_classes = list(tools.tools_by_name.values())
        tool_node = tools
    else:
        llm_builtin_tools = [t for t in tools if isinstance(t, dict)]
        tool_node = ToolNode([t for t in tools if not isinstance(t, dict)])
        tool_classes = list(tool_node.tools_by_name.values())

    if isinstance(model, str):
        try:
            from langchain.chat_models import (
                init_chat_model
            )
        except ImportError:
            raise ImportError(
                "Please install langchain (`pip install langchain`) to use '<provider>:<model>' string syntax for `model` parameter."
            )

        model = cast(BaseChatModel, init_chat_model(model))

    tool_calling_enabled = len(tool_classes) > 0   

    if (
        _should_bind_tools(model, tool_classes) 
        and len(tool_classes) > 0
        or (len(llm_builtin_tools) > 0)
    ):
        model = cast(BaseChatModel, model).bind_tools(tool_classes + llm_builtin_tools)

    model_runnable = _get_prompt_runnable(prompt) | model

    # If any of the tools are configured to return_directly after running,
    # our graph needs to check if these were called
    should_return_direct = {t.name for t in tool_classes if t.return_direct}

    def _are_more_steps_needed(state: StateSchema, response: BaseMessage) -> bool:
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        all_tools_return_direct = (
            all(call["name"] in should_return_direct for call in response.tool_calls)
            if isinstance(response, AIMessage)
            else False
        )
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        is_last_step = _get_state_value(state, "is_last_step", False)
        return (
            (remaining_steps is None and is_last_step and has_tool_calls)
            or (
                remaining_steps is not None
                and remaining_steps < 1
                and all_tools_return_direct
            )
            or (remaining_steps is not None and remaining_steps < 2 and has_tool_calls)
        )
    
    def _get_model_input_state(state: StateSchema) -> StateSchema:
        if pre_model_hook is not None:
            messages = (
                _get_state_value(state, "llm_input_messages")
            ) or _get_state_value(state, "messages")
            error_msg = f"Expected input to call_model to have 'llm_input_messages' or 'messages' key, but got {state}"
        else:
            messages = _get_state_value(state, "messages")
            error_msg = (
                f"Expected input to call_model to have 'messages' key, but got {state}"
            )

        if messages is None:
            raise ValueError(error_msg)
        
        _validate_chat_history(messages)
        # we're passing messages under `messages` key, as this is expected by the prompt
        if isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
            state.messages = messages
        else:
            state["messages"] = messages

        return state
    
    # Define the function that calls the model
    def call_model(state: StateSchema, config: RunnableConfig) -> StateSchema:
        state = _get_model_input_state(state)
        response = cast(AIMessage, model_runnable.invoke(state, config))
        # add agent name to the AIMessage
        response.name = name

        if _are_more_steps_needed(state, response):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}
    
    async def acall_model(state: StateSchema, config: RunnableConfig) -> StateSchema:
        state = _get_model_input_state(state)
        response = cast(AIMessage, await model_runnable.ainvoke(state, config))
        # add agent name to the AIMessage
        response.name = name
        if _are_more_steps_needed(state, response):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}
    
    input_schema: StateSchemaType
    if pre_model_hook is not None:
        # Dynamically create a schema that inherits from state_schema and adds 'llm_input_messages'
        if isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
            # For Pydantic schemas
            from pydantic import create_model

            input_schema = create_model(
                "CallModelInputSchema",
                llm_input_messages=(list[AnyMessage], ...),
                __base__=state_schema,
            )
        else:
            # For TypedDict schemas
            class CallModelInputSchema(state_schema):  # type: ignore
                llm_input_messages: list[AnyMessage]

            input_schema = CallModelInputSchema
    else:
        input_schema = state_schema

    def generate_structured_response(
        state: StateSchema, config: RunnableConfig
    ) -> StateSchema:
        messages = _get_state_value(state, "messagese")
        structured_response_schema = response_format
        if isinstance(response_format, tuple):
            system_prompt, structured_response_schema = response_format
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        model_with_structured_output = _get_model(model).with_structured_output(
            cast(StructuredResponseSchema, structured_response_schema)
        )
        response = model_with_structured_output.invoke(messages, config)
        return {"structured_response": response}

    async def agenerate_structured_response(
        state: StateSchema, config: RunnableConfig
    ) -> StateSchema:
        messages = _get_state_value(state, "messages")
        structured_response_schema = response_format
        if isinstance(response_format, tuple):
            system_prompt, structured_response_schema = response_format
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        model_with_structured_output = _get_model(model).with_structured_output(
            cast(StructuredResponseSchema, structured_response_schema)
        )
        response = await model_with_structured_output.ainvoke(messages, config)
        return {"structured_response": response}
    
    if not tool_calling_enabled:
        # Define a new graph
        workflow = StateGraph(state_schema, config_schema=config_schema)
        workflow.add_node(
            "agent",
            RunnableCallable(call_model, acall_model),
            input=input_schema
        )
        if pre_model_hook is not None:
            workflow.add_node("pre_model_hook", pre_model_hook)
            workflow.add_edge("pre_model_hook", "agent")
            entrypoint = "pre_model_hook"
        else:
            entrypoint = "agent"

        workflow.set_entry_point(entrypoint)

        if post_model_hook is not None:
            workflow.add_node("post_model_hook", post_model_hook)
            workflow.add_edge("agent", "post_model_hook")
    
        if response_format is not None:
            workflow.add_node(
                "generate_structured_response",
                RunnableCallable(
                    generate_structured_response, agenerate_structured_response
                ),
            )
            if post_model_hook is not None:
                workflow.add_edge("post_model_hook", "generate_structured_response")
            else:
                workflow.add_edge("agent", "generate_structured_response")

        return workflow.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
            name=name,
        )
    
    # Define the function that determines whether to continue or not


    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
    )