"""
Math Benchmark Agents: CoT, ReAct, and Reflexion
Functions for creating different types of math problem-solving agents.
"""
from typing import Dict, Any, Optional, Union
import yaml
import os
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="/Users/admin/Desktop/workspace/my_github/langgraph_service/.env")

# Math tools
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract second number from first number."""
    return a - b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide first number by second number."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@tool
def calculate_power(base: float, exponent: float) -> float:
    """Calculate base raised to the power of exponent."""
    return base ** exponent

@tool
def calculate_sqrt(number: float) -> float:
    """Calculate square root of a number."""
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return number ** 0.5

@tool
def calculate_factorial(n: int) -> int:
    """Calculate factorial of a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# All available math tools
MATH_TOOLS = [
    add_numbers,
    subtract_numbers, 
    multiply_numbers,
    divide_numbers,
    calculate_power,
    calculate_sqrt,
    calculate_factorial
]

def load_math_prompts() -> Dict[str, Any]:
    """Load math prompts from YAML file."""
    prompt_file = "/Users/admin/Desktop/workspace/my_github/langgraph_service/graphs/agent/prompts/math_prompt.yaml"
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

def create_cot_math_agent(
    model: str = "openai:gpt-4o-mini",
    benchmark: str = "gsm8k",
    tools: Optional[list] = None,
    **kwargs
):
    """
    Create a Chain of Thought (CoT) math agent.
    
    Args:
        model: The language model to use
        benchmark: Math benchmark type (gsm8k, basic, competition, aime)
        tools: List of tools to use (defaults to all math tools)
        **kwargs: Additional arguments for create_react_agent
    
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = MATH_TOOLS
    
    # Load prompts
    prompts_data = load_math_prompts()
    
    # Get CoT prompt for the benchmark
    cot_prompt = ""
    if prompts_data and "prompts" in prompts_data:
        benchmark_prompts = prompts_data["prompts"].get(benchmark, {})
        cot_config = benchmark_prompts.get("cot", {})
        cot_prompt = cot_config.get("template", "")
    
    # Fallback prompt if not found in YAML
    if not cot_prompt:
        cot_prompt = """You are solving math problems using Chain of Thought reasoning.

Follow this step-by-step thinking process:

**Step 1: Problem Understanding**
Let me carefully read and understand this problem...

**Step 2: Information Extraction** 
From the problem, I can identify these key pieces:
- Given: [List all numbers and facts provided]
- Find: [What we need to calculate]

**Step 3: Solution Planning**
To solve this problem, I need to think through the logical steps:
[Plan the sequence of operations needed]

**Step 4: Step-by-Step Solution**
Now I'll work through each step, using calculation tools:
[Use tools for each calculation step with clear reasoning]

**Step 5: Answer Verification**
Let me check if this answer makes sense:
[Verify the reasonableness of the result]

**Final Answer:** [Clear numerical result]"""

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=cot_prompt,
        **kwargs
    )

def create_react_math_agent(
    model: str = "openai:gpt-4o-mini", 
    benchmark: str = "gsm8k",
    tools: Optional[list] = None,
    **kwargs
):
    """
    Create a ReAct (Reasoning and Acting) math agent.
    
    Args:
        model: The language model to use
        benchmark: Math benchmark type (gsm8k, basic, competition, aime)
        tools: List of tools to use (defaults to all math tools)
        **kwargs: Additional arguments for create_react_agent
    
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = MATH_TOOLS
    
    # Load prompts
    prompts_data = load_math_prompts()
    
    # Get ReAct prompt for the benchmark
    react_prompt = ""
    if prompts_data and "prompts" in prompts_data:
        benchmark_prompts = prompts_data["prompts"].get(benchmark, {})
        react_config = benchmark_prompts.get("react", {})
        react_prompt = react_config.get("template", "")
    
    # Fallback prompt if not found in YAML
    if not react_prompt:
        react_prompt = """You are solving math problems using ReAct methodology.

**Thought 1**: I need to understand what this math problem is asking me to find.

**Action 1**: [ANALYZE_PROBLEM]
[Carefully read and break down the problem]

**Observation 1**: [What information you extracted from the problem]

**Thought 2**: I should identify all the given information and what I need to calculate.

**Action 2**: [EXTRACT_KEY_INFORMATION] 
Given: [List all provided numbers and facts]
Find: [What needs to be calculated]

**Observation 2**: [Assessment of the information available]

**Thought 3**: I need to plan the sequence of calculations required.

**Action 3**: [PLAN_SOLUTION_STEPS]
[Outline the mathematical operations needed]

**Observation 3**: [Evaluation of the solution plan]

**Thought 4**: Now I'll execute the calculations step by step using tools.

**Action 4**: [EXECUTE_CALCULATIONS]
[Perform calculations using math tools with explanations]

**Observation 4**: [Review of calculation results]

**Thought 5**: I should verify that my answer makes sense in the context.

**Action 5**: [VERIFY_ANSWER]
[Check reasonableness of the result]

**Observation 5**: [Final verification and answer confirmation]"""

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=react_prompt,
        **kwargs
    )

def create_reflexion_math_agent(
    model: str = "openai:gpt-4o-mini",
    benchmark: str = "gsm8k", 
    tools: Optional[list] = None,
    **kwargs
):
    """
    Create a Reflexion (Self-reflective) math agent.
    
    Args:
        model: The language model to use
        benchmark: Math benchmark type (gsm8k, basic, competition, aime)
        tools: List of tools to use (defaults to all math tools)
        **kwargs: Additional arguments for create_react_agent
    
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = MATH_TOOLS
    
    # Load prompts
    prompts_data = load_math_prompts()
    
    # Get Reflexion prompt for the benchmark
    reflexion_prompt = ""
    if prompts_data and "prompts" in prompts_data:
        benchmark_prompts = prompts_data["prompts"].get(benchmark, {})
        reflexion_config = benchmark_prompts.get("reflexion", {})
        reflexion_prompt = reflexion_config.get("template", "")
    
    # Fallback prompt if not found in YAML
    if not reflexion_prompt:
        reflexion_prompt = """You are solving math problems using Reflexion methodology.

**Initial Problem Attempt**:
Let me first try to solve this problem:

Given: [Initial understanding of given information]
Approach: [Initial solution approach]

[Initial calculation steps using tools]

Initial Answer: [First attempt result]

**Self-Reflection**:
Let me critically examine my initial solution:
- Did I correctly identify all the given information?
- Are my calculation steps logical and in the right order?
- Did I use the right mathematical operations?
- Does my answer make sense in the real-world context?
- Are there any arithmetic errors in my calculations?

**Issues Identified**:
[Any problems found in the initial attempt]

**Improved Approach**:
Based on my reflection, here's a better way to solve this:
[Refined understanding and approach]

**Refined Solution**:
[Step-by-step improved solution using calculation tools]

**Final Verification**:
[Check the improved solution for correctness and reasonableness]

**Learning Summary**:
Through this reflexive process, I learned:
[Key insights gained from self-reflection]"""

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=reflexion_prompt,
        **kwargs
    )

def create_math_agent(
    agent_type: str,
    model: str = "openai:gpt-4o-mini",
    benchmark: str = "gsm8k",
    tools: Optional[list] = None,
    **kwargs
):
    """
    Create a math agent of the specified type.
    
    Args:
        agent_type: Type of agent ("cot", "react", "reflexion")
        model: The language model to use
        benchmark: Math benchmark type (gsm8k, basic, competition, aime)
        tools: List of tools to use (defaults to all math tools)
        **kwargs: Additional arguments for create_react_agent
    
    Returns:
        Compiled LangGraph agent
    
    Raises:
        ValueError: If agent_type is not supported
    """
    agent_type = agent_type.lower()
    
    if agent_type == "cot":
        return create_cot_math_agent(model, benchmark, tools, **kwargs)
    elif agent_type == "react":
        return create_react_math_agent(model, benchmark, tools, **kwargs)
    elif agent_type == "reflexion":
        return create_reflexion_math_agent(model, benchmark, tools, **kwargs)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}. Use 'cot', 'react', or 'reflexion'")

# Test functions for each agent type
def test_agent(agent, problem: str, agent_type: str = ""):
    """Test an agent with a math problem."""
    print(f"\n=== Testing {agent_type} Agent ===")
    print(f"Problem: {problem}")
    print("\n--- Agent Response ---")
    
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": problem}]})
        print(result["messages"][-1].content)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    # Example usage
    test_problem = "Sarah has 15 apples. She gives 3 apples to her friend and then buys 8 more apples. How many apples does Sarah have now?"
    
    # Create different types of agents
    cot_agent = create_cot_math_agent(benchmark="gsm8k")
    react_agent = create_react_math_agent(benchmark="gsm8k") 
    reflexion_agent = create_reflexion_math_agent(benchmark="gsm8k")
    
    # Test each agent
    test_agent(cot_agent, test_problem, "CoT")
    test_agent(react_agent, test_problem, "ReAct")
    test_agent(reflexion_agent, test_problem, "Reflexion")
