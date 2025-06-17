"""
Math Benchmark Agents Package

This package provides math problem-solving agents using different reasoning methodologies:
- CoT (Chain of Thought): Step-by-step reasoning
- ReAct (Reasoning and Acting): Thought-Action-Observation cycles  
- Reflexion: Self-reflective problem solving

Usage:
    from graphs.agent.experiments import (
        create_cot_math_agent,
        create_react_math_agent,
        create_reflexion_math_agent,
        create_math_agent,
        MathAgentRunner,
        MathAgentEvaluator
    )
    
    # Create agents
    cot_agent = create_cot_math_agent(benchmark="gsm8k")
    react_agent = create_react_math_agent(benchmark="gsm8k")
    reflexion_agent = create_reflexion_math_agent(benchmark="gsm8k")
    
    # Or use the unified function
    agent = create_math_agent("cot", benchmark="gsm8k")
    
    # Use the runner for interactive testing
    runner = MathAgentRunner(agent_type="react", benchmark="gsm8k")
    runner.run_interactive()
"""

from .math_agents import (
    create_cot_math_agent,
    create_react_math_agent,
    create_reflexion_math_agent,
    create_math_agent,
    MATH_TOOLS,
    load_math_prompts,
)

# from .math_agent import MathAgentRunner  # 주석 처리 - 존재하지 않음

from .legacy.math_agent_evaluator import (
    MathAgentEvaluator,
    EvaluationResult,
)

__all__ = [
    # Agent creation functions
    "create_cot_math_agent",
    "create_react_math_agent", 
    "create_reflexion_math_agent",
    "create_math_agent",
    
    # Tools and utilities
    "MATH_TOOLS",
    "load_math_prompts",
    
    # Classes
    # "MathAgentRunner",  # 주석 처리 - 존재하지 않음
    "MathAgentEvaluator", 
    "EvaluationResult",
]

# Version info
__version__ = "1.0.0"
__author__ = "LangGraph Math Agents Team"
