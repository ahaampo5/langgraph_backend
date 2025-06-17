#!/usr/bin/env python3
"""
Quick demo of math evaluation system
Tests with a few problems from the hendrycks_math dataset
"""
import sys
import os
import time

# Add the experiments directory to the path
sys.path.insert(0, "/Users/admin/Desktop/workspace/my_github/langgraph_service/graphs/agent/experiments")

def main():
    print("🧮 Math Evaluation Demo")
    print("=" * 50)
    
    try:
        # Import required modules
        from math_agents import create_cot_math_agent, create_react_math_agent, create_reflexion_math_agent
        from datasets import load_dataset
        
        print("✅ Imports successful")
        
        # Load a small dataset for testing
        print("\n📚 Loading algebra dataset...")
        dataset = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
        print(f"✅ Loaded dataset")
        
        # Create agents
        print("\n🤖 Creating agents...")
        cot_agent = create_cot_math_agent(benchmark="competition")
        react_agent = create_react_math_agent(benchmark="competition")
        reflexion_agent = create_reflexion_math_agent(benchmark="competition")
        print("✅ All agents created")
        
        # Test with a few problems
        print("\n🧪 Testing with sample problems...")
        
        # Get first 2 problems for quick demo
        test_problems = []
        for i in range(min(2, len(dataset) if hasattr(dataset, '__len__') else 2)):
            try:
                item = dataset[i]
                test_problems.append({
                    'problem': item['problem'],
                    'solution': item['solution'],
                    'level': item['level'],
                    'type': item['type']
                })
            except:
                break
        
        if not test_problems:
            print("❌ Could not load test problems")
            return
        
        agents = {
            'CoT': cot_agent,
            'ReAct': react_agent, 
            'Reflexion': reflexion_agent
        }
        
        for i, problem_data in enumerate(test_problems):
            print(f"\n{'='*60}")
            print(f"Problem {i+1} (Level {problem_data['level']})")
            print(f"{'='*60}")
            print(f"Problem: {problem_data['problem']}")
            print(f"Expected: {problem_data['solution'][:100]}...")
            print()
            
            for agent_name, agent in agents.items():
                print(f"🤖 Testing {agent_name} Agent...")
                start_time = time.time()
                
                try:
                    result = agent.invoke({
                        "messages": [{"role": "user", "content": problem_data['problem']}]
                    })
                    answer = result["messages"][-1].content
                    execution_time = time.time() - start_time
                    
                    print(f"✅ {agent_name} Answer ({execution_time:.1f}s):")
                    print(f"   {answer[:150]}...")
                    
                except Exception as e:
                    print(f"❌ {agent_name} Error: {e}")
                
                print()
        
        print("🎉 Demo completed!")
        print("\nTo run full evaluation:")
        print("python evaluate.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure all dependencies are installed:")
        print("- pip install datasets")
        print("- pip install transformers")

if __name__ == "__main__":
    main()
