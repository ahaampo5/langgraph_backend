#!/usr/bin/env python3
"""
Simple test of math evaluation system
"""
import sys
import os

# Add paths
sys.path.insert(0, "/Users/admin/Desktop/workspace/my_github/langgraph_service/graphs/agent/experiments")

def test_imports():
    """Test if all imports work correctly."""
    print("🔍 Testing imports...")
    
    try:
        from math_agents import create_cot_math_agent, create_react_math_agent, create_reflexion_math_agent
        print("✅ Math agents imported successfully")
    except Exception as e:
        print(f"❌ Failed to import math agents: {e}")
        return False
    
    try:
        from datasets import load_dataset
        print("✅ Datasets library imported successfully")
    except Exception as e:
        print(f"❌ Failed to import datasets: {e}")
        return False
    
    return True

def test_dataset_loading():
    """Test loading a small sample of the math dataset."""
    print("\n📚 Testing dataset loading...")
    
    try:
        from datasets import load_dataset
        
        # Load just one category for testing
        dataset = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
        print(f"✅ Loaded algebra dataset")
        
        # Check first item
        if len(dataset) > 0:
            first_item = dataset[0]
            print(f"✅ First problem loaded: {first_item['problem'][:50]}...")
            print(f"   Level: {first_item['level']}")
            print(f"   Type: {first_item['type']}")
            return True
        else:
            print("❌ Dataset is empty")
            return False
            
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False

def test_agent_creation():
    """Test creating the math agents."""
    print("\n🤖 Testing agent creation...")
    
    try:
        from math_agents import create_cot_math_agent, create_react_math_agent, create_reflexion_math_agent
        
        # Create one of each type
        cot_agent = create_cot_math_agent(benchmark="competition")
        print("✅ CoT agent created")
        
        react_agent = create_react_math_agent(benchmark="competition")
        print("✅ ReAct agent created")
        
        reflexion_agent = create_reflexion_math_agent(benchmark="competition")
        print("✅ Reflexion agent created")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create agents: {e}")
        return False

def test_simple_evaluation():
    """Test a simple evaluation with one problem."""
    print("\n🧪 Testing simple evaluation...")
    
    try:
        from math_agents import create_cot_math_agent
        
        # Create agent
        agent = create_cot_math_agent(benchmark="competition")
        
        # Simple test problem
        problem = "What is 2 + 3?"
        
        result = agent.invoke({"messages": [{"role": "user", "content": problem}]})
        answer = result["messages"][-1].content
        
        print(f"✅ Problem: {problem}")
        print(f"✅ Answer: {answer[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed evaluation test: {e}")
        return False

def main():
    """Run all tests."""
    print("🧮 Math Evaluation System Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_dataset_loading,
        test_agent_creation,
        test_simple_evaluation
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✅ All tests passed! Evaluation system is ready.")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
