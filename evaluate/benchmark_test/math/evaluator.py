"""
Math Benchmark Evaluation using EleutherAI/hendrycks_math dataset
Evaluates CoT, ReAct, and Reflexion agents across different math categories.
"""
import sys
import os
import time
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datasets import load_dataset
import pandas as pd

# Add the experiments directory to the path
experiments_dir = "/Users/admin/Desktop/workspace/my_github/langgraph_service/graphs/agent/experiments"
sys.path.insert(0, experiments_dir)

try:
    from math_agents import (
        create_cot_math_agent,
        create_react_math_agent,
        create_reflexion_math_agent
    )
except ImportError as e:
    print(f"Warning: Could not import math_agents: {e}")
    print("Make sure you're running from the correct directory")

@dataclass
class EvaluationResult:
    """Result of evaluating an agent on a problem."""
    category: str
    level: int
    problem: str
    solution: str
    agent_type: str
    agent_answer: str
    is_correct: bool
    execution_time: float
    error: Optional[str] = None

class MathBenchmarkEvaluator:
    """Evaluator for math agents using EleutherAI/hendrycks_math dataset."""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        self.model = model
        self.agents = {}
        self.datasets = {}
        self.categories = [
            'algebra', 
            'counting_and_probability', 
            'geometry', 
            'intermediate_algebra', 
            'number_theory', 
            'prealgebra', 
            'precalculus'
        ]
        self._load_datasets()
        self._initialize_agents()
    
    def _load_datasets(self):
        """Load all math datasets."""
        print("📚 Loading EleutherAI/hendrycks_math datasets...")
        
        for category in self.categories:
            try:
                print(f"  Loading {category}...")
                dataset = load_dataset("EleutherAI/hendrycks_math", category, split="test")
                self.datasets[category] = dataset
                # Handle different dataset types
                if hasattr(dataset, '__len__'):
                    dataset_size = len(dataset)
                else:
                    dataset_size = "unknown"
                print(f"  ✅ {category}: {dataset_size} problems")
            except Exception as e:
                print(f"  ❌ Failed to load {category}: {e}")
        
        print(f"✅ Loaded {len(self.datasets)} categories")
    
    def _initialize_agents(self):
        """Initialize all agent types."""
        print("🤖 Initializing math agents...")
        
        agent_types = ["cot", "react", "reflexion"]
        
        for agent_type in agent_types:
            try:
                if agent_type == "cot":
                    agent = create_cot_math_agent(
                        model=self.model,
                        benchmark="competition"  # Use competition level for math dataset
                    )
                elif agent_type == "react":
                    agent = create_react_math_agent(
                        model=self.model,
                        benchmark="competition"
                    )
                elif agent_type == "reflexion":
                    agent = create_reflexion_math_agent(
                        model=self.model,
                        benchmark="competition"
                    )
                
                self.agents[agent_type] = agent
                print(f"  ✅ {agent_type.upper()} agent initialized")
                
            except Exception as e:
                print(f"  ❌ Failed to initialize {agent_type} agent: {e}")
        
        print(f"✅ Initialized {len(self.agents)} agents")
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract the final numerical answer from agent response."""
        # Look for patterns like "Final Answer: X" or numbers at the end
        patterns = [
            r"Final Answer[:\s]*([^\n]+)",
            r"答[:\s]*([^\n]+)",
            r"정답[:\s]*([^\n]+)",
            r"Answer[:\s]*([^\n]+)",
            r"결과[:\s]*([^\n]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern found, try to extract the last number or expression
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]
        
        # Return last non-empty line as fallback
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return lines[-1] if lines else ""
    
    def _check_correctness(self, agent_answer: str, solution: str) -> bool:
        """Check if the agent's answer matches the solution."""
        try:
            # Extract final answers from both
            agent_final = self._extract_final_answer(agent_answer)
            solution_final = self._extract_final_answer(solution)
            
            # Try numerical comparison first
            try:
                agent_num = float(agent_final.replace('$', '').replace(',', ''))
                solution_num = float(solution_final.replace('$', '').replace(',', ''))
                return abs(agent_num - solution_num) < 1e-6
            except:
                pass
            
            # Fallback to string comparison
            agent_clean = re.sub(r'[^\w\d\.\-]', '', agent_final.lower())
            solution_clean = re.sub(r'[^\w\d\.\-]', '', solution_final.lower())
            
            return agent_clean == solution_clean or agent_clean in solution_clean
            
        except Exception as e:
            print(f"Error in correctness check: {e}")
            return False
    
    def evaluate_single_problem(
        self, 
        problem: str, 
        solution: str,
        category: str,
        level: int,
        agent_type: str
    ) -> EvaluationResult:
        """Evaluate a single problem with a specific agent."""
        start_time = time.time()
        
        try:
            agent = self.agents[agent_type]
            result = agent.invoke({"messages": [{"role": "user", "content": problem}]})
            agent_answer = result["messages"][-1].content
            execution_time = time.time() - start_time
            
            # Check correctness
            is_correct = self._check_correctness(agent_answer, solution)
            
            return EvaluationResult(
                category=category,
                level=level,
                problem=problem,
                solution=solution,
                agent_type=agent_type,
                agent_answer=agent_answer,
                is_correct=is_correct,
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return EvaluationResult(
                category=category,
                level=level,
                problem=problem,
                solution=solution,
                agent_type=agent_type,
                agent_answer="",
                is_correct=False,
                execution_time=execution_time,
                error=str(e)
            )
    
    def evaluate_category(
        self, 
        category: str, 
        max_problems: int = 50,
        agent_types: Optional[List[str]] = None
    ) -> List[EvaluationResult]:
        """Evaluate a specific category with all agents."""
        if agent_types is None:
            agent_types = ["cot", "react", "reflexion"]
        
        if category not in self.datasets:
            print(f"❌ Category {category} not found")
            return []
        
        dataset = self.datasets[category]
        results = []
        
        # Limit the number of problems for efficiency
        num_problems = min(len(dataset), max_problems)
        
        print(f"\n📊 Evaluating {category} ({num_problems} problems)")
        print("=" * 60)
        
        for i in range(num_problems):
            item = dataset[i]
            problem = item['problem']
            solution = item['solution']
            level = item['level']
            
            print(f"\nProblem {i+1}/{num_problems} (Level {level})")
            print(f"Problem: {problem[:100]}...")
            
            for agent_type in agent_types:
                print(f"  Testing {agent_type} agent...", end=" ")
                
                result = self.evaluate_single_problem(
                    problem, solution, category, level, agent_type
                )
                results.append(result)
                
                status = "✅" if result.is_correct else "❌"
                print(f"{status} ({result.execution_time:.1f}s)")
        
        return results
    
    def evaluate_all_categories(
        self, 
        max_problems_per_category: int = 20,
        agent_types: Optional[List[str]] = None
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluate all categories."""
        if agent_types is None:
            agent_types = ["cot", "react", "reflexion"]
        
        all_results = {}
        
        print(f"\n🚀 Starting evaluation of all categories")
        print(f"Max problems per category: {max_problems_per_category}")
        print(f"Agent types: {', '.join(agent_types)}")
        print("=" * 80)
        
        for category in self.categories:
            if category in self.datasets:
                results = self.evaluate_category(
                    category, 
                    max_problems_per_category, 
                    agent_types
                )
                all_results[category] = results
        
        return all_results
    
    def generate_report(
        self, 
        results: Dict[str, List[EvaluationResult]]
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            "overall": {},
            "by_category": {},
            "by_agent": {},
            "by_level": {}
        }
        
        # Flatten all results
        all_results = []
        for category_results in results.values():
            all_results.extend(category_results)
        
        # Overall statistics
        agent_types = list(set(r.agent_type for r in all_results))
        
        for agent_type in agent_types:
            agent_results = [r for r in all_results if r.agent_type == agent_type]
            
            total = len(agent_results)
            correct = sum(1 for r in agent_results if r.is_correct)
            avg_time = sum(r.execution_time for r in agent_results) / total if total > 0 else 0
            errors = sum(1 for r in agent_results if r.error)
            
            report["overall"][agent_type] = {
                "accuracy": correct / total if total > 0 else 0,
                "correct": correct,
                "total": total,
                "avg_time": avg_time,
                "errors": errors
            }
        
        # By category
        for category, category_results in results.items():
            report["by_category"][category] = {}
            
            for agent_type in agent_types:
                agent_results = [r for r in category_results if r.agent_type == agent_type]
                
                if agent_results:
                    total = len(agent_results)
                    correct = sum(1 for r in agent_results if r.is_correct)
                    
                    report["by_category"][category][agent_type] = {
                        "accuracy": correct / total,
                        "correct": correct,
                        "total": total
                    }
        
        # By level
        levels = list(set(r.level for r in all_results))
        for level in levels:
            report["by_level"][level] = {}
            
            for agent_type in agent_types:
                agent_results = [r for r in all_results 
                               if r.agent_type == agent_type and r.level == level]
                
                if agent_results:
                    total = len(agent_results)
                    correct = sum(1 for r in agent_results if r.is_correct)
                    
                    report["by_level"][level][agent_type] = {
                        "accuracy": correct / total,
                        "correct": correct,
                        "total": total
                    }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted evaluation report."""
        print("\n" + "="*80)
        print("📊 MATH BENCHMARK EVALUATION REPORT")
        print("="*80)
        
        # Overall results
        print("\n🎯 OVERALL RESULTS")
        print("-" * 40)
        for agent_type, metrics in report["overall"].items():
            print(f"{agent_type.upper()} Agent:")
            print(f"  📈 Accuracy: {metrics['accuracy']:.2%}")
            print(f"  ✅ Correct: {metrics['correct']}/{metrics['total']}")
            print(f"  ⏱️  Avg Time: {metrics['avg_time']:.2f}s")
            print(f"  ❌ Errors: {metrics['errors']}")
            print()
        
        # Best performer
        best_agent = max(report["overall"].keys(), 
                        key=lambda x: report["overall"][x]['accuracy'])
        print(f"🏆 Best Overall: {best_agent.upper()} Agent "
              f"({report['overall'][best_agent]['accuracy']:.2%})")
        
        # By category
        print(f"\n📚 RESULTS BY CATEGORY")
        print("-" * 40)
        for category, category_results in report["by_category"].items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for agent_type, metrics in category_results.items():
                print(f"  {agent_type.upper()}: {metrics['accuracy']:.2%} "
                      f"({metrics['correct']}/{metrics['total']})")
        
        # By level
        print(f"\n📊 RESULTS BY DIFFICULTY LEVEL")
        print("-" * 40)
        for level in sorted(report["by_level"].keys()):
            print(f"\nLevel {level}:")
            level_results = report["by_level"][level]
            for agent_type, metrics in level_results.items():
                print(f"  {agent_type.upper()}: {metrics['accuracy']:.2%} "
                      f"({metrics['correct']}/{metrics['total']})")
        
        print("="*80)
    
    def save_results(
        self, 
        results: Dict[str, List[EvaluationResult]], 
        filename: str = "math_evaluation_results.json"
    ):
        """Save evaluation results to JSON file."""
        serializable_results = {}
        
        for category, category_results in results.items():
            serializable_results[category] = []
            for result in category_results:
                serializable_results[category].append({
                    "category": result.category,
                    "level": result.level,
                    "problem": result.problem,
                    "solution": result.solution,
                    "agent_type": result.agent_type,
                    "agent_answer": result.agent_answer,
                    "is_correct": result.is_correct,
                    "execution_time": result.execution_time,
                    "error": result.error
                })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to {filename}")

def main():
    """Main evaluation function."""
    print("🧮 Math Benchmark Evaluation")
    print("Dataset: EleutherAI/hendrycks_math")
    print("Agents: CoT, ReAct, Reflexion")
    print("="*80)
    
    # Initialize evaluator
    evaluator = MathBenchmarkEvaluator()
    
    # Run evaluation (start with small number for testing)
    results = evaluator.evaluate_all_categories(
        max_problems_per_category=10,  # Start small for testing
        agent_types=["cot", "react", "reflexion"]
    )
    
    # Generate and print report
    report = evaluator.generate_report(results)
    evaluator.print_report(report)
    
    # Save results
    evaluator.save_results(results)
    
    print("\n✅ Evaluation completed!")

if __name__ == "__main__":
    main()