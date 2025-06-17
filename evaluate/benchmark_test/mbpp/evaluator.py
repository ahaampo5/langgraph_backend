"""
MBPP (Mostly Basic Python Programming) 벤치마크 평가기
Hugging Face datasets의 MBPP 데이터셋을 사용하여 CoT, ReAct, Reflexion 에이전트를 평가
"""

import os
import sys
import json
import time
import asyncio
import tempfile
import subprocess
from typing import List, Dict, Any, Optional, Tuple, Literal, cast
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

# Hugging Face datasets
from datasets import load_dataset, Dataset

# LangGraph and LangChain
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# 프로젝트 imports - 상대 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from graphs.agent.experiments.code_benchmark_agents import (
    create_benchmark_agent,
    create_cot_agent,
    create_react_reasoning_agent,
    create_reflexion_agent
)


@dataclass
class MBPPProblem:
    """MBPP 문제 데이터 구조"""
    task_id: str
    text: str
    code: str
    test_imports: List[str]
    test_list: List[str]
    challenge_test_list: Optional[List[str]] = None


@dataclass
class EvaluationResult:
    """평가 결과 데이터 구조"""
    task_id: str
    reasoning_type: str
    problem_text: str
    generated_solution: str
    execution_time: float
    success: bool
    test_passed: bool
    test_results: Dict[str, Any]
    error_message: Optional[str] = None


class MBPPEvaluator:
    """MBPP 벤치마크 평가 클래스"""
    
    def __init__(self, model: str = "gpt-4o-mini", max_problems: int = 10):
        self.model = model
        self.max_problems = max_problems
        self.results: List[EvaluationResult] = []
        
        # 데이터셋 로드
        print("MBPP 데이터셋 로드 중...")
        dataset_raw = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        self.dataset = cast(Dataset, dataset_raw)
        try:
            dataset_len = len(self.dataset)
            print(f"총 {dataset_len}개 문제 로드됨")
        except:
            print("데이터셋 로드됨 (크기 미확인)")
        
        # 문제 변환
        self.problems = self._convert_dataset_to_problems()[:max_problems]
        print(f"평가할 문제 수: {len(self.problems)}개")
    
    def _convert_dataset_to_problems(self) -> List[MBPPProblem]:
        """데이터셋을 MBPPProblem 객체로 변환"""
        problems = []
        
        for item in self.dataset:
            # 타입 안전성을 위해 cast 사용
            item_dict = cast(Dict[str, Any], item)
            problem = MBPPProblem(
                task_id=str(item_dict.get('task_id', '')),
                text=str(item_dict.get('text', '')),
                code=str(item_dict.get('code', '')),
                test_imports=item_dict.get('test_imports', []),
                test_list=item_dict.get('test_list', []),
                challenge_test_list=item_dict.get('challenge_test_list', [])
            )
            problems.append(problem)
        
        return problems
    
    def _create_problem_prompt(self, problem: MBPPProblem) -> str:
        """문제를 에이전트에게 전달할 프롬프트로 변환"""
        prompt = f"""다음 Python 프로그래밍 문제를 해결해주세요:

문제: {problem.text}

"""
        
        if problem.test_imports:
            prompt += f"필요한 import문들:\n"
            for imp in problem.test_imports:
                prompt += f"- {imp}\n"
            prompt += "\n"
        
        if problem.test_list:
            prompt += f"테스트 케이스들:\n"
            for i, test in enumerate(problem.test_list, 1):
                prompt += f"{i}. {test}\n"
            prompt += "\n"
        
        prompt += """완전한 함수를 작성해주세요. 함수명과 매개변수는 테스트 케이스에서 유추할 수 있습니다.
모든 테스트 케이스가 통과하도록 구현해주세요."""
        
        return prompt
    
    def _execute_code_with_tests(self, problem: MBPPProblem, solution_code: str) -> Tuple[bool, Dict[str, Any]]:
        """생성된 코드와 테스트를 실행하여 결과를 반환"""
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # import문들 추가
                for imp in problem.test_imports:
                    f.write(f"{imp}\n")
                
                f.write("\n")
                f.write(solution_code)
                f.write("\n\n")
                
                # 테스트 케이스들 추가
                f.write("# Test cases\n")
                test_count = 0
                passed_count = 0
                
                for i, test in enumerate(problem.test_list):
                    test_count += 1
                    f.write(f"try:\n")
                    f.write(f"    {test}\n")
                    f.write(f"    print(f'Test {i+1} PASSED: {test}')\n")
                    f.write(f"except Exception as e:\n")
                    f.write(f"    print(f'Test {i+1} FAILED: {test} - Error: {{e}}')\n")
                f.write(f"\nprint(f'Tests completed')\n")
                
                temp_file = f.name
            
            # 코드 실행
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # 임시 파일 삭제
            os.unlink(temp_file)
            
            # 결과 분석
            output = result.stdout
            stderr = result.stderr
            
            # PASSED 개수 세기
            passed_count = output.count("PASSED")
            failed_count = output.count("FAILED")
            
            success = result.returncode == 0 and failed_count == 0
            
            test_results = {
                "stdout": output,
                "stderr": stderr,
                "return_code": result.returncode,
                "total_tests": len(problem.test_list),
                "passed_tests": passed_count,
                "failed_tests": failed_count,
                "success_rate": passed_count / len(problem.test_list) if problem.test_list else 0
            }
            
            return success, test_results
            
        except subprocess.TimeoutExpired:
            return False, {"error": "Execution timeout (30s)"}
        except Exception as e:
            return False, {"error": f"Execution error: {str(e)}"}
    
    def _extract_function_code(self, solution_text: str) -> str:
        """에이전트의 응답에서 함수 코드를 추출"""
        lines = solution_text.split('\n')
        code_lines = []
        in_code_block = False
        found_function = False
        
        for line in lines:
            # 코드 블록 시작/끝 감지
            if line.strip().startswith('```python') or line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            
            # 함수 정의 시작 감지
            if line.strip().startswith('def '):
                found_function = True
                in_code_block = True
            
            # 코드 블록 내부이거나 함수를 찾은 경우
            if in_code_block or found_function:
                code_lines.append(line)
                
                # 함수 끝 감지 (빈 줄 또는 들여쓰기 없는 줄)
                if found_function and not in_code_block and line.strip() == '':
                    break
        
        # 추출된 코드가 없으면 전체 텍스트에서 def로 시작하는 부분 찾기
        if not code_lines:
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    # 함수 정의부터 끝까지 추출
                    remaining_lines = lines[i:]
                    for j, remaining_line in enumerate(remaining_lines):
                        code_lines.append(remaining_line)
                        # 다음 함수나 클래스 정의가 나오면 중단
                        if j > 0 and (remaining_line.strip().startswith('def ') or 
                                     remaining_line.strip().startswith('class ')):
                            code_lines = code_lines[:-1]
                            break
                    break
        
        return '\n'.join(code_lines)
    
    async def evaluate_single_problem(
        self,
        problem: MBPPProblem,
        reasoning_type: Literal["cot", "react", "reflexion"],
        timeout: int = 120
    ) -> EvaluationResult:
        """단일 문제에 대한 평가 수행"""
        
        start_time = time.time()
        
        try:
            print(f"문제 {problem.task_id} ({reasoning_type}) 평가 중...")
            
            # 에이전트 생성
            memory = MemorySaver()
            agent = create_benchmark_agent(
                reasoning_type=reasoning_type,
                model=self.model,
                benchmark_type="mbpp",
                checkpointer=memory
            )
            
            # 문제 프롬프트 생성
            prompt = self._create_problem_prompt(problem)
            
            # 에이전트 실행
            config = {"configurable": {"thread_id": f"mbpp_{problem.task_id}_{reasoning_type}"}}
            
            response = await asyncio.wait_for(
                agent.ainvoke({
                    "messages": [HumanMessage(content=prompt)]
                }, config=config),
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            solution_text = response["messages"][-1].content
            
            # 함수 코드 추출
            solution_code = self._extract_function_code(solution_text)
            
            # 코드 실행 및 테스트
            test_passed, test_results = self._execute_code_with_tests(problem, solution_code)
            
            return EvaluationResult(
                task_id=problem.task_id,
                reasoning_type=reasoning_type,
                problem_text=problem.text,
                generated_solution=solution_code,
                execution_time=execution_time,
                success=True,
                test_passed=test_passed,
                test_results=test_results
            )
            
        except asyncio.TimeoutError:
            return EvaluationResult(
                task_id=problem.task_id,
                reasoning_type=reasoning_type,
                problem_text=problem.text,
                generated_solution="",
                execution_time=timeout,
                success=False,
                test_passed=False,
                test_results={},
                error_message="Agent timeout"
            )
        except Exception as e:
            return EvaluationResult(
                task_id=problem.task_id,
                reasoning_type=reasoning_type,
                problem_text=problem.text,
                generated_solution="",
                execution_time=time.time() - start_time,
                success=False,
                test_passed=False,
                test_results={},
                error_message=str(e)
            )
    
    async def evaluate_all_agents(self, max_concurrent: int = 2) -> List[EvaluationResult]:
        """모든 에이전트에 대해 전체 평가 수행"""
        
        reasoning_types: List[Literal["cot", "react", "reflexion"]] = ["cot", "react", "reflexion"]
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(problem, reasoning_type):
            async with semaphore:
                return await self.evaluate_single_problem(problem, reasoning_type)
        
        print(f"\n총 {len(self.problems)}개 문제 × {len(reasoning_types)}개 에이전트 = {len(self.problems) * len(reasoning_types)}개 평가 시작")
        
        tasks = []
        for problem in self.problems:
            for reasoning_type in reasoning_types:
                task = evaluate_with_semaphore(problem, reasoning_type)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 처리
        valid_results = []
        for result in results:
            if isinstance(result, EvaluationResult):
                valid_results.append(result)
                self.results.append(result)
            else:
                print(f"평가 중 오류 발생: {result}")
        
        return valid_results
    
    def save_results(self, filepath: str = "mbpp_evaluation_results.csv"):
        """결과를 CSV 파일로 저장"""
        results_data = []
        
        for result in self.results:
            results_data.append({
                'task_id': result.task_id,
                'reasoning_type': result.reasoning_type,
                'problem_text': result.problem_text[:100] + "..." if len(result.problem_text) > 100 else result.problem_text,
                'execution_time': result.execution_time,
                'success': result.success,
                'test_passed': result.test_passed,
                'error_message': result.error_message,
                'solution_length': len(result.generated_solution),
                'test_success_rate': result.test_results.get('success_rate', 0) if result.test_results else 0
            })
        
        df = pd.DataFrame(results_data)
        df.to_csv(filepath, index=False)
        print(f"결과가 {filepath}에 저장되었습니다.")
        
        # 상세 결과도 JSON으로 저장
        detailed_filepath = filepath.replace('.csv', '_detailed.json')
        with open(detailed_filepath, 'w', encoding='utf-8') as f:
            json.dump([{
                'task_id': r.task_id,
                'reasoning_type': r.reasoning_type,
                'problem_text': r.problem_text,
                'generated_solution': r.generated_solution,
                'execution_time': r.execution_time,
                'success': r.success,
                'test_passed': r.test_passed,
                'test_results': r.test_results,
                'error_message': r.error_message
            } for r in self.results], f, ensure_ascii=False, indent=2)
        print(f"상세 결과가 {detailed_filepath}에 저장되었습니다.")
    
    def generate_report(self) -> Dict[str, Any]:
        """평가 결과 리포트 생성"""
        if not self.results:
            return {"error": "평가 결과가 없습니다."}
        
        df = pd.DataFrame([{
            'reasoning_type': r.reasoning_type,
            'success': r.success,
            'test_passed': r.test_passed,
            'execution_time': r.execution_time,
            'test_success_rate': r.test_results.get('success_rate', 0) if r.test_results else 0
        } for r in self.results])
        
        report = {
            'total_evaluations': len(self.results),
            'total_problems': len(self.problems),
            'reasoning_types': df['reasoning_type'].unique().tolist(),
            
            # 추론 타입별 성과
            'success_rate_by_reasoning': df.groupby('reasoning_type')['success'].mean().to_dict(),
            'test_pass_rate_by_reasoning': df.groupby('reasoning_type')['test_passed'].mean().to_dict(),
            'avg_execution_time_by_reasoning': df.groupby('reasoning_type')['execution_time'].mean().to_dict(),
            'avg_test_success_rate_by_reasoning': df.groupby('reasoning_type')['test_success_rate'].mean().to_dict(),
            
            # 전체 통계
            'overall_success_rate': df['success'].mean(),
            'overall_test_pass_rate': df['test_passed'].mean(),
            'overall_avg_execution_time': df['execution_time'].mean(),
            'overall_avg_test_success_rate': df['test_success_rate'].mean()
        }
        
        return report
    
    def print_summary(self):
        """결과 요약 출력"""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("MBPP 벤치마크 평가 결과 요약")
        print("="*60)
        
        print(f"총 문제 수: {report['total_problems']}")
        print(f"총 평가 수: {report['total_evaluations']}")
        print(f"평가 추론 방식: {', '.join(report['reasoning_types'])}")
        
        print(f"\n전체 성공률: {report['overall_success_rate']:.2%}")
        print(f"전체 테스트 통과율: {report['overall_test_pass_rate']:.2%}")
        print(f"전체 평균 실행 시간: {report['overall_avg_execution_time']:.2f}초")
        print(f"전체 평균 테스트 성공률: {report['overall_avg_test_success_rate']:.2%}")
        
        print(f"\n--- 추론 방식별 성과 ---")
        for reasoning_type in report['reasoning_types']:
            print(f"\n{reasoning_type.upper()}:")
            print(f"  성공률: {report['success_rate_by_reasoning'][reasoning_type]:.2%}")
            print(f"  테스트 통과율: {report['test_pass_rate_by_reasoning'][reasoning_type]:.2%}")
            print(f"  평균 실행 시간: {report['avg_execution_time_by_reasoning'][reasoning_type]:.2f}초")
            print(f"  평균 테스트 성공률: {report['avg_test_success_rate_by_reasoning'][reasoning_type]:.2%}")


async def main():
    """메인 실행 함수"""
    print("MBPP 벤치마크 평가 시작")
    
    # 평가기 생성 (처음 10개 문제만 테스트)
    evaluator = MBPPEvaluator(model="gpt-4o-mini", max_problems=5)
    
    # 평가 실행
    results = await evaluator.evaluate_all_agents(max_concurrent=1)
    
    # 결과 출력
    evaluator.print_summary()
    
    # 결과 저장
    evaluator.save_results()
    
    print("\n평가 완료!")


if __name__ == "__main__":
    # 비동기 실행
    asyncio.run(main())
