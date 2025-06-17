#!/usr/bin/env python3
"""
HotpotQA 평가 결과 분석 스크립트
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Optional


class HotpotQAResultAnalyzer:
    """HotpotQA 평가 결과 분석기"""
    
    def __init__(self, results_dir: Optional[str] = None):
        self.results_dir = Path(results_dir) if results_dir else Path(__file__).parent / "results"
    
    def load_latest_results(self):
        """최신 평가 결과 로드"""
        summary_files = list(self.results_dir.glob("hotpotqa_summary_*.json"))
        detailed_files = list(self.results_dir.glob("hotpotqa_detailed_results_*.json"))
        
        if not summary_files:
            raise FileNotFoundError("No summary files found")
        
        # 최신 파일 선택
        latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
        latest_detailed = max(detailed_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_summary, 'r') as f:
            summary_data = json.load(f)
        
        with open(latest_detailed, 'r') as f:
            detailed_data = json.load(f)
        
        print(f"Loaded results from: {latest_summary.name}")
        return summary_data, detailed_data
    
    def create_performance_comparison(self, summary_data):
        """성능 비교 차트 생성"""
        methods = list(summary_data.keys())
        accuracies = [summary_data[method]['accuracy'] for method in methods]
        bridge_accuracies = [summary_data[method]['bridge_accuracy'] for method in methods]
        comparison_accuracies = [summary_data[method]['comparison_accuracy'] for method in methods]
        response_times = [summary_data[method]['avg_response_time'] for method in methods]
        
        # 2x2 서브플롯 생성
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 전체 정확도
        bars1 = ax1.bar(methods, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Overall Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 질문 타입별 정확도
        x = range(len(methods))
        width = 0.35
        bars2 = ax2.bar([i - width/2 for i in x], bridge_accuracies, width, 
                       label='Bridge Questions', color='#FF6B6B', alpha=0.7)
        bars3 = ax2.bar([i + width/2 for i in x], comparison_accuracies, width,
                       label='Comparison Questions', color='#4ECDC4', alpha=0.7)
        ax2.set_title('Accuracy by Question Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Method')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.upper() for m in methods])
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # 응답 시간
        bars4 = ax3.bar(methods, response_times, color=['#FFA07A', '#98D8C8', '#87CEEB'])
        ax3.set_title('Average Response Time', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xlabel('Method')
        for i, v in enumerate(response_times):
            ax3.text(i, v + 0.1, f'{v:.2f}s', ha='center', va='bottom')
        
        # 정확도 vs 응답시간 산점도
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, method in enumerate(methods):
            ax4.scatter(response_times[i], accuracies[i], 
                       s=200, c=colors[i], alpha=0.7, label=method.upper())
        ax4.set_xlabel('Response Time (seconds)')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy vs Response Time', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 차트 저장
        chart_path = self.results_dir / f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison chart saved to: {chart_path}")
        plt.show()
    
    def analyze_question_types(self, detailed_data):
        """질문 타입별 분석"""
        print("\n" + "="*60)
        print("QUESTION TYPE ANALYSIS")
        print("="*60)
        
        for method, results in detailed_data.items():
            print(f"\n{method.upper()} Method:")
            
            bridge_questions = [r for r in results if r['question_type'] == 'bridge']
            comparison_questions = [r for r in results if r['question_type'] == 'comparison']
            
            print(f"  Bridge Questions: {len(bridge_questions)}")
            if bridge_questions:
                bridge_correct = sum(1 for r in bridge_questions if r['is_correct'])
                print(f"    Correct: {bridge_correct}/{len(bridge_questions)} ({bridge_correct/len(bridge_questions):.3f})")
                
                # 오답 예시
                bridge_wrong = [r for r in bridge_questions if not r['is_correct']]
                if bridge_wrong:
                    print(f"    Example wrong answer:")
                    example = bridge_wrong[0]
                    print(f"      Q: {example['question'][:80]}...")
                    print(f"      Expected: {example['expected_answer']}")
                    print(f"      Predicted: {example['predicted_answer']}")
            
            print(f"  Comparison Questions: {len(comparison_questions)}")
            if comparison_questions:
                comparison_correct = sum(1 for r in comparison_questions if r['is_correct'])
                print(f"    Correct: {comparison_correct}/{len(comparison_questions)} ({comparison_correct/len(comparison_questions):.3f})")
    
    def analyze_common_errors(self, detailed_data):
        """공통 오류 분석"""
        print("\n" + "="*60)
        print("COMMON ERROR ANALYSIS")
        print("="*60)
        
        all_errors = []
        for method, results in detailed_data.items():
            wrong_answers = [r for r in results if not r['is_correct']]
            for result in wrong_answers:
                all_errors.append({
                    'method': method,
                    'question_type': result['question_type'],
                    'question': result['question'],
                    'expected': result['expected_answer'],
                    'predicted': result['predicted_answer']
                })
        
        if not all_errors:
            print("No errors found!")
            return
        
        # 질문 타입별 오류율
        error_by_type = {}
        for error in all_errors:
            q_type = error['question_type']
            if q_type not in error_by_type:
                error_by_type[q_type] = 0
            error_by_type[q_type] += 1
        
        print(f"Error distribution by question type:")
        for q_type, count in error_by_type.items():
            print(f"  {q_type}: {count} errors")
        
        # 방법론별 오류율
        error_by_method = {}
        for error in all_errors:
            method = error['method']
            if method not in error_by_method:
                error_by_method[method] = 0
            error_by_method[method] += 1
        
        print(f"\nError distribution by method:")
        for method, count in error_by_method.items():
            print(f"  {method.upper()}: {count} errors")
    
    def create_detailed_report(self, summary_data, detailed_data):
        """상세 보고서 생성"""
        report_path = self.results_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("HotpotQA Benchmark Evaluation Report\n")
            f.write("="*50 + "\n\n")
            
            # 요약 통계
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*30 + "\n")
            for method, stats in summary_data.items():
                f.write(f"\n{method.upper()} Method:\n")
                f.write(f"  Total Questions: {stats['total_questions']}\n")
                f.write(f"  Correct Answers: {stats['correct_answers']}\n")
                f.write(f"  Overall Accuracy: {stats['accuracy']:.3f}\n")
                f.write(f"  Bridge Accuracy: {stats['bridge_accuracy']:.3f}\n")
                f.write(f"  Comparison Accuracy: {stats['comparison_accuracy']:.3f}\n")
                f.write(f"  Average Response Time: {stats['avg_response_time']:.2f}s\n")
            
            # 최고 성능
            f.write(f"\nBEST PERFORMANCE\n")
            f.write("-"*30 + "\n")
            best_overall = max(summary_data.items(), key=lambda x: x[1]['accuracy'])
            fastest = min(summary_data.items(), key=lambda x: x[1]['avg_response_time'])
            f.write(f"Best Overall Accuracy: {best_overall[0].upper()} ({best_overall[1]['accuracy']:.3f})\n")
            f.write(f"Fastest Response: {fastest[0].upper()} ({fastest[1]['avg_response_time']:.2f}s)\n")
            
            # 상세 분석
            f.write(f"\nDETAILED ANALYSIS\n")
            f.write("-"*30 + "\n")
            
            for method, results in detailed_data.items():
                f.write(f"\n{method.upper()} Method Details:\n")
                correct_count = sum(1 for r in results if r['is_correct'])
                f.write(f"  Correct: {correct_count}/{len(results)}\n")
                
                # 첫 번째 정답과 오답 예시
                correct_examples = [r for r in results if r['is_correct']]
                wrong_examples = [r for r in results if not r['is_correct']]
                
                if correct_examples:
                    example = correct_examples[0]
                    f.write(f"  Correct Example:\n")
                    f.write(f"    Q: {example['question']}\n")
                    f.write(f"    A: {example['expected_answer']}\n")
                    f.write(f"    Predicted: {example['predicted_answer']}\n")
                
                if wrong_examples:
                    example = wrong_examples[0]
                    f.write(f"  Wrong Example:\n")
                    f.write(f"    Q: {example['question']}\n")
                    f.write(f"    Expected: {example['expected_answer']}\n")
                    f.write(f"    Predicted: {example['predicted_answer']}\n")
        
        print(f"Detailed report saved to: {report_path}")
    
    def run_analysis(self):
        """전체 분석 실행"""
        try:
            print("🔍 Starting HotpotQA Results Analysis...")
            
            # 결과 로드
            summary_data, detailed_data = self.load_latest_results()
            
            # 성능 비교 차트
            self.create_performance_comparison(summary_data)
            
            # 질문 타입 분석
            self.analyze_question_types(detailed_data)
            
            # 공통 오류 분석
            self.analyze_common_errors(detailed_data)
            
            # 상세 보고서 생성
            self.create_detailed_report(summary_data, detailed_data)
            
            print("\n🎉 Analysis completed successfully!")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()


def main():
    """메인 함수"""
    analyzer = HotpotQAResultAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
