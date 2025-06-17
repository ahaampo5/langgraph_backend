#!/usr/bin/env python3
"""
HotpotQA 평가 결과 간단 분석 스크립트 (차트 없는 버전)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class SimpleResultAnalyzer:
    """간단한 HotpotQA 평가 결과 분석기"""
    
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
        
        print(f"📊 Loaded results from: {latest_summary.name}")
        return summary_data, detailed_data
    
    def print_performance_summary(self, summary_data):
        """성능 요약 출력"""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        # 표 헤더
        print(f"{'Method':<12} {'Accuracy':<10} {'Bridge':<10} {'Comparison':<12} {'Avg Time':<12}")
        print("-" * 70)
        
        # 각 방법론 결과
        for method, stats in summary_data.items():
            print(f"{method.upper():<12} "
                  f"{stats['accuracy']:<10.3f} "
                  f"{stats['bridge_accuracy']:<10.3f} "
                  f"{stats['comparison_accuracy']:<12.3f} "
                  f"{stats['avg_response_time']:<12.2f}s")
        
        # 최고 성능 방법론
        print("\n" + "="*60)
        print("BEST PERFORMANCE")
        print("="*60)
        
        best_overall = max(summary_data.items(), key=lambda x: x[1]['accuracy'])
        best_bridge = max(summary_data.items(), key=lambda x: x[1]['bridge_accuracy'])
        best_comparison = max(summary_data.items(), key=lambda x: x[1]['comparison_accuracy'])
        fastest = min(summary_data.items(), key=lambda x: x[1]['avg_response_time'])
        
        print(f"🏆 Best Overall Accuracy: {best_overall[0].upper()} ({best_overall[1]['accuracy']:.3f})")
        print(f"🌉 Best Bridge Questions: {best_bridge[0].upper()} ({best_bridge[1]['bridge_accuracy']:.3f})")
        print(f"⚖️  Best Comparison Questions: {best_comparison[0].upper()} ({best_comparison[1]['comparison_accuracy']:.3f})")
        print(f"⚡ Fastest Response: {fastest[0].upper()} ({fastest[1]['avg_response_time']:.2f}s)")
    
    def analyze_question_distribution(self, summary_data):
        """질문 분포 분석"""
        print("\n" + "="*60)
        print("QUESTION TYPE DISTRIBUTION")
        print("="*60)
        
        # 첫 번째 방법론의 통계를 사용 (모든 방법론이 같은 데이터셋 사용)
        first_method = list(summary_data.keys())[0]
        stats = summary_data[first_method]
        
        total = stats['total_questions']
        bridge_count = stats['bridge_count']
        comparison_count = stats['comparison_count']
        
        print(f"📋 Total Questions: {total}")
        print(f"🌉 Bridge Questions: {bridge_count} ({bridge_count/total*100:.1f}%)")
        print(f"⚖️  Comparison Questions: {comparison_count} ({comparison_count/total*100:.1f}%)")
    
    def analyze_detailed_results(self, detailed_data):
        """상세 결과 분석"""
        print("\n" + "="*60)
        print("DETAILED ANALYSIS")
        print("="*60)
        
        for method, results in detailed_data.items():
            print(f"\n{method.upper()} Method Analysis:")
            print("-" * 30)
            
            total = len(results)
            correct = sum(1 for r in results if r['is_correct'])
            wrong = total - correct
            
            print(f"  ✅ Correct: {correct}/{total} ({correct/total*100:.1f}%)")
            print(f"  ❌ Wrong: {wrong}/{total} ({wrong/total*100:.1f}%)")
            
            # 질문 타입별 분석
            bridge_results = [r for r in results if r['question_type'] == 'bridge']
            comparison_results = [r for r in results if r['question_type'] == 'comparison']
            
            if bridge_results:
                bridge_correct = sum(1 for r in bridge_results if r['is_correct'])
                print(f"  🌉 Bridge: {bridge_correct}/{len(bridge_results)} correct")
            
            if comparison_results:
                comp_correct = sum(1 for r in comparison_results if r['is_correct'])
                print(f"  ⚖️  Comparison: {comp_correct}/{len(comparison_results)} correct")
            
            # 응답 시간 분석
            times = [r['response_time'] for r in results if r['response_time'] > 0]
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                print(f"  ⏱️  Response Time: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")
    
    def show_examples(self, detailed_data, num_examples: int = 2):
        """정답 및 오답 예시 출력"""
        print("\n" + "="*60)
        print("EXAMPLES")
        print("="*60)
        
        for method, results in detailed_data.items():
            print(f"\n{method.upper()} Examples:")
            print("-" * 30)
            
            # 정답 예시
            correct_examples = [r for r in results if r['is_correct']]
            if correct_examples:
                print("✅ Correct Answer Example:")
                example = correct_examples[0]
                print(f"   Q: {example['question']}")
                print(f"   Expected: {example['expected_answer']}")
                print(f"   Predicted: {example['predicted_answer']}")
                print(f"   Type: {example['question_type']}")
            
            # 오답 예시
            wrong_examples = [r for r in results if not r['is_correct']]
            if wrong_examples:
                print("\n❌ Wrong Answer Example:")
                example = wrong_examples[0]
                print(f"   Q: {example['question']}")
                print(f"   Expected: {example['expected_answer']}")
                print(f"   Predicted: {example['predicted_answer']}")
                print(f"   Type: {example['question_type']}")
    
    def create_text_report(self, summary_data, detailed_data):
        """텍스트 보고서 생성"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.results_dir / f"simple_analysis_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("HotpotQA Benchmark Evaluation Analysis\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 성능 요약
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-"*30 + "\n")
            f.write(f"{'Method':<12} {'Accuracy':<10} {'Bridge':<10} {'Comparison':<12} {'Time':<8}\n")
            f.write("-" * 60 + "\n")
            
            for method, stats in summary_data.items():
                f.write(f"{method.upper():<12} "
                       f"{stats['accuracy']:<10.3f} "
                       f"{stats['bridge_accuracy']:<10.3f} "
                       f"{stats['comparison_accuracy']:<12.3f} "
                       f"{stats['avg_response_time']:<8.2f}s\n")
            
            # 최고 성능
            f.write("\nBEST PERFORMANCE\n")
            f.write("-"*30 + "\n")
            
            best_overall = max(summary_data.items(), key=lambda x: x[1]['accuracy'])
            fastest = min(summary_data.items(), key=lambda x: x[1]['avg_response_time'])
            
            f.write(f"Best Overall: {best_overall[0].upper()} ({best_overall[1]['accuracy']:.3f})\n")
            f.write(f"Fastest: {fastest[0].upper()} ({fastest[1]['avg_response_time']:.2f}s)\n")
            
            # 추천사항
            f.write("\nRECOMMENDATIONS\n")
            f.write("-"*30 + "\n")
            
            if best_overall[1]['accuracy'] > 0.8:
                f.write("✅ Excellent performance! Consider using for production.\n")
            elif best_overall[1]['accuracy'] > 0.6:
                f.write("⚠️  Good performance, but room for improvement.\n")
            else:
                f.write("❌ Performance needs significant improvement.\n")
                f.write("   Consider: prompt engineering, model tuning, or additional tools.\n")
            
            if fastest[1]['avg_response_time'] < 5:
                f.write("⚡ Response time is excellent.\n")
            elif fastest[1]['avg_response_time'] < 10:
                f.write("⏱️  Response time is acceptable.\n")
            else:
                f.write("🐌 Response time may be too slow for real-time applications.\n")
        
        print(f"📄 Text report saved to: {report_path}")
    
    def run_analysis(self):
        """전체 분석 실행"""
        try:
            print("🔍 Starting Simple HotpotQA Results Analysis...")
            
            # 결과 로드
            summary_data, detailed_data = self.load_latest_results()
            
            # 분석 실행
            self.print_performance_summary(summary_data)
            self.analyze_question_distribution(summary_data)
            self.analyze_detailed_results(detailed_data)
            self.show_examples(detailed_data)
            
            # 텍스트 보고서 생성
            self.create_text_report(summary_data, detailed_data)
            
            print("\n🎉 Simple analysis completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()


def main():
    """메인 함수"""
    analyzer = SimpleResultAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
