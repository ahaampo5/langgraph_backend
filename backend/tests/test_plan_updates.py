"""
AutoAgent 계획 업데이트 기능 테스트
"""

import asyncio
import os
from graphs.agent.autoagent import AutoAgent

async def test_plan_updates():
    """계획 업데이트 기능 테스트"""
    
    # 환경 변수 설정 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
        return
    
    # 에이전트 생성
    agent = AutoAgent(enable_mcp=False)
    
    # 복잡한 작업 요청 - 여러 단계로 나뉠 수 있는 작업
    test_query = "Python으로 간단한 계산기 프로그램을 만들어서 calculator.py 파일로 저장하고, 테스트도 해주세요"
    
    print(f"테스트 질문: {test_query}")
    print("="*60)
    
    try:
        result = await agent.ainvoke(test_query, thread_id="plan_update_test")
        
        print(f"\n최종 답변:")
        print(result.get('final_answer', '답변을 생성하지 못했습니다.'))
        
        if result.get('plan'):
            print(f"\n📋 최종 계획 상태:")
            plan = result['plan']
            print(f"🎯 목표: {plan.goal}")
            print(f"📊 상태: {plan.status}")
            print(f"📈 진행률: {sum(1 for s in plan.steps if s.status == 'completed')}/{len(plan.steps)}")
            
            print(f"\n📝 단계별 상세:")
            for step in plan.steps:
                status_emoji = {
                    "completed": "✅",
                    "in_progress": "⏳", 
                    "failed": "❌",
                    "pending": "📋"
                }.get(step.status, "❓")
                
                print(f"{status_emoji} 단계 {step.step_id}: {step.description}")
                print(f"   도구: {', '.join(step.tools_needed) if step.tools_needed else 'N/A'}")
                print(f"   상태: {step.status}")
                
                if step.result:
                    result_preview = step.result[:150] + "..." if len(step.result) > 150 else step.result
                    print(f"   결과: {result_preview}")
                print()
                    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_plan_updates())
