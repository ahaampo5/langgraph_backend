"""
AutoAgent 테스트 스크립트
"""

import asyncio
import os
from graphs.agent.autoagent import AutoAgent

async def test_autoagent():
    """AutoAgent 기본 동작 테스트"""
    
    # 환경 변수 설정 (테스트용)
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
        return
    
    # 에이전트 생성 (MCP 비활성화로 간단히 테스트)
    agent = AutoAgent(enable_mcp=False)
    
    # 테스트 질문
    test_query = "현재 디렉토리의 파일 목록을 확인하고, README.md 파일이 있는지 확인해주세요"
    
    print(f"테스트 질문: {test_query}")
    print("="*50)
    
    try:
        result = await agent.ainvoke(test_query, thread_id="test_1")
        
        print(f"최종 답변: {result.get('final_answer', '답변을 생성하지 못했습니다.')}")
        
        if result.get('plan'):
            print(f"\n실행된 계획:")
            plan = result['plan']
            print(f"목표: {plan.goal}")
            for step in plan.steps:
                status_emoji = "✅" if step.status == "completed" else "⏳" if step.status == "in_progress" else "📋"
                print(f"{status_emoji} 단계 {step.step_id}: {step.description} ({step.status})")
                if step.result:
                    print(f"   결과: {step.result[:100]}...")
                    
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_autoagent())
