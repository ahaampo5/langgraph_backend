#!/usr/bin/env python3
"""
WebSocket 챗봇 테스트 클라이언트
"""

import asyncio
import json
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_chat():
    uri = "ws://localhost:8000/api/v1/ws/chat/test_client_123"
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("WebSocket 연결 성공!")
            
            # 연결 메시지 수신
            response = await websocket.recv()
            data = json.loads(response)
            print(f"📡 {data.get('message', data)}")
            
            # 테스트 메시지 전송
            test_message = {
                "type": "user_message",
                "message": "안녕하세요! 간단한 테스트 메시지입니다.",
                "thread_id": "test_thread_123",
                "timestamp": "test_time"
            }
            
            await websocket.send(json.dumps(test_message))
            logger.info(f"✅ 메시지 전송: {test_message['message']}")
            
            # 응답 수신 (타임아웃 설정)
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    data = json.loads(response)
                    
                    message_type = data.get('type', 'unknown')
                    content = data.get('message', data.get('content', str(data)))
                    
                    if message_type == 'user_message':
                        print(f"👤 사용자: {content}")
                    elif message_type == 'agent_thinking':
                        print(f"🤔 AI: {content}")
                    elif message_type == 'plan_created':
                        plan = data.get('plan', {})
                        print(f"📋 계획 생성됨: {plan.get('goal', 'N/A')}")
                        for step in plan.get('steps', []):
                            print(f"   └─ {step.get('step_id')}: {step.get('description')}")
                    elif message_type == 'step_update':
                        step = data.get('step', {})
                        status_emoji = {"completed": "✅", "in_progress": "⏳", "pending": "📋"}.get(step.get('status'), "⚪")
                        print(f"   {status_emoji} 단계 {step.get('step_id')}: {step.get('status')}")
                    elif message_type == 'agent_response':
                        print(f"🤖 AI 답변: {content}")
                        break  # 최종 답변을 받으면 종료
                    elif message_type == 'error':
                        print(f"❌ 오류: {content}")
                        break
                    else:
                        print(f"📬 {message_type}: {content}")
                        
            except asyncio.TimeoutError:
                logger.error("응답 대기 시간 초과")
                
    except websockets.exceptions.ConnectionRefused:
        logger.error("❌ 서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인하세요.")
        print("💡 서버 실행 방법:")
        print("   cd backend && python run_server.py")
    except Exception as e:
        logger.error(f"오류 발생: {e}")

def main():
    print("🧪 WebSocket 챗봇 테스트 클라이언트")
    print("=" * 50)
    
    asyncio.run(test_websocket_chat())
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    main()
