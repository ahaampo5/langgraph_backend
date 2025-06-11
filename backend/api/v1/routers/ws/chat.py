"""
WebSocket을 통한 실시간 챗봇 통신 라우터
"""

import asyncio
import json
import logging
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.graphs.agent.autoagent import AutoAgent

logger = logging.getLogger(__name__)

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.agents: Dict[str, AutoAgent] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        # 새로운 에이전트 인스턴스 생성
        self.agents[client_id] = AutoAgent(enable_mcp=False)
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.agents:
            del self.agents[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_text(json.dumps(message, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: dict):
        disconnected = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected.append(client_id)
        
        for client_id in disconnected:
            self.disconnect(client_id)

manager = ConnectionManager()

@router.websocket("/chat/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        # 연결 성공 메시지 전송
        await manager.send_message({
            "type": "connection",
            "status": "connected",
            "message": "채팅봇에 연결되었습니다.",
            "client_id": client_id
        }, client_id)
        
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "user_message":
                user_message = message_data.get("message", "")
                thread_id = message_data.get("thread_id", f"thread_{client_id}")
                
                # 사용자 메시지 에코
                await manager.send_message({
                    "type": "user_message",
                    "message": user_message,
                    "timestamp": message_data.get("timestamp")
                }, client_id)
                
                # 에이전트 처리 시작 알림
                await manager.send_message({
                    "type": "agent_thinking",
                    "message": "생각하는 중..."
                }, client_id)
                
                try:
                    # 에이전트로부터 응답 받기
                    agent = manager.agents[client_id]
                    
                    # 에이전트 실행을 위한 콜백 함수 정의
                    async def send_intermediate_result(result_type: str, content: Any):
                        await manager.send_message({
                            "type": "intermediate_result",
                            "result_type": result_type,
                            "content": content
                        }, client_id)
                    
                    # 에이전트 실행
                    result = await agent.ainvoke(user_message, thread_id=thread_id)
                    
                    # 계획이 있는 경우 단계별 결과 전송
                    if result.get('plan'):
                        plan = result['plan']
                        await manager.send_message({
                            "type": "plan_created",
                            "plan": {
                                "goal": plan.goal,
                                "steps": [
                                    {
                                        "step_id": step.step_id,
                                        "description": step.description,
                                        "status": step.status,
                                        "result": step.result
                                    }
                                    for step in plan.steps
                                ]
                            }
                        }, client_id)
                        
                        # 각 단계별 진행상황 전송
                        for step in plan.steps:
                            await manager.send_message({
                                "type": "step_update",
                                "step": {
                                    "step_id": step.step_id,
                                    "description": step.description,
                                    "status": step.status,
                                    "result": step.result
                                }
                            }, client_id)
                    
                    # 최종 답변 전송
                    final_answer = result.get('final_answer', '답변을 생성하지 못했습니다.')
                    await manager.send_message({
                        "type": "agent_response",
                        "message": final_answer,
                        "full_result": result
                    }, client_id)
                    
                except Exception as e:
                    logger.error(f"Error processing message for {client_id}: {e}")
                    await manager.send_message({
                        "type": "error",
                        "message": f"처리 중 오류가 발생했습니다: {str(e)}"
                    }, client_id)
            
            elif message_data.get("type") == "ping":
                # Keepalive ping 응답
                await manager.send_message({
                    "type": "pong",
                    "timestamp": message_data.get("timestamp")
                }, client_id)
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)

@router.get("/status")
async def get_status():
    """현재 연결된 클라이언트 상태 조회"""
    return {
        "active_connections": len(manager.active_connections),
        "connected_clients": list(manager.active_connections.keys())
    }
