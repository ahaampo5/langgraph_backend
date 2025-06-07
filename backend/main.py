
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1.routers.ws import chat

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="LangGraph API",
    description="LangGraph API for building and deploying stateful applications with WebSocket support",
    version="1.0.0",
)

# CORS 설정 (Electron 앱을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket 라우터 등록
app.include_router(chat.router, prefix="/api/v1/ws", tags=["websocket"])

@app.get("/")
async def root():
    return {"message": "LangGraph Backend API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}