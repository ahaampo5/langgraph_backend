#!/bin/bash

# 백엔드와 프론트엔드를 동시에 실행하는 스크립트

echo "🚀 LangGraph Chatbot 애플리케이션을 시작합니다..."

# 백엔드 실행 (백그라운드)
echo "📡 백엔드 서버를 시작합니다..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# 프론트엔드 실행
echo "🖥️ 프론트엔드 애플리케이션을 시작합니다..."
cd ../frontend
npm run electron-dev &
FRONTEND_PID=$!

# Ctrl+C로 종료할 때 백그라운드 프로세스들도 함께 종료
trap "echo '🛑 애플리케이션을 종료합니다...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT

# 백그라운드 프로세스들을 기다림
wait $BACKEND_PID $FRONTEND_PID
