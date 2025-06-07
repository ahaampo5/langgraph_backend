# LangGraph Chatbot

WebSocket을 통한 실시간 AI 챗봇 데스크톱 애플리케이션입니다. FastAPI 백엔드와 Electron + React 프론트엔드로 구성되어 있습니다.

## 🚀 기능

- **실시간 WebSocket 통신**: 메시지 송수신 및 중간 결과 스트리밍
- **단계별 실행 계획 표시**: AI 에이전트의 작업 계획과 진행상황을 실시간으로 확인
- **데스크톱 애플리케이션**: Electron을 사용한 크로스 플랫폼 데스크톱 앱
- **모던한 UI**: React + CSS를 사용한 아름다운 사용자 인터페이스
- **자동 재연결**: 연결이 끊어져도 자동으로 재연결 시도

## 📁 프로젝트 구조

```
langgraph_backend/
├── backend/                    # FastAPI 백엔드
│   ├── main.py                # 메인 FastAPI 애플리케이션
│   ├── run_server.py          # 서버 실행 스크립트
│   ├── requirements.txt       # Python 의존성
│   ├── api/v1/routers/
│   │   ├── ws/chat.py        # WebSocket 챗봇 라우터
│   │   └── http/             # HTTP 라우터 (향후 확장)
│   └── graphs/agent/
│       └── autoagent.py      # LangGraph 에이전트
├── frontend/                  # Electron + React 프론트엔드
│   ├── package.json
│   ├── vite.config.js
│   ├── public/electron.js    # Electron 메인 프로세스
│   └── src/
│       ├── App.jsx           # 메인 React 컴포넌트
│       └── components/       # React 컴포넌트들
└── start-app.sh              # 전체 애플리케이션 실행 스크립트
```

## 🛠️ 설치 및 실행

### 1. 환경 설정

먼저 OpenAI API 키를 설정해야 합니다:

```bash
# 백엔드 디렉토리에서
cd backend
cp .env.example .env
# .env 파일을 편집하여 실제 API 키 입력
```

### 2. 백엔드 설정

```bash
# Python 가상환경 활성화 (이미 생성되어 있다면)
source .venv/bin/activate

# 의존성 설치
cd backend
pip install -r requirements.txt
```

### 3. 프론트엔드 설정

```bash
# Node.js 의존성 설치
cd frontend
npm install
```

### 4. 애플리케이션 실행

#### 방법 1: 통합 실행 (권장)
```bash
# 루트 디렉토리에서
./start-app.sh
```

#### 방법 2: 개별 실행
```bash
# 터미널 1 - 백엔드 실행
cd backend
python run_server.py

# 터미널 2 - 프론트엔드 실행
cd frontend
npm run electron-dev
```

## 🔧 개발 모드

### 백엔드 개발
```bash
cd backend
# 자동 리로드로 서버 실행
python run_server.py
```

### 프론트엔드 개발
```bash
cd frontend
# Vite 개발 서버만 실행
npm run dev

# Electron과 함께 개발 모드 실행
npm run electron-dev
```

## 🌐 API 엔드포인트

### WebSocket
- `ws://localhost:8000/api/v1/ws/chat/{client_id}` - 챗봇 WebSocket 연결

### HTTP
- `GET /` - 루트 엔드포인트
- `GET /health` - 헬스 체크
- `GET /api/v1/ws/status` - WebSocket 연결 상태 조회

## 📱 사용법

1. 애플리케이션을 실행하면 데스크톱 창이 열립니다
2. 상단에 연결 상태가 표시됩니다 (✅ 연결됨 / ❌ 연결 끊김)
3. 하단 입력창에 메시지를 입력하고 Enter 키를 누르거나 전송 버튼을 클릭합니다
4. AI 에이전트가 작업을 수행하는 동안 다음과 같은 정보가 표시됩니다:
   - 🤔 생각하는 중...
   - 📋 실행 계획 (목표와 단계별 작업)
   - ⏳ 진행 중인 단계
   - ✅ 완료된 단계
   - 🤖 최종 답변

## 🔍 메시지 타입

WebSocket을 통해 주고받는 메시지 타입들:

- `connection` - 연결 상태 알림
- `user_message` - 사용자 메시지
- `agent_thinking` - AI가 생각하는 중
- `plan_created` - 실행 계획 생성됨
- `step_update` - 단계별 진행상황 업데이트
- `agent_response` - AI의 최종 답변
- `error` - 오류 메시지

## 🎨 UI 특징

- **실시간 애니메이션**: 메시지가 나타날 때 부드러운 페이드인 효과
- **타이핑 인디케이터**: AI가 응답을 생성할 때 점 애니메이션
- **단계별 진행 표시**: 계획의 각 단계가 색상으로 구분됨
  - 회색: 대기 중
  - 노란색: 진행 중 (깜빡임 효과)
  - 초록색: 완료됨
- **반응형 디자인**: 창 크기에 따라 자동 조정
- **스크롤 자동화**: 새 메시지가 올 때 자동으로 아래로 스크롤

## 🔧 기술 스택

### 백엔드
- **FastAPI**: 웹 프레임워크
- **uvicorn**: ASGI 서버
- **WebSocket**: 실시간 통신
- **LangGraph**: AI 에이전트 프레임워크
- **OpenAI**: LLM 서비스

### 프론트엔드
- **Electron**: 데스크톱 애플리케이션 프레임워크
- **React**: UI 라이브러리
- **Vite**: 빌드 도구
- **CSS3**: 스타일링 (Flexbox, Grid, Animations)

## 🚀 빌드 및 배포

### 프로덕션 빌드
```bash
# 프론트엔드 빌드
cd frontend
npm run build

# Electron 앱 패키징 (향후 추가 예정)
npm run dist
```

## 🐛 문제 해결

### 연결 오류
- 백엔드 서버가 실행 중인지 확인: `http://localhost:8000/health`
- OpenAI API 키가 올바르게 설정되어 있는지 확인
- 방화벽이 8000 포트를 차단하지 않는지 확인

### 프론트엔드 오류
- Node.js 버전 확인 (권장: 16.x 이상)
- `npm install`로 의존성 재설치
- 브라우저 콘솔에서 오류 메시지 확인

## 📝 라이센스

MIT
