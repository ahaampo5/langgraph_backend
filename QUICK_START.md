# 🚀 빠른 시작 가이드

## 전체 애플리케이션 실행 (권장)

```bash
# 프로젝트 루트에서
./start-app.sh
```

## 개별 실행

### 1. 백엔드 실행
```bash
cd backend
python run_server.py
```
- 서버가 http://localhost:8000 에서 실행됩니다
- WebSocket은 ws://localhost:8000/api/v1/ws/chat/{client_id} 에서 사용 가능합니다

### 2. 프론트엔드 실행

#### 웹 브라우저에서 실행:
```bash
cd frontend
npm run dev
```
- http://localhost:5173 에서 접속 가능합니다

#### Electron 데스크톱 앱으로 실행:
```bash
cd frontend
npm run electron
```
- 데스크톱 애플리케이션 창이 열립니다

#### 개발 모드 (Vite + Electron 동시 실행):
```bash
cd frontend
npm run electron-dev
```

## 🧪 테스트

### WebSocket 연결 테스트:
```bash
cd backend
python test_websocket.py
```

### API 헬스 체크:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/ws/status
```

## 📱 사용법

1. 애플리케이션을 실행하면 챗봇 인터페이스가 나타납니다
2. 상단에서 연결 상태를 확인할 수 있습니다 (✅ 연결됨)
3. 하단 입력창에 메시지를 입력하고 Enter를 누르거나 전송 버튼을 클릭합니다
4. AI가 응답을 생성하는 과정을 실시간으로 볼 수 있습니다:
   - 🤔 생각하는 중...
   - 📋 실행 계획 표시
   - ⏳ 단계별 진행상황
   - 🤖 최종 답변

## 🔧 문제 해결

### "연결 끊김" 상태인 경우:
1. 백엔드 서버가 실행 중인지 확인: `curl http://localhost:8000/health`
2. 포트 8000이 사용 중이 아닌지 확인
3. 방화벽 설정 확인

### Electron 앱이 실행되지 않는 경우:
1. Node.js 버전 확인 (16.x 이상 권장)
2. `npm install`로 의존성 재설치
3. `npm run build`로 빌드 테스트

### WebSocket 연결 오류:
1. 브라우저 개발자 도구에서 네트워크 탭 확인
2. CORS 설정 확인
3. 백엔드 로그 확인

---

**즐거운 채팅 되세요! 🎉**
