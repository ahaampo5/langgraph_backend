const { app, BrowserWindow } = require('electron');
const path = require('path');
const isDev = process.env.NODE_ENV === 'development';

function createWindow() {
  // 메인 윈도우 생성
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      webSecurity: false, // 개발 중에만 사용
    },
    titleBarStyle: 'default',
    title: 'LangGraph Chatbot'
  });

  // 개발 모드에서는 Vite 서버로 연결, 빌드된 버전에서는 로컬 파일 로드
  const startUrl = isDev 
    ? 'http://localhost:5173' 
    : `file://${path.join(__dirname, '../dist/index.html')}`;
  
  mainWindow.loadURL(startUrl);

  // 개발 모드에서는 개발자 도구 열기
  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  // 윈도우가 닫혔을 때
  mainWindow.on('closed', () => {
    app.quit();
  });
}

// Electron이 초기화를 완료하고 브라우저 윈도우를 생성할 준비가 되었을 때 호출
app.whenReady().then(createWindow);

// 모든 윈도우가 닫혔을 때
app.on('window-all-closed', () => {
  // macOS에서는 사용자가 명시적으로 Cmd + Q를 누르기 전까지는 
  // 애플리케이션이 메뉴바에 남아있는 것이 일반적입니다
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  // macOS에서는 dock 아이콘이 클릭되고 다른 윈도우가 열려있지 않았다면
  // 새로운 윈도우를 생성하는 것이 일반적입니다
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// 이 파일에서는 애플리케이션의 나머지 메인 프로세스 코드를 포함할 수 있습니다
// 별도의 파일에 분리하고 여기서 require할 수도 있습니다
