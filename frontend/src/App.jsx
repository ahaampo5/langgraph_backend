import React, { useState, useEffect, useRef } from 'react';
import ChatContainer from './components/ChatContainer';
import './App.css';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [socket, setSocket] = useState(null);
  const [clientId] = useState(() => `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);

  const connectWebSocket = () => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      return;
    }

    setIsConnecting(true);
    const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/chat/${clientId}`);
    
    ws.onopen = () => {
      console.log('WebSocket 연결됨');
      setIsConnected(true);
      setIsConnecting(false);
      setSocket(ws);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      } catch (error) {
        console.error('메시지 파싱 오류:', error);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket 연결 종료됨');
      setIsConnected(false);
      setIsConnecting(false);
      setSocket(null);
      
      // 재연결 시도 (5초 후)
      setTimeout(() => {
        if (!isConnected) {
          connectWebSocket();
        }
      }, 5000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket 오류:', error);
      setIsConnecting(false);
    };
  };

  const handleWebSocketMessage = (data) => {
    const timestamp = new Date().toLocaleTimeString();
    
    switch (data.type) {
      case 'connection':
        setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'system',
          content: data.message,
          timestamp
        }]);
        break;
        
      case 'user_message':
        // 사용자 메시지는 이미 UI에 추가되어 있으므로 무시
        break;
        
      case 'agent_thinking':
        setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'thinking',
          content: data.message,
          timestamp
        }]);
        break;
        
      case 'plan_created':
        setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'plan',
          content: data.plan,
          timestamp
        }]);
        break;
        
      case 'step_update':
        setMessages(prev => {
          // 기존 계획 메시지를 찾아서 업데이트
          const updatedMessages = [...prev];
          const planMessageIndex = updatedMessages.findIndex(msg => msg.type === 'plan');
          
          if (planMessageIndex !== -1) {
            const planMessage = updatedMessages[planMessageIndex];
            const updatedSteps = planMessage.content.steps.map(step => 
              step.step_id === data.step.step_id ? data.step : step
            );
            
            updatedMessages[planMessageIndex] = {
              ...planMessage,
              content: {
                ...planMessage.content,
                steps: updatedSteps
              }
            };
          }
          
          return updatedMessages;
        });
        break;
        
      case 'agent_response':
        // thinking 메시지 제거
        setMessages(prev => {
          const filtered = prev.filter(msg => msg.type !== 'thinking');
          return [...filtered, {
            id: Date.now(),
            type: 'agent',
            content: data.message,
            timestamp,
            fullResult: data.full_result
          }];
        });
        break;
        
      case 'error':
        setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'error',
          content: data.message,
          timestamp
        }]);
        break;
        
      default:
        console.log('알 수 없는 메시지 타입:', data);
    }
  };

  const sendMessage = (message) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      console.error('WebSocket이 연결되지 않았습니다');
      return;
    }

    const messageData = {
      type: 'user_message',
      message: message,
      thread_id: `thread_${clientId}`,
      timestamp: new Date().toLocaleTimeString()
    };

    // 사용자 메시지를 즉시 UI에 추가
    setMessages(prev => [...prev, {
      id: Date.now(),
      type: 'user',
      content: message,
      timestamp: messageData.timestamp
    }]);

    // 서버로 메시지 전송
    socket.send(JSON.stringify(messageData));
  };

  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, []);

  return (
    <div className="app">
      <header className="header">
        <h1>🤖 LangGraph Chatbot</h1>
        <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnecting ? '연결 중...' : isConnected ? '✅ 연결됨' : '❌ 연결 끊김'}
        </div>
      </header>
      
      <ChatContainer 
        messages={messages}
        onSendMessage={sendMessage}
        isConnected={isConnected}
      />
    </div>
  );
};

export default App;
