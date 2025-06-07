import React, { useState, useRef, useEffect } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import './ChatContainer.css';

const ChatContainer = ({ messages, onSendMessage, isConnected }) => {
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // 에이전트가 thinking 상태인지 확인
    const isAgentThinking = messages.some(msg => msg.type === 'thinking');
    setIsTyping(isAgentThinking);
  }, [messages]);

  const handleSendMessage = () => {
    if (!inputValue.trim() || !isConnected) return;

    onSendMessage(inputValue.trim());
    setInputValue('');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chat-container">
      <div className="messages-container">
        <MessageList messages={messages} />
        {isTyping && (
          <div className="typing-indicator">
            <span>AI가 응답을 생성하는 중</span>
            <div className="typing-dots">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <MessageInput
        value={inputValue}
        onChange={setInputValue}
        onSend={handleSendMessage}
        onKeyPress={handleKeyPress}
        disabled={!isConnected || isTyping}
        placeholder={
          !isConnected 
            ? "서버에 연결 중..." 
            : isTyping 
              ? "AI가 응답을 생성하는 중..." 
              : "메시지를 입력하세요..."
        }
      />
    </div>
  );
};

export default ChatContainer;
