import React, { useState, useRef, useEffect } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';

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
    <div className="flex-1 flex flex-col h-[calc(100vh-120px)] bg-white">
      <div className="flex-1 overflow-y-auto p-4 bg-gradient-to-b from-gray-50 to-gray-100 scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100 hover:scrollbar-thumb-gray-400">
        <MessageList messages={messages} />
        {isTyping && (
          <div className="flex items-center gap-2 p-4 mx-2 my-2 bg-gray-50 rounded-xl text-gray-600 italic animate-fade-in">
            <span>AI가 응답을 생성하는 중</span>
            <div className="flex gap-1">
              <div className="w-2 h-2 rounded-full bg-blue-500 animate-typing"></div>
              <div className="w-2 h-2 rounded-full bg-blue-500 animate-typing" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-2 h-2 rounded-full bg-blue-500 animate-typing" style={{ animationDelay: '0.4s' }}></div>
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
