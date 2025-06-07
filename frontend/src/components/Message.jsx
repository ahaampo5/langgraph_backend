import React from 'react';
import PlanDisplay from './PlanDisplay';
import './Message.css';

const Message = ({ message }) => {
  const { type, content, timestamp } = message;

  const renderMessageContent = () => {
    switch (type) {
      case 'plan':
        return <PlanDisplay plan={content} />;
      
      case 'error':
        return (
          <div className="error-content">
            <span className="error-icon">⚠️</span>
            {content}
          </div>
        );
      
      default:
        return content;
    }
  };

  const getMessageIcon = () => {
    switch (type) {
      case 'user':
        return '👤';
      case 'agent':
        return '🤖';
      case 'system':
        return 'ℹ️';
      case 'thinking':
        return '🤔';
      case 'plan':
        return '📋';
      case 'error':
        return '❌';
      default:
        return '';
    }
  };

  return (
    <div className={`message ${type}`}>
      <div className="message-wrapper">
        <div className="message-icon">
          {getMessageIcon()}
        </div>
        <div className="message-content">
          {renderMessageContent()}
          {timestamp && (
            <div className="message-timestamp">
              {timestamp}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Message;
