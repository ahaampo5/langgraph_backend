import React from 'react';
import PlanDisplay from './PlanDisplay';

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

  const getMessageStyles = () => {
    const baseStyles = "flex mb-4 animate-fade-in";
    switch (type) {
      case 'user':
        return `${baseStyles} justify-end`;
      case 'agent':
      case 'system':
      case 'thinking':
      case 'plan':
      case 'error':
        return `${baseStyles} justify-start`;
      default:
        return `${baseStyles} justify-start`;
    }
  };

  const getWrapperStyles = () => {
    const baseStyles = "flex items-start gap-2 max-w-[80%]";
    return type === 'user' ? `${baseStyles} flex-row-reverse` : baseStyles;
  };

  const getContentStyles = () => {
    switch (type) {
      case 'user':
        return "bg-gradient-to-r from-blue-500 to-blue-600 text-white px-4 py-3 rounded-xl shadow-chat relative leading-relaxed";
      case 'agent':
        return "bg-gray-50 border border-gray-200 text-gray-700 px-4 py-3 rounded-xl shadow-chat relative leading-relaxed";
      case 'system':
        return "bg-blue-50 border border-blue-200 text-blue-700 italic px-4 py-3 rounded-xl shadow-chat relative leading-relaxed";
      case 'thinking':
        return "bg-orange-50 border border-orange-200 text-orange-700 italic px-4 py-3 rounded-xl shadow-chat relative leading-relaxed";
      case 'plan':
        return "bg-purple-50 border border-purple-200 text-purple-700 p-4 rounded-xl shadow-chat relative";
      case 'error':
        return "bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl shadow-chat relative leading-relaxed";
      default:
        return "bg-white px-4 py-3 rounded-xl shadow-chat relative leading-relaxed";
    }
  };

  return (
    <div className={getMessageStyles()}>
      <div className={getWrapperStyles()}>
        <div className="text-xl mt-1 min-w-6 text-center">
          {getMessageIcon()}
        </div>
        <div className={getContentStyles()}>
          {type === 'error' ? (
            <div className="flex items-center gap-2">
              <span className="text-lg">⚠️</span>
              {content}
            </div>
          ) : (
            renderMessageContent()
          )}
          {timestamp && (
            <div className="text-xs text-black/50 mt-2 text-right">
              {timestamp}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Message;
