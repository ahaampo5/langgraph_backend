import React from 'react';
import './MessageInput.css';

const MessageInput = ({ 
  value, 
  onChange, 
  onSend, 
  onKeyPress, 
  disabled, 
  placeholder 
}) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!disabled && value.trim()) {
      onSend();
    }
  };

  return (
    <form className="message-input-container" onSubmit={handleSubmit}>
      <textarea
        className="message-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyPress={onKeyPress}
        placeholder={placeholder}
        disabled={disabled}
        rows={1}
        style={{
          minHeight: '44px',
          maxHeight: '120px',
          resize: 'none',
          overflow: 'auto'
        }}
      />
      <button
        type="submit"
        className="send-button"
        disabled={disabled || !value.trim()}
        title="메시지 전송 (Enter)"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
        </svg>
      </button>
    </form>
  );
};

export default MessageInput;
