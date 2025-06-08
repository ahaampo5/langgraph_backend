import React from 'react';

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
    <form className="flex p-4 bg-white border-t border-gray-200 gap-2 items-end" onSubmit={handleSubmit}>
      <textarea
        className="flex-1 px-4 py-3 border-2 border-gray-200 rounded-3xl text-base outline-none font-sans leading-normal transition-colors focus:border-blue-500 focus:shadow-[0_0_0_3px_rgba(59,130,246,0.1)] disabled:bg-gray-50 disabled:text-gray-500 disabled:cursor-not-allowed resize-none overflow-auto"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyPress={onKeyPress}
        placeholder={placeholder}
        disabled={disabled}
        rows={1}
        style={{
          minHeight: '44px',
          maxHeight: '120px'
        }}
      />
      <button
        type="submit"
        className={`bg-gradient-to-r from-blue-500 to-blue-600 text-white border-none rounded-full w-11 h-11 cursor-pointer flex items-center justify-center transition-all duration-200 shadow-button ${
          !disabled && value.trim() 
            ? 'hover:from-blue-600 hover:to-blue-700 hover:-translate-y-0.5 hover:shadow-button-hover active:translate-y-0 active:shadow-button' 
            : '!bg-gray-500 cursor-not-allowed transform-none shadow-none'
        }`}
        disabled={disabled || !value.trim()}
        title="메시지 전송 (Enter)"
      >
        <svg 
          width="20" 
          height="20" 
          viewBox="0 0 24 24" 
          fill="currentColor"
          className={`transition-transform duration-200 ${!disabled && value.trim() ? 'group-hover:translate-x-0.5' : ''}`}
        >
          <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
        </svg>
      </button>
    </form>
  );
};

export default MessageInput;
