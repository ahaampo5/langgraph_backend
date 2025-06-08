import React from 'react';
import Message from './Message';

const MessageList = ({ messages }) => {
  return (
    <div className="flex flex-col gap-4 pb-4">
      {messages.map((message) => (
        <Message key={message.id} message={message} />
      ))}
    </div>
  );
};

export default MessageList;
