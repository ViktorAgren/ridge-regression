import React from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { solarizedlight } from 'react-syntax-highlighter/dist/esm/styles/prism';

export const CodeWindow = ({ code, language = "python", title }) => {
  return (
    <div className="code-block overflow-hidden">
      {title && (
        <div className="bg-gray-100 px-4 py-2 border-b">
          <span className="text-sm font-medium text-gray-700">{title}</span>
        </div>
      )}
      <SyntaxHighlighter
        language={language}
        style={solarizedlight}
        customStyle={{
          margin: 0,
          padding: '1rem',
          background: '#f8fafc',
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
};
