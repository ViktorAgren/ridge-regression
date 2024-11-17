import React from 'react';

export const Theorem = ({ title, children }) => {
  return (
    <div className="theorem-block">
      {title && (
        <h4 className="theorem-title">{title}</h4>
      )}
      <div className="theorem-content">
        {children}
      </div>
    </div>
  );
};