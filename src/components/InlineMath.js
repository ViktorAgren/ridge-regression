import React from 'react';
import { MathComponent } from 'mathjax-react';

export const InlineMath = ({ tex }) => {
  return (
    <span className="math-inline">
      <MathComponent 
        tex={tex} 
        display={false}
        style={{
          fontSize: '1em',
        }}
      />
    </span>
  );
};