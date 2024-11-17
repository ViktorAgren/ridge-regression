import React from 'react';
import { MathComponent } from 'mathjax-react';

export const LatexBlock = ({ equation, description }) => {
  return (
    <div className="math-display">
      <div className="math-display-content">
        <MathComponent
          tex={equation}
          display={true}
          style={{
            fontSize: '1.1em',
          }}
        />
      </div>
      {description && (
        <div className="math-description">
          {description}
        </div>
      )}
    </div>
  );
};