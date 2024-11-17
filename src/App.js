import React from 'react';
import { Article } from './components/Article';

const App = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Article />
      </main>
      <footer className="bg-white mt-12 border-t">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <p className="text-sm text-gray-500">
            Viktor Ågren | Ridge Regression Tutorial.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;