import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export const InteractiveDemo = () => {
  // Generate tradeoff data
  const tradeoffData = Array.from({ length: 100 }, (_, i) => {
    const lambda = i / 10;
    // Simulate bias-variance tradeoff
    const variance = 2 / (1 + lambda);  // Decreases with λ
    const bias = 0.3 * lambda;          // Increases with λ
    const mse = variance + bias * bias;  // Total MSE
    
    return {
      lambda: lambda.toFixed(2),
      variance: variance.toFixed(3),
      bias_squared: bias * bias.toFixed(3),
      mse: mse.toFixed(3)
    };
  });

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm space-y-6">
      <div>
        <h3 className="text-lg font-medium mb-2">Bias-Variance Tradeoff in Ridge Regression</h3>
        <p className="text-sm text-gray-600 mb-4">
          Decomposition of Mean Squared Error (MSE = Variance + Bias²)
        </p>
      </div>

      <div style={{ height: 400 }}>
        <ResponsiveContainer>
          <LineChart 
            data={tradeoffData}
            margin={{ top: 20, right: 30, left: 60, bottom: 40 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="lambda" 
              label={{ 
                value: 'Regularization Parameter (λ)', 
                position: 'bottom',
                offset: 0
              }}
            />
            <YAxis
              label={{
                value: 'Error Component',
                angle: -90,
                position: 'insideLeft',
                offset: 0
              }}
            />
            <Tooltip 
              formatter={(value, name) => [
                parseFloat(value).toFixed(3), 
                name === 'mse' ? 'MSE' : 
                name === 'variance' ? 'Variance' : 
                'Bias²'
              ]}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="variance" 
              name="Variance" 
              stroke="#2563eb" 
              strokeWidth={2}
              dot={false}
            />
            <Line 
              type="monotone" 
              dataKey="bias_squared" 
              name="Bias²" 
              stroke="#dc2626" 
              strokeWidth={2}
              dot={false}
            />
            <Line 
              type="monotone" 
              dataKey="mse" 
              name="MSE" 
              stroke="#059669" 
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 gap-8 text-sm">
        <div>
          <h4 className="font-medium mb-2">As λ increases, variance decreases but bias increases, optimal λ balances this tradeoff to minimize MSE, OLS (λ = 0) has high variance but no bias, and large λ reduces variance at cost of increased bias. </h4>
        </div>
        <div>
          <div className="text-gray-600 space-y-2">
            <p>MSE(λ) = Variance(λ) + Bias²(λ)</p>
            <p>Variance ∝ 1/(1 + λ)</p>
            <p>Bias² ∝ λ²</p>
          </div>
        </div>
      </div>

      <div className="text-sm text-gray-500 border-t pt-4">
        <p>This visualization demonstrates how Ridge Regression's 
        regularization parameter λ controls the bias-variance tradeoff. The optimal value 
        of λ occurs at the minimum of the MSE curve, where the best balance between 
        bias and variance is achieved.</p>
      </div>
    </div>
  );
};