import React from 'react';
import { CodeWindow } from './CodeWindow';
import { LatexBlock } from './LatexBlock';
import { InteractiveDemo } from './InteractiveDemo';
import { Theorem } from './Theorem';
import { Note } from './Note';
import { InlineMath } from './InlineMath';

export const Article = () => {
  return (
    <article className="journal-article">
      <header>
        <h1 className="text-2xl font-semibold mb-2">Ridge Regression</h1>
        <p className="text-gray-600 italic mb-4">
          Regularized least squares estimation
        </p>
      </header>

      <section>
        <h2 className="section-title">1. Theoretical Framework</h2>
        
        <p>
          The fundamental problem in regression analysis is to estimate an unknown parameter vector 
          <InlineMath tex="\beta^* \in \mathbb{R}^p" /> from noisy observations. Consider a probability
          space <InlineMath tex="(\Omega, \mathcal{F}, \mathbb{P})" />.
        </p>

        <Theorem title="Linear Model Foundation">
          <p>
            On our probability space, we observe pairs <InlineMath tex="(X_i, y_i)_{i=1}^n" /> following:
          </p>
          
          <LatexBlock 
            equation="y = X\beta^* + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2I_n)"
          />
        </Theorem>

        <h3 className="subsection-title">Classical Approach</h3>
        <p>
          The method of least squares seeks to minimize:
        </p>

        <LatexBlock 
          equation="\hat{\beta}_{\text{OLS}} = \underset{\beta \in \mathbb{R}^p}{\text{argmin}} \|y - X\beta\|_2^2"
        />

        <h3 className="subsection-title">Ridge Solution</h3>
        <p>
          Ridge Regression modifies the objective by adding an <InlineMath tex="\ell_2" /> penalty:
        </p>

        <LatexBlock 
          equation="\hat{\beta}_{\text{ridge}} = \underset{\beta \in \mathbb{R}^p}{\text{argmin}} \left\{\|y - X\beta\|_2^2 + \lambda\|\beta\|_2^2\right\}"
        />

        <h3 className="subsection-title">Statistical Properties</h3>
        <LatexBlock 
          equation="\begin{array}{ll}
            \mathbb{E}[\hat{\beta}_{\text{ridge}}] &= (X^\top X + \lambda I_p)^{-1}X^\top X\beta^* \\[1em]
            \text{Bias}(\hat{\beta}_{\text{ridge}}) &= -\lambda(X^\top X + \lambda I_p)^{-1}\beta^* \\[1em]
            \text{Cov}(\hat{\beta}_{\text{ridge}}) &= \sigma^2(X^\top X + \lambda I_p)^{-1}X^\top X(X^\top X + \lambda I_p)^{-1}
          \end{array}"
        />

        <InteractiveDemo />

        <h3 className="subsection-title mt-8">Implementation and Simulation</h3>
        <p className="mb-4">
          The following Python code demonstrates the bias-variance tradeoff in Ridge Regression 
          through simulation:
        </p>

        <CodeWindow
          language="python"
          title="Ridge Regression Simulation"
          code={`
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split

def simulate_ridge_tradeoff(n_samples=1000, n_features=20, n_simulations=100):
    """
    Simulate bias-variance tradeoff in Ridge Regression.
    
    Parameters:
        n_samples: Number of samples
        n_features: Number of features
        n_simulations: Number of Monte Carlo simulations
    """
    # True parameters
    np.random.seed(42)
    beta_true = np.array([1.0 / (i + 1) for i in range(n_features)])
    
    # Generate correlated features
    def generate_data():
        # Create correlation structure
        correlation = 0.7
        cov_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                cov_matrix[i,j] = correlation ** abs(i-j)
        
        # Generate correlated features
        X = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=cov_matrix,
            size=n_samples
        )
        
        # Generate response with noise
        y = X @ beta_true + np.random.normal(0, 0.1, n_samples)
        return X, y
    
    # Arrays to store results
    lambdas = np.logspace(-2, 2, 20)
    bias_results = np.zeros((len(lambdas), n_simulations))
    variance_results = np.zeros((len(lambdas), n_simulations))
    mse_results = np.zeros((len(lambdas), n_simulations))
    
    # Monte Carlo simulation
    for sim in range(n_simulations):
        # Generate training and test data
        X, y = generate_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Fit models with different lambda values
        for i, lambda_val in enumerate(lambdas):
            # Fit Ridge model
            ridge = Ridge(alpha=lambda_val)
            ridge.fit(X_train, y_train)
            
            # Compute bias
            y_pred = ridge.predict(X_test)
            bias = np.mean((y_test - y_pred) ** 2)
            bias_results[i, sim] = bias
            
            # Compute variance (using bootstrap)
            n_bootstrap = 50
            preds = np.zeros((len(y_test), n_bootstrap))
            
            for b in range(n_bootstrap):
                # Bootstrap sample
                idx = np.random.choice(len(X_train), len(X_train))
                ridge_boot = Ridge(alpha=lambda_val)
                ridge_boot.fit(X_train[idx], y_train[idx])
                preds[:, b] = ridge_boot.predict(X_test)
            
            variance = np.mean(np.var(preds, axis=1))
            variance_results[i, sim] = variance
            
            # Compute MSE
            mse_results[i, sim] = bias + variance
    
    # Average results over simulations
    mean_bias = np.mean(bias_results, axis=1)
    mean_variance = np.mean(variance_results, axis=1)
    mean_mse = np.mean(mse_results, axis=1)
    
    return lambdas, mean_bias, mean_variance, mean_mse

# Run simulation
lambdas, bias, variance, mse = simulate_ridge_tradeoff()

# Find optimal lambda
optimal_idx = np.argmin(mse)
optimal_lambda = lambdas[optimal_idx]

print(f"Optimal λ: {optimal_lambda:.3f}")
print(f"At optimal λ:")
print(f"  Bias: {bias[optimal_idx]:.3f}")
print(f"  Variance: {variance[optimal_idx]:.3f}")
print(f"  MSE: {mse[optimal_idx]:.3f}")

# Compare with OLS
ols_bias = bias[0]  # λ ≈ 0
ols_variance = variance[0]
ols_mse = mse[0]

print(f"\nOLS Performance:")
print(f"  Bias: {ols_bias:.3f}")
print(f"  Variance: {ols_variance:.3f}")
print(f"  MSE: {ols_mse:.3f}")
          `}
        />

      </section>
    </article>
  );
};