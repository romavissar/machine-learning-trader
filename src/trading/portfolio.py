import numpy as np
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, risk_tolerance=1.0, constraints=None):
        self.risk_tol = risk_tolerance
        self.constraints = constraints or [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    def mean_variance_optimize(self, expected_returns, cov_matrix):
        """Maximize utility: w.T * mu - (1/(2*risk_tol)) * w.T * Sigma * w"""
        n_assets = len(expected_returns)
        # Minimize negative utility (maximize utility)
        def objective(w):
            return 0.5 * (w @ cov_matrix @ w) - self.risk_tol * (w @ expected_returns)
        
        res = minimize(objective, np.ones(n_assets) / n_assets, method='SLSQP',
                       bounds=[(0, 1) for _ in range(n_assets)], constraints=self.constraints)
        return res.x

    def risk_parity_weights(self, cov_matrix):
        """Minimize dispersion of risk contributions."""
        n = cov_matrix.shape[0]
        def risk_budget_obj(w):
            p_vol = np.sqrt(w @ cov_matrix @ w)
            risk_contrib = w * (cov_matrix @ w) / p_vol
            return np.sum((risk_contrib - p_vol / n) ** 2)

        res = minimize(risk_budget_obj, np.ones(n) / n, method='SLSQP',
                       bounds=[(0, 1) for _ in range(n)],
                       constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        return res.x

    def apply_constraints(self, weights, min_weight=0.0, max_weight=0.3):
        """Clip weights to constraints and renormalize."""
        clipped = np.clip(weights, min_weight, max_weight)
        if np.sum(clipped) == 0: return np.ones_like(weights) / len(weights)
        return clipped / np.sum(clipped)

