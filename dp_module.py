import numpy as np
from config import CONFIG

def solve_bellman(wealth_grid: np.ndarray, mu: float, sigma: float,
                  discount_factor: float, risk_aversion: float,
                  T: int, behavioral_constraints: dict) -> np.ndarray:
    """
    Оптимальная доля в рисковом активе по уравнению Беллмана.
    """
    M = len(wealth_grid)
    V = np.zeros((T, M))
    policy = np.zeros((T, M))
    
    # Терминальная ценность (CRRA)
    for i, w in enumerate(wealth_grid):
        if risk_aversion == 1:
            V[T-1, i] = np.log(w)
        else:
            V[T-1, i] = (w ** (1 - risk_aversion)) / (1 - risk_aversion)
    
    # Обратная индукция
    for t in range(T-2, -1, -1):
        for i, w in enumerate(wealth_grid):
            # Оптимальная доля по формуле Мертона
            opt_share = (mu - 0) / (risk_aversion * sigma**2)
            max_share = behavioral_constraints.get('max_stock_allocation', 1.0)
            opt_share = np.clip(opt_share, 0.0, max_share)
            
            next_wealth = w * (1 + opt_share * mu)
            value_next = np.interp(next_wealth, wealth_grid, V[t+1])
            
            if risk_aversion == 1:
                current_utility = np.log(w)
            else:
                current_utility = (w ** (1 - risk_aversion)) / (1 - risk_aversion)
            
            V[t, i] = current_utility + discount_factor * value_next
            policy[t, i] = opt_share
    
    return policy
