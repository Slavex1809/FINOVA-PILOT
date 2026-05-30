import numpy as np

def calculate_sharpe(returns, risk_free=0.0):
    """Расчёт коэффициента Шарпа"""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) - risk_free) / np.std(returns) * np.sqrt(252)

def calculate_max_drawdown(equity):
    """Расчёт максимальной просадки"""
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / (peak + 1e-10)
    return np.max(drawdown)

def calculate_sortino(returns, risk_free=0.0):
    """Расчёт коэффициента Сортино"""
    if len(returns) < 2:
        return 0.0
    downside_returns = returns[returns < risk_free]
    if len(downside_returns) == 0:
        return float('inf')
    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return float('inf')
    return (np.mean(returns) - risk_free) / downside_std * np.sqrt(252)
