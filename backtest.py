import pandas as pd
import numpy as np
import json
from datetime import datetime
from dp_module import solve_bellman
from rl_agent import TradingEnv, train_rl_agent
from config import CONFIG

def walk_forward_backtest(df_prices: pd.DataFrame, behavioral_constraints: dict, audit_file: str = None):
    returns = np.log(df_prices / df_prices.shift(1)).dropna()
    total_days = len(returns)
    train_days = CONFIG.TRAIN_MONTHS * 21
    val_days = CONFIG.VAL_MONTHS * 21
    test_days = CONFIG.TEST_MONTHS * 21
    step_days = test_days
    results = []
    audit_records = []
    
    for start_idx in range(0, total_days - train_days - val_days - test_days, step_days):
        train_end = start_idx + train_days
        embargo_end = train_end + CONFIG.EMBARGO_DAYS
        val_end = embargo_end + val_days
        test_end = val_end + test_days
        if test_end > total_days:
            break
        
        train_returns = returns.iloc[start_idx:train_end]
        val_returns = returns.iloc[embargo_end:val_end]
        test_returns = returns.iloc[val_end:test_end]
        
        # --- DP обучение ---
        wealth_grid = np.linspace(0.7, 1.3, CONFIG.WEALTH_GRID_SIZE)
        mu = train_returns.mean().mean()
        sigma = train_returns.std().mean()
        dp_policy = solve_bellman(wealth_grid, mu, sigma, CONFIG.DISCOUNT_FACTOR, CONFIG.RISK_AVERSION,
                                  CONFIG.DP_HORIZON_MONTHS, behavioral_constraints)
        dp_weights_seq = np.full((len(test_returns), len(train_returns.columns)), dp_policy.mean())
        
        # --- RL обучение на валидации ---
        env = TradingEnv(val_returns, dp_weights_seq, behavioral_constraints)
        rl_model = train_rl_agent(env, total_timesteps=CONFIG.RL_TRAINING_TIMESTEPS)
        
        # --- Тестирование ---
        test_env = TradingEnv(test_returns, dp_weights_seq, behavioral_constraints)
        obs, _ = test_env.reset()
        done = False
        while not done:
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
        
        # --- Сбор метрик ---
        from metrics import calculate_sharpe, calculate_max_drawdown
        rets_array = np.array(test_env.returns_history)
        eq_array = np.array(test_env.equity_curve)
        window_res = {
            'train_start': train_returns.index[0].isoformat(),
            'test_start': test_returns.index[0].isoformat(),
            'test_end': test_returns.index[-1].isoformat(),
            'sharpe': calculate_sharpe(rets_array),
            'mdd': calculate_max_drawdown(eq_array),
            'turnover': np.mean([np.sum(np.abs(w)) for w in test_env.weights_history if len(w) > 0]),
        }
        results.append(window_res)
        
        for rec in test_env.audit_log:
            rec['window'] = window_res
            rec['timestamp'] = datetime.now().isoformat()
            audit_records.append(rec)
    
    if audit_file:
        with open(audit_file, 'w', encoding='utf-8') as f:
            for rec in audit_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    return pd.DataFrame(results)
