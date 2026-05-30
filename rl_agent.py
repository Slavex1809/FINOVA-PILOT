import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from config import CONFIG

class TradingEnv(gym.Env):
    def __init__(self, data, dp_weights_sequence, behavioral_constraints):
        super().__init__()
        self.data = data
        self.dp_weights_sequence = dp_weights_sequence
        self.constraints = behavioral_constraints
        self.n_assets = len(data.columns)
        
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(self.n_assets,), dtype=np.float32)
        obs_dim = self.n_assets * 5 + self.n_assets + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.current_step = 0
        self.returns_history = []
        self.equity_curve = [1.0]
        self.weights_history = []
        self.audit_log = []
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.returns_history = []
        self.equity_curve = [1.0]
        self.weights_history = [np.ones(self.n_assets) / self.n_assets]
        self.audit_log = []
        obs = self._get_obs()
        obs = self._validate_obs(obs)
        return obs, {}
    
    def _validate_obs(self, obs):
        if isinstance(obs, (list, tuple)) or (hasattr(obs, 'ndim') and obs.ndim > 1):
            obs = np.array(obs).flatten()
        expected = self.observation_space.shape[0]
        if len(obs) != expected:
            if len(obs) > expected:
                obs = obs[:expected]
            else:
                obs = np.pad(obs, (0, expected - len(obs)), constant_values=0)
        return obs.astype(np.float32)
    
    def _get_obs(self):
        start = max(0, self.current_step - 4)
        rets = self.data.iloc[start:self.current_step+1].values
        if len(rets) < 5:
            pad = 5 - len(rets)
            rets = np.pad(rets, ((pad, 0), (0, 0)), constant_values=0)
        rets_flat = rets.flatten()
        weights = self.weights_history[-1] if self.weights_history else np.ones(self.n_assets) / self.n_assets
        vol = self.data.iloc[self.current_step].std() if self.current_step < len(self.data) else 0.02
        obs = np.concatenate([rets_flat, weights, [vol]]).astype(np.float32)
        return obs
    
    def step(self, action):
        dp_weights = self.dp_weights_sequence[self.current_step % len(self.dp_weights_sequence)]
        raw_weights = dp_weights + action
        raw_weights = np.clip(raw_weights, 0, self.constraints.get('max_stock_allocation', 1.0))
        total = np.sum(raw_weights)
        if total == 0:
            final_weights = np.ones_like(raw_weights) / len(raw_weights)
        else:
            final_weights = raw_weights / total
        
        asset_returns = self.data.iloc[self.current_step].values
        turnover = np.sum(np.abs(final_weights - self.weights_history[-1]))
        transaction_costs = turnover * CONFIG.TRANSACTION_COST
        portfolio_return = np.sum(final_weights * asset_returns) - transaction_costs
        
        self.returns_history.append(portfolio_return)
        new_equity = self.equity_curve[-1] * (1 + portfolio_return)
        self.equity_curve.append(new_equity)
        
        sharpe = self._calculate_sharpe(np.array(self.returns_history))
        mdd = self._calculate_max_drawdown(np.array(self.equity_curve))
        reward = sharpe - 0.5 * mdd - 0.01 * turnover
        
        self.audit_log.append({
            'step': self.current_step,
            'timestamp': pd.Timestamp.now().isoformat(),
            'dp_weights': dp_weights.tolist(),
            'action': action.tolist(),
            'final_weights': final_weights.tolist(),
            'reward': float(reward),
            'portfolio_return': portfolio_return,
            'sharpe': float(sharpe),
            'mdd': float(mdd),
            'turnover': turnover
        })
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        self.weights_history.append(final_weights)
        
        obs = self._get_obs()
        obs = self._validate_obs(obs)
        return obs, float(reward), done, False, {'sharpe': sharpe, 'mdd': mdd, 'turnover': turnover}
    
    @staticmethod
    def _calculate_sharpe(returns, risk_free=0.0):
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return (np.mean(returns) - risk_free) / np.std(returns) * np.sqrt(252)
    
    @staticmethod
    def _calculate_max_drawdown(equity):
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / (peak + 1e-10)
        return np.max(drawdown)

def train_rl_agent(env, total_timesteps=50000):
    model = SAC('MlpPolicy', env, verbose=1, learning_rate=CONFIG.RL_LEARNING_RATE,
                buffer_size=CONFIG.RL_BUFFER_SIZE, batch_size=CONFIG.RL_BATCH_SIZE,
                learning_starts=1000, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1)
    model.learn(total_timesteps=total_timesteps)
    return model
