import os
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass(frozen=True)
class FINOVAConfig:
    # ===== Данные =====
    START_DATE: str = "2018-01-01"
    END_DATE: str = "2025-12-31"
    ASSETS: List[str] = field(default_factory=lambda: [
        "IMOEX", "SBER", "GAZP", "LKOH",          # MOEX
        "BTC/USDT", "ETH/USDT",                   # Binance
        "^GSPC"                                    # S&P 500 (Yahoo)
    ])
    
    # ===== Пути =====
    DATA_DIR: str = "./data"
    MODELS_DIR: str = "./models"
    LOGS_DIR: str = "./logs"
    RESULTS_DIR: str = "./results"
    
    # ===== Динамическое программирование =====
    DP_HORIZON_MONTHS: int = 60
    DISCOUNT_FACTOR: float = 0.98          # β (дисконтирование будущих наград)
    RISK_AVERSION: float = 3.0             # γ (коэффициент неприятия риска CRRA)
    WEALTH_GRID_SIZE: int = 100
    
    # ===== RL (SAC) =====
    RL_LEARNING_RATE: float = 3e-4
    RL_BUFFER_SIZE: int = 100_000
    RL_BATCH_SIZE: int = 256
    RL_TRAINING_TIMESTEPS: int = 50_000
    RL_EXPLORATION_NOISE: float = 0.1
    
    # ===== GigaChat =====
    GIGACHAT_CLIENT_SECRET: str = os.getenv("GIGACHAT_CLIENT_SECRET", "")
    GIGACHAT_SCOPE: str = "GIGACHAT_API_PERS"
    GIGACHAT_AUTH_URL: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    GIGACHAT_API_URL: str = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    
    # ===== Поведенческие профили =====
    BEHAVIORAL_CONSTRAINTS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'panicker': {'max_stock_allocation': 0.6, 'max_crypto_allocation': 0.0, 'stop_loss_threshold': 0.05, 'rebalance_frequency_days': 30},
        'fomo':     {'max_stock_allocation': 1.0, 'max_crypto_allocation': 0.3, 'stop_loss_threshold': 0.15, 'rebalance_frequency_days': 7},
        'degen':    {'max_stock_allocation': 1.0, 'max_crypto_allocation': 0.5, 'stop_loss_threshold': 0.20, 'rebalance_frequency_days': 1},
        'passive':  {'max_stock_allocation': 0.8, 'max_crypto_allocation': 0.1, 'stop_loss_threshold': 0.10, 'rebalance_frequency_days': 60},
    })
    DEFAULT_PROFILE: str = "passive"
    
    # ===== Бэктест =====
    TRAIN_MONTHS: int = 24
    VAL_MONTHS: int = 12
    TEST_MONTHS: int = 6
    EMBARGO_DAYS: int = 20
    TRANSACTION_COST: float = 0.001   # 0.1%
    SLIPPAGE_BPS: float = 5.0          # 5 базисных пунктов
    
    # ===== Метрики =====
    RISK_FREE_RATE: float = 0.0
    PLAN_ADHERENCE_THRESHOLD: float = 0.10

CONFIG = FINOVAConfig()

# Создаём директории
for d in [CONFIG.DATA_DIR, CONFIG.MODELS_DIR, CONFIG.LOGS_DIR, CONFIG.RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
