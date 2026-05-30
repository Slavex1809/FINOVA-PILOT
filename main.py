import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from config import CONFIG
from data_loader import RealTimeDataConnector
from backtest import walk_forward_backtest
from metrics import calculate_sharpe, calculate_max_drawdown

def check_goal_feasibility(initial_capital, target_capital, horizon_months, profile):
    """FeasibilityGuard: проверка реалистичности цели."""
    required_cagr = (target_capital / initial_capital) ** (12.0 / horizon_months) - 1.0
    limits = {'panicker': 0.12, 'passive': 0.25, 'fomo': 0.40, 'degen': 0.80}
    max_cagr = limits.get(profile, 0.25)
    feasible = required_cagr <= max_cagr
    risk_level = 'green' if feasible else ('yellow' if required_cagr <= max_cagr * 1.5 else 'red')
    explanation = f"Требуемая CAGR: {required_cagr:.1%}. Максимум: {max_cagr:.1%}."
    return {
        'feasible': feasible,
        'required_cagr': required_cagr,
        'risk_level': risk_level,
        'explanation': explanation,
        'alternatives': [
            f"Увеличить горизонт до {int(horizon_months * max_cagr / required_cagr)} мес.",
            f"Снизить цель до {initial_capital * (1 + max_cagr) ** (horizon_months/12):.0f} ₽"
        ]
    }

def main():
    print("="*70)
    print("FINOVA PILOT – FULL BACKTEST (NO FALLBACKS)")
    print("="*70)
    
    # ---------- FeasibilityGuard ----------
    initial_capital = 1_000_000
    target_capital = 5_000_000
    horizon_months = 60
    profile = 'fomo'
    
    feasibility = check_goal_feasibility(initial_capital, target_capital, horizon_months, profile)
    print(f"Feasibility: {feasibility['feasible']} ({feasibility['risk_level']})")
    print(f"Explanation: {feasibility['explanation']}")
    
    if not feasibility['feasible']:
        print("Цель нереалистична. Бэктест не будет запущен.")
        return
    
    # Сохраняем результат проверки
    os.makedirs(CONFIG.RESULTS_DIR, exist_ok=True)
    with open(f"{CONFIG.RESULTS_DIR}/feasibility_check.json", 'w') as f:
        json.dump(feasibility, f, indent=2)
    
    # ---------- Загрузка данных ----------
    connector = RealTimeDataConnector(CONFIG)
    print("📡 Загрузка исторических данных...")
    df_prices = connector.get_historical_prices(CONFIG.START_DATE, CONFIG.END_DATE)
    if df_prices.empty:
        raise RuntimeError("Не удалось загрузить исторические данные.")
    print(f"✅ Загружено {len(df_prices)} дней, активы: {list(df_prices.columns)}")
    
    # ---------- Бэктест ----------
    constraints = CONFIG.BEHAVIORAL_CONSTRAINTS[profile]
    os.makedirs(CONFIG.LOGS_DIR, exist_ok=True)
    audit_file = f"{CONFIG.LOGS_DIR}/audit_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    print("🚀 Запуск walk‑forward бэктеста...")
    results_df = walk_forward_backtest(df_prices, constraints, audit_file=audit_file)
    if results_df.empty:
        print("⚠️ Недостаточно данных для бэктеста.")
        return
    
    print(results_df)
    
    # ---------- Итоговые метрики ----------
    avg_sharpe = results_df['sharpe'].mean()
    avg_mdd = results_df['mdd'].mean()
    print(f"\n📊 Итоговые метрики:")
    print(f"   Средний Sharpe: {avg_sharpe:.3f}")
    print(f"   Средний MDD:    {avg_mdd:.2%}")
    
    # Сохраняем результаты
    results_df.to_csv(f"{CONFIG.RESULTS_DIR}/backtest_results.csv", index=False)
    with open(f"{CONFIG.RESULTS_DIR}/final_metrics.json", 'w') as f:
        json.dump({
            'avg_sharpe': avg_sharpe,
            'avg_mdd': avg_mdd,
            'profile': profile,
            'feasibility': feasibility
        }, f, indent=2)
    
    # График
    plt.figure(figsize=(10,6))
    plt.plot(results_df['sharpe'], marker='o', linestyle='-', linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('Sharpe ratio по тестовым окнам')
    plt.xlabel('Окно')
    plt.ylabel('Sharpe ratio')
    plt.grid(True)
    plt.savefig(f"{CONFIG.RESULTS_DIR}/sharpe_by_window.png", dpi=150)
    plt.close()
    
    print(f"✅ Бэктест завершён. Аудит сохранён в {audit_file}")

if __name__ == "__main__":
    main()
