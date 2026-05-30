# FINOVA PILOT 2.0
AI-навигатор для персонализированного инвестирования (DP + RL + LLM)

## Запуск
1. `pip install -r requirements.txt`
2. Скопируйте `.env.example` в `.env` и вставьте `GIGACHAT_CLIENT_SECRET`
3. `python main.py`

## Архитектура
- DP-Executor: опорная стратегия (уравнение Беллмана)
- RL-Strategist: адаптивная коррекция (SAC)
- LLM-Coordinator: интерпретация и координация (GigaChat)
- FeasibilityGuard: отсев нереалистичных целей
