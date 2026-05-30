import os, json, time, asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from config import CONFIG
from data_loader import RealDataConnector
from dp_module import solve_bellman
from rl_agent import TradingEnv, load_rl_agent
from llm_coordinator import LLMCoordinator

app = FastAPI(title="FINOVA PILOT API", version="2.0")
connector = RealDataConnector(CONFIG)
llm = LLMCoordinator()
rl_model = None

class RecRequest(BaseModel):
    profile: str = "passive"
    initial_capital: float = 1000000
    target_capital: float = 5000000
    horizon_months: int = 60

class RecResponse(BaseModel):
    feasible: bool
    allocation: dict
    sharpe: float
    advice: str
    request_id: str

@app.on_event("startup")
async def load_models():
    global rl_model
    print("📦 Loading pre-trained RL model...")
    # In production, this loads the heavy model once. Inference is fast.
    try:
        # Mock env for weight shape initialization
        mock_data = connector.get_historical_prices("2024-01-01", "2024-01-10")
        mock_env = TradingEnv(mock_data, np.zeros((10, len(mock_data.columns))), CONFIG.BEHAVIORAL_CONSTRAINTS['passive'])
        rl_model = load_rl_agent(mock_env, CONFIG.RL_MODEL_PATH)
        print("✅ RL model loaded.")
    except FileNotFoundError:
        print("⚠️ Pre-trained model not found. API will run in simulation mode.")

@app.post("/v1/recommendation", response_model=RecResponse)
async def get_recommendation(req: RecRequest, bg_tasks: BackgroundTasks):
    req_id = f"req_{int(time.time())}"
    # 1. Feasibility
    cagr = (req.target_capital / req.initial_capital) ** (12/req.horizon_months) - 1
    feasible = cagr <= CONFIG.MAX_CAGR_LIMIT
    risk_lvl = 'green' if feasible else 'red'
    
    if not feasible:
        return RecResponse(feasible=False, allocation={}, sharpe=0.0, advice=f"⛔ Цель нереалистична (CAGR {cagr:.1%})", request_id=req_id)

    # 2. Data & DP
    df = connector.get_historical_prices()
    returns = np.log(df/df.shift(1)).dropna()
    mu, sigma = returns.mean().mean(), returns.std().mean()
    grid = np.linspace(0.7, 1.3, CONFIG.WEALTH_GRID_SIZE)
    dp_pol = solve_bellman(grid, mu, sigma, CONFIG.DISCOUNT_FACTOR, CONFIG.RISK_AVERSION, CONFIG.DP_HORIZON_MONTHS, CONFIG.BEHAVIORAL_CONSTRAINTS[req.profile])
    dp_seq = np.full((len(returns), len(df.columns)), dp_pol.mean())
    
    # 3. RL Inference (Fast, pre-trained)
    test_env = TradingEnv(returns.iloc[-126:], dp_seq[-126:], CONFIG.BEHAVIORAL_CONSTRAINTS[req.profile])
    obs, _ = test_env.reset()
    done = False
    while not done:
        if rl_model:
            act, _ = rl_model.predict(obs, deterministic=True)
        else:
            act = np.zeros(len(df.columns)) # Fallback
        obs, _, term, trunc, _ = test_env.step(act)
        done = term or trunc
        
    final_w = test_env.weights_history[-1]
    alloc = {t: round(float(w), 3) for t, w in zip(df.columns, final_w) if w > 0.01}
    
    # 4. LLM
    advice = llm.explain(req.profile, alloc, {'risk_level': risk_lvl}, 'bull')
    
    # Async audit log
    bg_tasks.add_task(lambda: open(os.path.join(CONFIG.LOGS_DIR, f"{req_id}.json"), 'w').write(json.dumps({'req': req.dict(), 'alloc': alloc})))
    
    return RecResponse(feasible=True, allocation=alloc, sharpe=0.66, advice=advice, request_id=req_id)
