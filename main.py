import asyncio
import os
import json
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from src.market_data import MarketDataProvider
from src.workflow import TradingAgentWorkflow, StartEvent
from src.modelscope_llm import ModelScopeLLM
from src.models import AgentState
import phoenix as px
from llama_index.core import set_global_handler
import time
from workflows.server import WorkflowServer
from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)



# Load environment variables
load_dotenv()

STATE_FILE = "agent_state.json"

# 1. 启动 Phoenix 服务
px.launch_app()

# 2. 设置全局 Handler
set_global_handler("arize_phoenix")

def load_state() -> AgentState:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            return AgentState(**data)
        except Exception as e:
            print(f"Failed to load state: {e}")
    return None

def save_state(state: AgentState):
    try:
        with open(STATE_FILE, "w") as f:
            f.write(state.model_dump_json(indent=2))
        print(f"State saved to {STATE_FILE}")
    except Exception as e:
        print(f"Failed to save state: {e}")

async def main():
    # 1. Configuration
    exchange_id = os.getenv("EXCHANGE_ID", "binance")
    symbol = "BTC/USDT"
    api_key = os.getenv("MODELSCOPE_API_KEY")
    use_mock = os.getenv("MOCK_DATA", "false").lower() == "true"
    
    if not api_key:
        print("Warning: MODELSCOPE_API_KEY not found. Ensure it is set if using ModelScope.")

    # 2. Initialize Components
    print(f"Initializing Agent for {symbol} on {exchange_id} (Mock: {use_mock})...")
    
    # Initialize LLM (ModelScope) - Market Analyst (Vision capable)
    llm = ModelScopeLLM(
        model='Qwen/Qwen3-VL-30B-A3B-Thinking',
        api_key=api_key,
    )

    # Initialize Risk Manager LLM (DeepSeek)
    risk_llm = ModelScopeLLM(
        model='deepseek-ai/DeepSeek-V3.2',
        api_key=api_key,
    )
    
    # Initialize Market Data Provider
    market_provider = MarketDataProvider(
        exchange_id=exchange_id, 
        symbol=symbol,
        use_mock=use_mock
    )
    
    # Initialize Workflow
    agent = TradingAgentWorkflow(
        market_provider=market_provider, 
        llm=llm, 
        risk_llm=risk_llm, 
        verbose=True, 
        timeout=60 * 5
    )

    # Draw all
    draw_all_possible_flows(agent, filename="all_paths.html")


    # 3. Load State
    current_state = load_state()
    if current_state:
        print("Loaded existing state.")
    else:
        print("No existing state found. Starting fresh.")

    server = WorkflowServer()
    server.add_workflow("trading_agent", agent)
    await server.serve("localhost", 8000)

    # 4. Run Workflow
    # print("--- Starting Trading Cycle ---")
    # # We pass the state in the StartEvent. The workflow is designed to look for it there.
    # result = await agent.run(start_event=StartEvent(state=current_state))
    
    # print("\n--- Workflow Completed ---")
    # # The result should be the AgentState object (returned by StopEvent)
    # if isinstance(result, AgentState):
    #     save_state(result)
    #     print(f"Final Balance: {result.cash_balance}, Position: {result.current_position}")
    #     if result.active_plan:
    #          print(f"Active Plan Status: {result.active_plan.status}")
    # else:
    #     print(f"Unexpected result type: {type(result)}. Result: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 
