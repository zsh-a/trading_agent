import asyncio
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from src.market_data import MarketDataProvider
from src.workflow import TradingAgentWorkflow, StartEvent

# Load environment variables
load_dotenv()

async def main():
    # 1. Configuration
    exchange_id = os.getenv("EXCHANGE_ID", "binance")
    symbol = "BTC/USDT"
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return

    # 2. Initialize Components
    print(f"Initializing Agent for {symbol} on {exchange_id}...")
    
    # Initialize LLM (GPT-4o for vision capabilities)
    llm = OpenAI(model="gpt-4o", api_key=api_key, max_tokens=500)
    
    # Initialize Market Data Provider
    market_provider = MarketDataProvider(exchange_id=exchange_id, symbol=symbol)
    
    # Initialize Workflow
    agent = TradingAgentWorkflow(market_provider=market_provider, llm=llm, verbose=True, timeout=120)

    # 3. Run Workflow
    print("--- Starting Trading Cycle ---")
    result = await agent.run(start_event=StartEvent())
    
    print("\n--- Workflow Completed ---")
    print(f"Final Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
