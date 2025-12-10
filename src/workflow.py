from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole, TextBlock, ImageBlock
from src.market_data import MarketDataProvider
from src.prompts import AL_BROOKS_SYSTEM_PROMPT
import json
import os
from typing import Optional

# --- Events ---
class AnalysisEvent(Event):
    summary: dict

class SignalEvent(Event):
    decision: dict
    current_price: float
    symbol: str

class ExecutionEvent(Event):
    action: str
    quantity: float
    symbol: str
    price: float

# --- Workflow ---
class TradingAgentWorkflow(Workflow):
    def __init__(self, market_provider: MarketDataProvider, llm: OpenAI, timeout: int = 60, verbose: bool = False):
        super().__init__(timeout=timeout, verbose=verbose)
        self.market_provider = market_provider
        self.llm = llm

    @step
    async def fetch_market_data(self, ctx: Context, ev: StartEvent) -> AnalysisEvent:
        """Step 1: Fetch data and generate chart."""
        print(">>> [1] Fetching Market Data...")
        summary = self.market_provider.get_market_snapshot()
        return AnalysisEvent(summary=summary)

    @step
    async def analyze_market(self, ctx: Context, ev: AnalysisEvent) -> SignalEvent:
        """Step 2: VL Analysis using GPT-4o."""
        print(">>> [2] Analyzing Chart...")
        
        # Construct prompt with numeric context
        user_prompt_text = f"""
        Current Price: {ev.summary['current_price']}
        Volume: {ev.summary['volume_24h']}
        Symbol: {ev.summary['symbol']}
        Timeframe: {ev.summary['timeframe']}
        """

        # Create Chat Messages
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM, 
                content=AL_BROOKS_SYSTEM_PROMPT
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    TextBlock(text=user_prompt_text),
                    ImageBlock(path=ev.summary['chart_path']),
                ]
            )
        ]

        response = await self.llm.achat(messages)
        
        # Parse JSON response
        # Note: In production, use structured output or Pydantic parsers suitable for the specific LLM
        try:
            raw_text = response.text.replace("```json", "").replace("```", "").strip()
            decision = json.loads(raw_text)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            decision = {"signal": "hold", "confidence": 0, "analysis": "Error parsing output"}
        
        # Basic filter provided here, could move to Risk Step
        if decision['confidence'] < 0.7:
             decision['signal'] = 'hold'
             decision['analysis'] += " (Low Confidence - Forced Hold)"
             
        return SignalEvent(decision=decision, current_price=ev.summary['current_price'], symbol=ev.summary['symbol'])

    @step
    async def risk_management(self, ctx: Context, ev: SignalEvent) -> Optional[ExecutionEvent | StopEvent]:
        """Step 3: Validate signal and calculate size."""
        print(f">>> [3] Risk Validation: {ev.decision['signal']} ({ev.decision['confidence']})")
        
        if ev.decision['signal'] == 'hold':
            return StopEvent(result="Hold Decision - No Action Taken")
            
        # Simplified Sizing Logic
        # In a real agent, check balance: self.market_provider.exchange.fetch_balance()
        usd_size = 100 # Fixed $100 bet for demo
        quantity = usd_size / ev.current_price
        
        return ExecutionEvent(
            action=ev.decision['signal'],
            quantity=quantity,
            symbol=ev.symbol,
            price=ev.current_price
        )

    @step
    async def execute_trade(self, ctx: Context, ev: ExecutionEvent) -> StopEvent:
        """Step 4: Execute Order."""
        print(f">>> [4] Executing {ev.action.upper()} {ev.quantity} {ev.symbol}")
        
        # Dry Run for Safety
        # result = self.market_provider.exchange.create_order(...)
        result = {
            "status": "filled_mock",
            "side": ev.action,
            "amount": ev.quantity,
            "price": ev.price
        }
        
        return StopEvent(result=f"Trade Executed: {result}")
