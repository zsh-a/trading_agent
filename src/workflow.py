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
from src.prompts import AL_BROOKS_SYSTEM_PROMPT, TRADER_SYSTEM_PROMPT, RISK_MANAGER_SYSTEM_PROMPT
from typing import Optional
from llama_index.core.output_parsers import PydanticOutputParser
from src.models import TradingDecision, AgentState, TradingPlan
import datetime
import json

# --- Events ---
class MarketAnalysisEvent(Event):
    summary: dict

class TradeProposalEvent(Event):
    proposal: TradingPlan
    market_summary: dict

class TradeSignalEvent(Event):
    pass

# --- Workflow ---
class TradingAgentWorkflow(Workflow):
    def __init__(self, market_provider: MarketDataProvider, llm: OpenAI, risk_llm: Optional[OpenAI] = None, timeout: int = 60, verbose: bool = False):
        super().__init__(timeout=timeout, verbose=verbose)
        self.market_provider = market_provider
        self.llm = llm
        self.risk_llm = risk_llm or llm

    @step
    async def analyze_market(self, ctx: Context, ev: StartEvent) -> MarketAnalysisEvent | TradeSignalEvent:
        """Step 1: Market Perception & State Recovery"""
        # 1. Recover "memory" from Context
        # Using a default state if not present (although it should be injected in StartEvent nominally, 
        # but let's be safe and also check ctx)
        current_state = await ctx.store.get("state", default=None)
        
        # If passed via StartEvent (first run), prioritize that and set to context
        event_state = getattr(ev, "state", None)
        if event_state:
            current_state = event_state
            await ctx.store.set("state", current_state)
            
        if not current_state:
             # Fallback default
             current_state = AgentState(cash_balance=10000)
             await ctx.store.set("state", current_state)

        print(f">>> [1] System State: Pos {current_state.current_position}, Cash {current_state.cash_balance}")
        
        # 2. [Logic Guard] Check for existing plan
        if current_state.active_plan and current_state.active_plan.status == "ACTIVE":
             print(f"Active Plan Found: {current_state.active_plan.reasoning}")
             print(">>> Skipping analysis, entering monitor mode...")
             return TradeSignalEvent()

        # 3. If no plan, fetch data and trigger analysis
        print("No active plan. Analyzing market...")
        summary = self.market_provider.get_market_snapshot()
        return MarketAnalysisEvent(summary=summary)

    @step
    async def formulate_plan(self, ctx: Context, ev: MarketAnalysisEvent) -> TradeProposalEvent:
        """Step 2: Trader Agent - Formulate Plan"""
        print(">>> [2] Trader Agent: Proposing Plan...")
        
        # Initialize Output Parser
        # Ideally we want the LLM to output a TradingPlan-like structure
        # But compatible with existing TradingDecision for now or map it
        parser = PydanticOutputParser(TradingPlan)
        format_instructions = parser.get_format_string()

        user_prompt_text = f"""
        Current Price: {ev.summary['current_price']}
        Volume: {ev.summary['volume_24h']}
        Symbol: {ev.summary['symbol']}
        Timeframe: {ev.summary['timeframe']}
        
        Analyze the market and propose a trading plan.
        You MUST return a JSON object with the following fields:
        - action: "BUY", "SELL", or "HOLD"
        - target_price: float (the price to take profit)
        - stop_loss: float (the price to stop loss)
        - reasoning: string (brief explanation)
        - status: "PENDING" (default)

        {format_instructions}
        """

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=TRADER_SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    TextBlock(text=user_prompt_text),
                    ImageBlock(path=ev.summary['chart_path']),
                ]
            )
        ]

        response = await self.llm.achat(messages)
        
        try:
            raw_text = response.message.content.replace("```json", "").replace("```", "").strip()
            proposal = parser.parse(raw_text)
            print(f"Trader Proposal: {proposal.action} - {proposal.reasoning}")
            
        except Exception as e:
            print(f"Error formulating plan: {e}")
            # Fallback to hold
            proposal = TradingPlan(action="HOLD", target_price=0, stop_loss=0, reasoning=f"Error: {e}")

        return TradeProposalEvent(proposal=proposal, market_summary=ev.summary)

    @step
    async def risk_review(self, ctx: Context, ev: TradeProposalEvent) -> TradeSignalEvent | StopEvent:
        """Step 2.5: Risk Manager Agent - Review Plan"""
        print(">>> [2.5] Risk Manager: Reviewing Plan...")
        
        proposal = ev.proposal
        if proposal.action == "HOLD":
            print("Risk Manager: HOLD approved automatically.")
            return StopEvent(result="Trader decided to HOLD. No action.")

        # Construct prompt for Risk Manager
        risk_prompt = f"""
        Trader Proposal:
        Action: {proposal.action}
        Entry: Current Market Price
        Target: {proposal.target_price}
        Stop Loss: {proposal.stop_loss}
        Reasoning: {proposal.reasoning}
        
        Market Context:
        Price: {ev.market_summary['current_price']}
        
        Review this proposal against your rules.
        """
        
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=RISK_MANAGER_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=risk_prompt)
        ]
        
        # Use risk_llm for this step
        response = await self.risk_llm.achat(messages)
        
        try:
            raw_text = response.message.content.replace("```json", "").replace("```", "").strip()
            # We expect a JSON with "approved" boolean
            decision = json.loads(raw_text)
            
            if decision.get("approved"):
                print(f"Risk Manager: APPROVED. {decision.get('reasoning')}")
                state = await ctx.store.get("state")
                
                # Activate the plan
                proposal.status = "ACTIVE" 
                state.active_plan = proposal
                state.last_update_time = datetime.datetime.now().isoformat()
                
                await ctx.store.set("state", state)
                return TradeSignalEvent()
            else:
                print(f"Risk Manager: REJECTED. {decision.get('reasoning')}")
                return StopEvent(result=f"Plan Rejected: {decision.get('reasoning')}")
                
        except Exception as e:
            print(f"Error in Risk Review: {e}")
            return StopEvent(result=f"Risk Review Error: {e}")

    @step
    async def execute_or_monitor(self, ctx: Context, ev: TradeSignalEvent) -> StopEvent:
        """Step 3: Execute & Monitor"""
        print(">>> [3] Execute / Monitor Phase")
        state = await ctx.store.get("state")
        plan = state.active_plan
        
        current_price = self.market_provider.get_current_price() # Assume this method exists or we use latest snapshot
        
        if not plan or plan.status != "ACTIVE":
             return StopEvent(result=state)

        # Monitor / Execute Logic
        if plan.action == "BUY":
            # Check entry conditions or if already entered
            # For this simplified agent, we assume 'ACTIVE' means ready to execute or already executed check
            # Let's say we check if we have position
            if state.current_position == 0:
                print(f"Executing BUY. Entry: {current_price}, Stop: {plan.stop_loss}, Target: {plan.target_price}")
                # Mock fill
                state.current_position = (state.cash_balance * 0.99) / current_price # All in logic for demo
                state.cash_balance -= (state.current_position * current_price)
                print(f"Filled. New Pos: {state.current_position:.4f}")
            else:
                # Monitor
                print(f"Monitoring Long. Current: {current_price}, Stop: {plan.stop_loss}, Target: {plan.target_price}")
                if current_price <= plan.stop_loss:
                    print("STOP LOSS HIT. Selling...")
                    state.cash_balance += state.current_position * current_price
                    state.current_position = 0
                    plan.status = "CLOSED"
                    plan.reasoning += " [Stopped Out]"
                elif current_price >= plan.target_price:
                    print("TARGET HIT. Selling...")
                    state.cash_balance += state.current_position * current_price
                    state.current_position = 0
                    plan.status = "CLOSED"
                    plan.reasoning += " [Target Hit]"
        
        elif plan.action == "SELL":
            # Short logic omitted for brevity, treat as similar to BUY or just exit
             pass
        
        await ctx.store.set("state", state)
        return StopEvent(result=state)
