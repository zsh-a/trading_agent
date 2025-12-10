from pydantic import BaseModel, Field
from typing import Literal, Optional

class TradingDecision(BaseModel):
    signal: Literal["buy", "sell", "hold"] = Field(..., description="The trading signal decision.")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0.")
    analysis: str = Field(..., description="Brief analysis reasoning behind the decision.")


# {
#   "market_cycle": "bull_trend",
#   "analysis": "Price above rising EMA 20 with higher highs (HH) and higher lows (HL) structure. Recent candle shows strong bullish signal bar closing near high with bottom tail. Low volume (2000.0) suggests reduced conviction but price action remains dominant per Al Brooks principles.",
#   "signal": "buy",
#   "confidence": 0.7,
#   "stop_loss_level": 36200.0,
#   "take_profit_level": 36800.0
# }
class TradingPlan(BaseModel):
    market_cycle: Literal["bull_trend", "bear_trend", "trading_range"]
    action: Literal["BUY", "SELL", "HOLD"]
    target_price: float
    stop_loss: float
    status: Literal["PENDING", "ACTIVE", "FILLED", "CLOSED"] = "PENDING"
    reasoning: str  # Record the reasoning for later reference

class AgentState(BaseModel):
    """
    Agent's long-term memory snapshot
    """
    cash_balance: float
    current_position: float = 0.0
    # Store the previous plan
    active_plan: Optional[TradingPlan] = None 
    last_update_time: str = ""
