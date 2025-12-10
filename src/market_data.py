import ccxt
import pandas as pd
import mplfinance as mpf
import os
from datetime import datetime
from typing import Optional, Dict

class MarketDataProvider:
    def __init__(self, exchange_id: str, symbol: str, timeframe: str = '1h', limit: int = 100, use_mock: bool = False, mock_data_path: str = "data/mock_data.csv"):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.use_mock = use_mock
        self.mock_data_path = mock_data_path
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'enableRateLimit': True,
        })
        
        # Check for testnet
        if os.getenv('EXCHANGE_TESTNET', 'false').lower() == 'true':
            self.exchange.set_sandbox_mode(True)

    def fetch_ohlcv(self) -> pd.DataFrame:
        """Fetch OHLCV data from the exchange or mock file."""
        if self.use_mock:
            print(f"Using mock data from {self.mock_data_path}")
            if not os.path.exists(self.mock_data_path):
                 raise FileNotFoundError(f"Mock data file not found: {self.mock_data_path}")
            df = pd.read_csv(self.mock_data_path)
            # Ensure timestamp is in correct format (ms)
            pass 
        else:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=self.limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df = df.assign(timestamp=pd.to_datetime(df['timestamp'], unit='ms'))
        df.set_index('timestamp', inplace=True)
        return df

    def generate_chart(self, df: pd.DataFrame, output_path: str = "chart.png"):
        """Generate a K-line chart with EMA and Volume."""
        # Custom style for better readability by VL models
        mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc)

        # Add EMA 20 as it is common in Al Brooks price action
        ema20 = df['close'].ewm(span=20).mean()
        
        apd = [
            mpf.make_addplot(ema20, color='blue', width=1.5)
        ]

        mpf.plot(
            df,
            type='candle',
            style=s,
            volume=True,
            addplot=apd,
            savefig=dict(fname=output_path, dpi=100, bbox_inches='tight'),
            title=f"{self.symbol} {self.timeframe} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            tight_layout=True
        )
        return output_path

    def get_market_snapshot(self) -> Dict:
        """Get both numerical data and chart path."""
        df = self.fetch_ohlcv()
        chart_path = f"data/{self.symbol.replace('/', '_')}_{self.timeframe}.png"
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        
        self.generate_chart(df, chart_path)
        
        last_row = df.iloc[-1]
        summary = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": last_row['close'],
            "chart_path": os.path.abspath(chart_path),
            "volume_24h": last_row['volume'], # Approximately last bar, simplistic
            "timestamp": last_row.name.isoformat()
        }
        return summary

    def get_current_price(self) -> float:
        """Fetch the latest price."""
        if self.use_mock:
             df = self.fetch_ohlcv()
             return float(df.iloc[-1]['close'])
        else:
             ticker = self.exchange.fetch_ticker(self.symbol)
             return float(ticker['last'])
