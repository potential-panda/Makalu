import pandas as pd
import numpy as np
from typing import Dict, Any
from .sdw_engine import SDWEngine

class SDWReporter:
    def __init__(self, engine: SDWEngine = None):
        self.engine = engine or SDWEngine()

    def attribute_pnl_to_regime(self, pnl_df: pd.DataFrame, symbol: str, date: str) -> str:
        """
        Correlate PnL data with market microstructure features.
        pnl_df expected to have ['ts', 'pnl_inc']
        """
        # Fetch high-level market features for the same date/symbol
        market_query = f"""
            SELECT 
                avg(vo_realized_vol_5m) as avg_vol,
                avg(ll_max_corr_30s) as avg_ll_corr,
                avg(bs_spread_bps_mean_5s) as avg_spread
            FROM volspread v
            JOIN leadlag l ON v.ts = l.ts AND v.symbol = l.symbol AND v.date = l.date
            JOIN bookstate b ON v.ts = b.ts AND v.symbol = b.symbol AND v.date = b.date
            WHERE v.symbol = '{symbol}' AND v.date = '{date.replace("-", "")}'
        """
        
        try:
            market_stats = self.engine.query_to_df(market_query)
            if market_stats.empty:
                return "No market data found for attribution."
            
            stats = market_stats.iloc[0].to_dict()
            total_pnl = pnl_df['pnl_inc'].sum()
            
            # Simple narrative logic
            narrative = f"### Daily PnL Attribution for {symbol} on {date}\n"
            narrative += f"- **Total PnL**: {total_pnl:.2f} bps\n"
            narrative += f"- **Market Conditions**:\n"
            narrative += f"  - Avg Volatility: {stats.get('avg_vol', 0):.4f}\n"
            narrative += f"  - Avg Lead-Lag Correlation: {stats.get('avg_ll_corr', 0):.2f}\n"
            narrative += f"  - Avg Spread: {stats.get('avg_spread', 0):.2f} bps\n\n"
            
            # Insights
            if total_pnl < 0 and stats.get('avg_ll_corr', 0) < 0.4:
                narrative += "> **Insight**: PnL was negative while Lead-Lag correlation was weak (< 0.4). This suggests the model's primary signal source (Binance) was not leading reliably.\n"
            elif total_pnl < 0 and stats.get('avg_spread', 0) > 2.0:
                narrative += "> **Insight**: High execution friction (Spread > 2.0 bps) likely eroded the scalping edge today.\n"
            else:
                narrative += "> **Insight**: Performance was consistent with the current market regime.\n"
                
            return narrative
            
        except Exception as e:
            return f"Failed to generate attribution: {e}"

if __name__ == "__main__":
    # Test with dummy PnL
    reporter = SDWReporter()
    dummy_pnl = pd.DataFrame({
        'ts': [1719878400, 1719878410],
        'pnl_inc': [-0.5, 0.2]
    })
    print(reporter.attribute_pnl_to_regime(dummy_pnl, "XRP_USDT", "20250701"))
