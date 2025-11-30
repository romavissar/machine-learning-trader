"""Paper trading executor - drop-in replacement for OrderExecutor.

FREE paper trading - no API keys required for price fetching.
All exchanges support Romania (EU/MiCA compliant or globally available).

Supported exchanges for price data:
- binance: Global, MiCA compliant in EU, available in Romania
- bybit: Global, testnet available at testnet.bybit.com
- kraken: EU regulated, fully available in Romania
- okx: Global, sandbox mode available
- bitget: Global, demo mode available
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import ccxt.async_support as ccxt

# Romania-friendly exchanges for price data (all FREE for public endpoints)
SUPPORTED_EXCHANGES = ("binance", "bybit", "kraken", "okx", "bitget")

@dataclass
class PaperBalance:
    cash: float = 10_000.0
    positions: dict[str, float] = field(default_factory=dict)

@dataclass 
class PaperOrderResult:
    order_id: str
    symbol: str
    side: str
    amount: float
    price: Optional[float]
    status: str
    raw: Any = None

class PaperExecutor:
    """Simulated executor using live prices from Romania-friendly exchanges.
    
    FREE to use - only fetches public price data, no API keys required.
    All exchanges support Romania (EU/MiCA compliant or globally available).
    """
    
    def __init__(self, initial_cash: float = 10_000.0, exchange: str = "binance") -> None:
        """Initialize paper executor.
        
        Args:
            initial_cash: Starting capital for simulation
            exchange: Exchange to use for price data (binance, bybit, kraken, okx, bitget)
                     All support Romania and are FREE for public price data.
        """
        if exchange not in SUPPORTED_EXCHANGES:
            exchange = "binance"  # Default to Binance (widely available)
        self.exchange_id = exchange
        self.balance = PaperBalance(cash=initial_cash)
        self.trades: list[dict] = []
        self._order_id = 0
        self._client: Optional[ccxt.Exchange] = None

    async def _get_price(self, symbol: str) -> float:
        if not self._client:
            # Create exchange client - no API key needed for public price data
            self._client = getattr(ccxt, self.exchange_id)()
        ticker = await self._client.fetch_ticker(symbol)
        return float(ticker['last'])

    async def market_order(self, symbol: str, side: str, amount: float) -> PaperOrderResult:
        price = await self._get_price(symbol)
        cost = amount * price
        base = symbol.split('/')[0]
        
        if side == 'buy':
            if cost > self.balance.cash:
                return PaperOrderResult('', symbol, side, amount, price, 'rejected:insufficient_funds')
            self.balance.cash -= cost
            self.balance.positions[base] = self.balance.positions.get(base, 0) + amount
        else:
            if self.balance.positions.get(base, 0) < amount:
                return PaperOrderResult('', symbol, side, amount, price, 'rejected:insufficient_position')
            self.balance.cash += cost
            self.balance.positions[base] -= amount
        
        self._order_id += 1
        trade = {'id': self._order_id, 'symbol': symbol, 'side': side, 'amount': amount, 'price': price, 'time': datetime.now(timezone.utc)}
        self.trades.append(trade)
        return PaperOrderResult(str(self._order_id), symbol, side, amount, price, 'filled', trade)

    async def limit_order(self, symbol: str, side: str, amount: float, price: float) -> PaperOrderResult:
        return await self.market_order(symbol, side, amount)  # Simplified: fill at market

    async def cancel_order(self, order_id: str) -> bool:
        return True  # Paper orders are instant

    def portfolio_value(self, prices: dict[str, float]) -> float:
        pos_value = sum(qty * prices.get(sym, 0) for sym, qty in self.balance.positions.items())
        return self.balance.cash + pos_value

    def summary(self) -> dict:
        return {'cash': self.balance.cash, 'positions': self.balance.positions, 'trade_count': len(self.trades)}

    async def close(self) -> None:
        if self._client:
            await self._client.close()

