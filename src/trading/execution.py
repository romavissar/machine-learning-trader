import asyncio
from dataclasses import dataclass
from typing import Any, Optional
@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    amount: float
    price: Optional[float]
    status: str
    raw: Any
class OrderExecutor:
    def __init__(self, client) -> None:
        self.client = client

    async def market_order(self, symbol: str, side: str, amount: float) -> OrderResult:
        return await self._submit_order(symbol, side, amount, "market", None)

    async def limit_order(self, symbol: str, side: str, amount: float, price: float) -> OrderResult:
        return await self._submit_order(symbol, side, amount, "limit", price)

    async def twap_execute(self, symbol: str, side: str, total_amount: float, duration: float, intervals: int) -> list[OrderResult]:
        if intervals <= 0: raise ValueError("intervals must be positive")
        slice_amount, delay = total_amount / intervals, duration / intervals if intervals > 1 else 0
        results = []
        for i in range(intervals):
            results.append(await self.market_order(symbol, side, slice_amount))
            if i < intervals - 1 and delay > 0:
                await asyncio.sleep(delay)
        return results

    async def cancel_order(self, order_id: str) -> bool:
        try:
            response = await self.client.cancel_order(order_id)
            return bool(response.get("id"))
        except Exception:
            return False

    async def _submit_order(self, symbol: str, side: str, amount: float, order_type: str, price: Optional[float]) -> OrderResult:
        try:
            payload = await self.client.create_order(symbol, order_type, side, amount, price)
            return OrderResult(
                order_id=str(payload.get("id", "")),
                symbol=symbol,
                side=side,
                amount=amount,
                price=payload.get("price", price),
                status=payload.get("status", "unknown"),
                raw=payload,
            )
        except Exception as exc:
            return OrderResult("", symbol, side, amount, price, f"error:{exc}", None)

