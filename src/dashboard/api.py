from datetime import datetime
import asyncio
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
class Position(BaseModel): asset: str; quantity: float; value: float; pnl: float
class PortfolioSnapshot(BaseModel): positions: list[Position]; total_value: float; total_pnl: float
class Signal(BaseModel): asset: str; direction: str; confidence: float; timestamp: datetime
class SignalsResponse(BaseModel): signals: list[Signal]
class RiskMetrics(BaseModel): drawdown: float; volatility: float; exposure: float
class Trade(BaseModel): id: str; asset: str; side: str; quantity: float; price: float; timestamp: datetime
class TradesResponse(BaseModel): trades: list[Trade]
app = FastAPI(title="MLT Dashboard API")
POSITIONS = [
    Position(asset="BTC-USDT", quantity=0.75, value=21000.0, pnl=1200.0),
    Position(asset="ETH-USDT", quantity=5.0, value=9000.0, pnl=-300.0),
]
SIGNALS = SignalsResponse(signals=[
    Signal(asset="BTC-USDT", direction="buy", confidence=0.74, timestamp=datetime.utcnow()),
    Signal(asset="ETH-USDT", direction="hold", confidence=0.52, timestamp=datetime.utcnow()),
])
RISK = RiskMetrics(drawdown=0.08, volatility=0.21, exposure=0.63)
TRADES = TradesResponse(trades=[
    Trade(id="t1", asset="BTC-USDT", side="buy", quantity=0.5, price=28000.0, timestamp=datetime.utcnow()),
    Trade(id="t2", asset="ETH-USDT", side="sell", quantity=1.0, price=1800.0, timestamp=datetime.utcnow()),
])
@app.get("/portfolio", response_model=PortfolioSnapshot)
async def get_portfolio() -> PortfolioSnapshot:
    total_value = sum(p.value for p in POSITIONS); total_pnl = sum(p.pnl for p in POSITIONS)
    return PortfolioSnapshot(positions=POSITIONS, total_value=total_value, total_pnl=total_pnl)
@app.get("/signals", response_model=SignalsResponse)
async def get_signals() -> SignalsResponse:
    return SIGNALS
@app.get("/risk", response_model=RiskMetrics)
async def get_risk() -> RiskMetrics:
    return RISK
@app.get("/trades", response_model=TradesResponse)
async def get_trades() -> TradesResponse:
    return TRADES
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket) -> None:
    await ws.accept()
    while True:
        snapshot = await get_portfolio()
        await ws.send_json({"portfolio": snapshot.dict(),"signals": SIGNALS.dict(),"risk": RISK.dict(),"trades": TRADES.dict()})
        await asyncio.sleep(2)

