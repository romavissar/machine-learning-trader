"""Multi-agent coordinator with observer pattern and thread-safe aggregation."""
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Protocol
import numpy as np

class TradingAgent(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray: ...

@dataclass
class Signal:
    action: int  # 0=hold, 1=buy, 2=sell
    confidence: float
    agent_id: str

@dataclass
class AggregatedSignal:
    action: int
    confidence: float
    votes: dict[int, float]

class AgentCoordinator:
    """Coordinates multiple trading agents via observer pattern."""

    def __init__(self, executors: dict[str, Any] | None = None) -> None:
        self._agents: dict[str, tuple[TradingAgent, float]] = {}  # id -> (agent, weight)
        self._executors = executors or {}  # asset_class -> executor
        self._lock = Lock()
        self._listeners: list[Callable[[AggregatedSignal], None]] = []

    def register(self, agent_id: str, agent: TradingAgent, weight: float = 1.0) -> None:
        with self._lock:
            self._agents[agent_id] = (agent, weight)

    def unregister(self, agent_id: str) -> None:
        with self._lock:
            self._agents.pop(agent_id, None)

    def subscribe(self, callback: Callable[[AggregatedSignal], None]) -> None:
        self._listeners.append(callback)

    def aggregate(self, obs: np.ndarray, method: str = "weighted") -> AggregatedSignal:
        with self._lock:
            signals = [
                Signal(int(agent.predict(obs)[0] if hasattr(agent.predict(obs), '__len__') else agent.predict(obs)), w, aid)
                for aid, (agent, w) in self._agents.items()
            ]
        votes: dict[int, float] = {}
        for s in signals:
            votes[s.action] = votes.get(s.action, 0) + (s.confidence if method == "confidence" else 1.0)
        if method == "weighted":
            for s in signals:
                votes[s.action] = votes.get(s.action, 0) + next(w for aid, (_, w) in self._agents.items() if aid == s.agent_id)
        best = max(votes, key=votes.get) if votes else 0
        total = sum(votes.values()) or 1
        result = AggregatedSignal(best, votes.get(best, 0) / total, votes)
        for listener in self._listeners:
            listener(result)
        return result

    def route_order(self, asset_class: str, symbol: str, signal: AggregatedSignal) -> Any:
        executor = self._executors.get(asset_class)
        if not executor or signal.action == 0:
            return None
        side = "buy" if signal.action == 1 else "sell"
        return executor, symbol, side, signal.confidence

