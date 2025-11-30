from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


@dataclass
class ArbitrageOpportunity:
    cycle: List[str]  # Token path e.g. ["BTC", "ETH", "USDT", "BTC"]
    profit_pct: float  # Expected profit percentage
    exchanges: List[str]  # Exchange for each hop


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class ArbitrageDetector:
    def __init__(self, num_tokens: int, num_exchanges: int, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_exchanges = num_exchanges
        self.edge_dim = 4 + num_exchanges  # rate, vol, volatility, fee + exchange one-hot
        self.encoder = GraphSAGEEncoder(in_dim=32, hidden=64, out_dim=32).to(self.device)
        self.edge_mlp = nn.Linear(self.edge_dim, 1).to(self.device)
        self.token_embed = nn.Embedding(num_tokens, 32).to(self.device)
        self.tokens: List[str] = []
        self.exchanges: List[str] = []

    def set_tokens(self, tokens: List[str], exchanges: List[str]):
        self.tokens, self.exchanges = tokens, exchanges

    @torch.inference_mode()
    def detect_cycles(self, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                      max_cycle_len: int = 4) -> List[ArbitrageOpportunity]:
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        n = len(self.tokens)
        x = self.token_embed(torch.arange(n, device=self.device))
        node_emb = self.encoder(x, edge_index)
        weights = self.edge_mlp(edge_attr).squeeze(-1) + edge_attr[:, 0]  # learned + log_rate
        dist, pred, ex_used = self._bellman_ford(n, edge_index, weights, edge_attr[:, -self.num_exchanges:])
        return self._extract_cycles(dist, pred, ex_used, max_cycle_len)

    def _bellman_ford(self, n, edge_index, weights, ex_onehot):
        src, dst = edge_index[0], edge_index[1]
        dist = torch.zeros(n, n, device=self.device)
        pred = torch.full((n, n), -1, dtype=torch.long, device=self.device)
        ex_used = torch.full((n, n), -1, dtype=torch.long, device=self.device)
        ex_ids = ex_onehot.argmax(dim=1)
        for start in range(n):
            d = torch.full((n,), float("inf"), device=self.device)
            d[start] = 0
            p = torch.full((n,), -1, dtype=torch.long, device=self.device)
            e = torch.full((n,), -1, dtype=torch.long, device=self.device)
            for _ in range(n - 1):
                new_d = d[src] + weights
                mask = new_d < d[dst]
                d = d.scatter(0, dst[mask], new_d[mask])
                p = p.scatter(0, dst[mask], src[mask])
                e = e.scatter(0, dst[mask], ex_ids[mask])
            dist[start], pred[start], ex_used[start] = d, p, e
        return dist, pred, ex_used

    def _extract_cycles(self, dist, pred, ex_used, max_len) -> List[ArbitrageOpportunity]:
        opportunities = []
        n = dist.size(0)
        for i in range(n):
            if dist[i, i] < -1e-6:  # negative cycle through i
                cycle, exs, cur = [self.tokens[i]], [], i
                for _ in range(max_len):
                    p = pred[i, cur].item()
                    if p < 0: break
                    exs.append(self.exchanges[ex_used[i, cur].item()])
                    cur = p
                    cycle.append(self.tokens[cur])
                    if cur == i: break
                if len(cycle) > 2 and cycle[-1] == cycle[0]:
                    profit = (1 - torch.exp(dist[i, i]).item()) * 100
                    opportunities.append(ArbitrageOpportunity(cycle[::-1], profit, exs[::-1]))
        return sorted(opportunities, key=lambda x: -x.profit_pct)

