import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2, output_dim: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass processing sequence data to prediction."""
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # Use last time-step hidden state

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, max_norm: float = 1.0) -> float:
        """Train for one epoch with gradient clipping."""
        self.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(self(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def predict(self, x: torch.Tensor, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Generate predictions without gradient tracking."""
        self.eval()
        with torch.no_grad():
            return self(x.to(device)).cpu()

