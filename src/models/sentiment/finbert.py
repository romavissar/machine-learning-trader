"""FinBERT sentiment analyzer for financial text."""
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline


class SentimentAnalyzer:
    """Financial sentiment analysis using FinBERT."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline(
            "sentiment-analysis", model=model_name, tokenizer=self.tokenizer,
            device=device, top_k=None, truncation=True, max_length=512
        )

    def analyze(self, text: str) -> dict:
        """Analyze single text, return {positive, negative, neutral} scores."""
        results = self.pipe(text)[0]
        return {r["label"].lower(): r["score"] for r in results}

    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """Batch analyze texts, return DataFrame with sentiment scores."""
        results = self.pipe(texts)
        rows = [{r["label"].lower(): r["score"] for r in res} for res in results]
        return pd.DataFrame(rows, columns=["positive", "negative", "neutral"])

