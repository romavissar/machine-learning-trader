"""Sentiment feature builder leveraging FinBERT analyzer."""
from pandas import DataFrame, Series
import pandas as pd
from src.models.sentiment.finbert import SentimentAnalyzer


class SentimentFeatures:
    def __init__(self) -> None:
        self.analyzer = SentimentAnalyzer()
        self._latest: DataFrame | None = None

    def process_news(self, news_df: DataFrame) -> DataFrame:
        df = news_df.copy()
        col = next((c for c in ("text", "headline", "content", "body") if c in df.columns), None)
        if col is None:
            raise ValueError("news_df must include a text column.")
        sentiments = self.analyzer.analyze_batch(df[col].fillna("").astype(str).tolist())
        df = df.reset_index(drop=True).join(sentiments.reset_index(drop=True))
        self._latest = df
        return df

    def aggregate_sentiment(self, timeframe: str) -> Series:
        if self._latest is None or "timestamp" not in self._latest.columns:
            raise ValueError("Process news with a timestamp column before aggregating.")
        df = self._latest.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        scores = df.dropna(subset=["timestamp"]).set_index("timestamp")[["positive", "negative"]]
        grouped = scores.resample(timeframe).mean()
        series = grouped["positive"] - grouped["negative"]
        series.name = "net_sentiment"
        return series

    def earnings_sentiment(self, report_text: str) -> float:
        scores = self.analyzer.analyze(report_text or "")
        return scores.get("positive", 0.0) - scores.get("negative", 0.0)

