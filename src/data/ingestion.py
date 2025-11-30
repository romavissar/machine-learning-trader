import asyncio
import pandas as pd
import ccxt.async_support as ccxt


class DataIngestion:
    EXCHANGES = ("kucoin", "gate", "huobi", "bitget", "mexc")
    TIMEFRAMES = {"ms": "1m", "seconds": "1s", "hourly": "1h", "daily": "1d", "monthly": "1M"}

    def __init__(self, exchanges=None, limit=200, default_timeframe="hourly"):
        self.ids = exchanges or self.EXCHANGES
        self.limit = limit
        self.default_tf = self.TIMEFRAMES.get(default_timeframe, default_timeframe)

    def _clients(self):
        return [getattr(ccxt, ex)() for ex in self.ids]

    def _df(self, exchange, rows, columns):
        df = pd.DataFrame(rows, columns=columns)
        df["exchange"] = exchange
        if "timestamp" in df:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    async def fetch_ohlcv(self, symbol, timeframe=None):
        tf = self.TIMEFRAMES.get(timeframe, timeframe) or self.default_tf
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        tasks = [self._fetch_ohlcv(c, symbol, tf) for c in self._clients()]
        frames = [f for f in await asyncio.gather(*tasks) if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else self._df("", [], cols)

    async def _fetch_ohlcv(self, client, symbol, timeframe):
        try:
            data = await client.fetch_ohlcv(symbol, timeframe=timeframe, limit=self.limit)
            cols = ["timestamp", "open", "high", "low", "close", "volume"]
            return self._df(client.id, data, cols)
        finally:
            await client.close()

    async def fetch_order_book(self, symbol, depth=20):
        cols = ["timestamp", "bid_price", "bid_size", "ask_price", "ask_size"]
        tasks = [self._fetch_order_book(c, symbol, depth) for c in self._clients()]
        frames = [f for f in await asyncio.gather(*tasks) if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else self._df("", [], cols)

    async def _fetch_order_book(self, client, symbol, depth):
        try:
            book = await client.fetch_order_book(symbol, limit=depth)
            ts = book.get("timestamp") or client.milliseconds()
            bids = book.get("bids", [])[:depth]
            asks = book.get("asks", [])[:depth]
            rows = [[ts, *bid, *ask] for bid, ask in zip(bids, asks)]
            cols = ["timestamp", "bid_price", "bid_size", "ask_price", "ask_size"]
            return self._df(client.id, rows, cols)
        finally:
            await client.close()

