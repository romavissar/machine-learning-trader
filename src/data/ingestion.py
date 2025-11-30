import asyncio
import pandas as pd
import ccxt.async_support as ccxt


class DataIngestion:
    """Data ingestion from crypto exchanges.
    
    Exchanges selected for:
    - Romania support (EU/MiCA compliant or globally available)
    - Free testnet/sandbox for paper trading
    - Good CCXT library support
    
    Testnets (free paper trading):
    - Binance: testnet.binance.vision
    - Bybit: testnet.bybit.com
    - OKX: Has testnet sandbox
    - Bitget: Has testnet sandbox
    - Kraken: EU regulated, demo available
    """
    # Romania-friendly exchanges with free testnet/paper trading support
    EXCHANGES = ("binance", "bybit", "kraken", "okx", "bitget")
    
    # Testnet configurations for paper trading (free)
    TESTNET_URLS = {
        "binance": {"urls": {"api": "https://testnet.binance.vision/api"}},
        "bybit": {"urls": {"api": "https://api-testnet.bybit.com"}},
        "okx": {"urls": {"api": "https://www.okx.com"}},  # OKX uses demo flag
        "bitget": {"urls": {"api": "https://api.bitget.com"}},  # Bitget uses demo flag
        "kraken": {},  # Kraken doesn't have public testnet, use demo credentials
    }
    
    TIMEFRAMES = {"ms": "1m", "seconds": "1s", "hourly": "1h", "daily": "1d", "monthly": "1M"}

    def __init__(self, exchanges=None, limit=200, default_timeframe="hourly", use_testnet=False):
        """Initialize data ingestion.
        
        Args:
            exchanges: List of exchange IDs to use (defaults to EXCHANGES)
            limit: Number of candles/orderbook levels to fetch
            default_timeframe: Default OHLCV timeframe
            use_testnet: If True, use testnet/sandbox URLs for paper trading (free)
        """
        self.ids = exchanges or self.EXCHANGES
        self.limit = limit
        self.default_tf = self.TIMEFRAMES.get(default_timeframe, default_timeframe)
        self.use_testnet = use_testnet

    def _clients(self):
        clients = []
        for ex in self.ids:
            client = getattr(ccxt, ex)()
            if self.use_testnet:
                client.set_sandbox_mode(True)
                # Apply testnet URLs if available
                if ex in self.TESTNET_URLS and self.TESTNET_URLS[ex]:
                    for key, value in self.TESTNET_URLS[ex].items():
                        setattr(client, key, value)
            clients.append(client)
        return clients

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

