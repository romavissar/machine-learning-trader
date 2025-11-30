"""Download historical OHLCV data for backtesting (FREE, no API key needed).

Supports fetching up to 2+ years of historical data through pagination.
All exchanges support Romania (EU/MiCA compliant or globally available).
Uses fallback exchanges if primary fails.
"""
import asyncio
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import ccxt.async_support as ccxt

# Romania-friendly exchanges (ordered by reliability for historical data)
EXCHANGES = ["binance", "kraken", "okx", "kucoin", "bitget"]

# Get project root (2 levels up from data/scripts/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Timeframe to milliseconds mapping
TIMEFRAME_MS = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


async def fetch_ohlcv_chunk(exchange, symbol: str, timeframe: str, since: int, limit: int = 1000):
    """Fetch a single chunk of OHLCV data."""
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        return ohlcv
    except Exception as e:
        print(f"    Error fetching chunk: {type(e).__name__}: {e}")
        return None


async def download_ohlcv_paginated(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 730,
    exchange_id: str = None,
) -> pd.DataFrame:
    """Download historical OHLCV data with pagination for extended history.
    
    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        timeframe: Candle timeframe (e.g., "1h", "4h", "1d")
        days: Number of days of history to fetch (default 730 = 2 years)
        exchange_id: Specific exchange to use (or None for auto-fallback)
        
    Returns:
        DataFrame with OHLCV data
    """
    tf_ms = TIMEFRAME_MS.get(timeframe, 60 * 60 * 1000)
    candles_per_day = (24 * 60 * 60 * 1000) // tf_ms
    total_candles = days * candles_per_day
    
    print(f"\n{'='*60}")
    print(f"Downloading {symbol} {timeframe} - {days} days (~{total_candles:,} candles)")
    print(f"{'='*60}")
    
    # Calculate start timestamp (days ago from now)
    now_ms = int(datetime.now().timestamp() * 1000)
    start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
    
    exchanges_to_try = [exchange_id] if exchange_id else EXCHANGES
    
    for ex_id in exchanges_to_try:
        print(f"\nTrying {ex_id}...")
        
        exchange = getattr(ccxt, ex_id)({
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
        try:
            all_candles = []
            current_since = start_ms
            chunk_num = 0
            
            while current_since < now_ms:
                chunk_num += 1
                chunk = await fetch_ohlcv_chunk(exchange, symbol, timeframe, current_since, limit=1000)
                
                if not chunk:
                    print(f"  No more data available from {ex_id}")
                    break
                
                all_candles.extend(chunk)
                
                # Progress update every 5 chunks
                if chunk_num % 5 == 0:
                    fetched_days = len(all_candles) / candles_per_day
                    print(f"  Progress: {len(all_candles):,} candles ({fetched_days:.1f} days)")
                
                # Move to next chunk (last candle timestamp + 1 timeframe)
                last_ts = chunk[-1][0]
                current_since = last_ts + tf_ms
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
                
                # Safety check - if we got less than expected, we've reached the end
                if len(chunk) < 1000:
                    break
            
            if all_candles:
                print(f"  ✓ Successfully fetched {len(all_candles):,} candles from {ex_id}")
                
                # Create DataFrame
                df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                
                # Remove duplicates and sort
                df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
                
                await exchange.close()
                return df
                
        except Exception as e:
            print(f"  ✗ {ex_id} failed: {type(e).__name__}: {e}")
        finally:
            await exchange.close()
    
    print(f"\n❌ Failed to download {symbol} from all exchanges")
    return None


async def download_and_save(symbol: str, timeframe: str = "1h", days: int = 730):
    """Download data and save to CSV file."""
    df = await download_ohlcv_paginated(symbol, timeframe, days)
    
    if df is not None and len(df) > 0:
        # Ensure data directory exists
        DATA_DIR.mkdir(exist_ok=True)
        
        # Save to CSV
        filename = DATA_DIR / f"{symbol.replace('/', '_')}_{timeframe}.csv"
        df.to_csv(filename, index=False)
        
        # Print summary
        print(f"\n📊 Data Summary for {symbol}:")
        print(f"   Total candles: {len(df):,}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Days covered: {(df['timestamp'].max() - df['timestamp'].min()).days}")
        print(f"   Saved to: {filename}")
        
        return df
    return None


async def main():
    """Download 2 years of data for multiple symbols."""
    print("=" * 60)
    print("Historical Data Downloader")
    print("Fetching 2 years of hourly data (FREE, no API key needed)")
    print("All exchanges support Romania")
    print("=" * 60)
    
    symbols = ["BTC/USDT", "ETH/USDT"]
    days = 730  # 2 years
    
    for symbol in symbols:
        await download_and_save(symbol, "1h", days)
        print()
    
    print("\n" + "=" * 60)
    print("🎉 Download complete! Check the 'data' folder for CSV files.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
