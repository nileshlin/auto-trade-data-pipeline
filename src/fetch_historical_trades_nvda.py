#!/usr/bin/env python3
"""
Fetches historical tick-level trade data for NVDA from Finnhub API, writes to a temporary CSV,
merges into a final deduplicated CSV, and cleans up daily files.

Usage:
  python fetch_historical_trades_nvda.py --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD> --output-dir <path> [--stream]

Example:
  python fetch_historical_trades_nvda.py --start-date 2023-01-01 --end-date 2023-01-15 --output-dir ../data

Purpose:
  - Fetch NVDA trade data over a specified date range or in real-time.
  - Write to historical_tick_data_3_temp.csv, merge into historical_tick_data_3.csv, and delete daily CSVs.
  - Returns: Candles_1s_calculated.
"""
import argparse
import logging
import json
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Any
import finnhub
import pandas as pd
import os
import pytz
import glob
import re
from multiprocessing import Pool
from dotenv import load_dotenv

load_dotenv()

# Configuration
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
API_CALLS_PER_MIN = 130
MARKET_OPEN_EST = (9, 30)
MARKET_CLOSE_EST = (16, 30)
BATCH_SIZE = 24000
RETRY_ATTEMPTS = 5
RETRY_INITIAL = 1
RETRY_MAX = 10
TEMP_CSV = "historical_tick_data_3_temp.csv"
FINAL_CSV = "historical_tick_data_3.csv"
DEBUG_CSV = "invalid_ticks.csv"
TIMESTAMP_REGEX = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6} UTC$"

# Logging Setup
logger = logging.getLogger("nvda_tick_fetcher")
logger.setLevel(logging.INFO)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "symbol": getattr(record, "symbol", "NVDA"),
            "date": getattr(record, "date", "N/A")
        }
        if hasattr(record, "exception"):
            log_entry["exception"] = record.exception
        return json.dumps(log_entry, default=str)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

# Timezone Setup
UTC = pytz.utc
EST = pytz.timezone("US/Eastern")

# Client Initialization
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
call_timestamps = deque()
rate_limit_lock = threading.Lock()

# Progress Tracking
total_rows = 0
failed_dates = {}

def enforce_rate_limit():
    """Ensure API calls stay within rate limit (130/min)."""
    with rate_limit_lock:
        now = time.time()
        while call_timestamps and now - call_timestamps[0] >= 60:
            call_timestamps.popleft()
        if len(call_timestamps) >= API_CALLS_PER_MIN:
            sleep_time = 60 - (now - call_timestamps[0]) + 0.1
            logger.info(f"Rate limit reached; sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        call_timestamps.append(time.time())

# Check if date is a weekend (EST)
def is_weekend(date: datetime) -> bool:
    est_date = date.astimezone(EST)
    return est_date.weekday() >= 5

# Validate trades for nulls, invalid values, and timestamp format
def validate_trades(rows: List[Dict], symbol: str, date_str: str) -> List[Dict]:
    valid_rows = []
    invalid_rows = []
    
    for row in rows:
        # Check nulls
        if row["price"] is None or row["volume"] is None:
            logger.warning(f"Null price/volume for {symbol} on {date_str}", 
                           extra={"symbol": symbol, "date": date_str})
            invalid_rows.append(row)
            continue
        # Check invalid values
        if row["price"] <= 0 or row["volume"] < 0:
            logger.warning(f"Invalid price/volume for {symbol} on {date_str}: {row}", 
                           extra={"symbol": symbol, "date": date_str})
            invalid_rows.append(row)
            continue
        # Check timestamp format
        if not re.match(TIMESTAMP_REGEX, row["timestamp"]):
            logger.warning(f"Invalid timestamp format for {symbol} on {date_str}: {row['timestamp']}", 
                           extra={"symbol": symbol, "date": date_str})
            invalid_rows.append(row)
            continue
        valid_rows.append(row)
    
    if invalid_rows:
        df_invalid = pd.DataFrame(invalid_rows)
        mode = "a" if os.path.exists(DEBUG_CSV) else "w"
        df_invalid.to_csv(DEBUG_CSV, mode=mode, header=not os.path.exists(DEBUG_CSV), index=False)
        logger.info(f"Saved {len(invalid_rows)} invalid rows to {DEBUG_CSV}", 
                    extra={"symbol": symbol, "date": date_str})
    
    if len(rows) != len(valid_rows):
        logger.warning(f"Dropped {len(rows) - len(valid_rows)} invalid trades for {symbol} on {date_str}", 
                       extra={"symbol": symbol, "date": date_str})
    
    return valid_rows

# Fetch a batch of ticks with retries
def fetch_tick_batch(symbol: str, market_open_utc: datetime, skip: int, limit: int) -> List[Dict]:
    enforce_rate_limit()
    est_date = market_open_utc.astimezone(EST).date()
    est_date_str = est_date.strftime("%Y-%m-%d")
    
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            data = finnhub_client.stock_tick(symbol, est_date_str, limit=limit, skip=skip)
            
            if not data.get("t"):
                logger.info(f"No ticks for {symbol} on {est_date_str} at skip={skip}", 
                            extra={"symbol": symbol, "date": est_date_str})
                return []
            
            rows = [{
                "symbol": data.get("s", symbol),
                "timestamp": datetime.utcfromtimestamp(t / 1000).strftime("%Y-%m-%d %H:%M:%S.%f UTC"),
                "price": float(data["p"][i]) if i < len(data["p"]) else None,
                "volume": float(data["v"][i]) if i < len(data["v"]) else None
            } for i, t in enumerate(data["t"])]
            
            valid_rows = validate_trades(rows, symbol, est_date_str)
            if not valid_rows:
                logger.info(f"No valid ticks for {symbol} on {est_date_str} at skip={skip}", 
                            extra={"symbol": symbol, "date": est_date_str})
                return []
            
            logger.info(f"Fetched {len(valid_rows)} ticks for {symbol} on {est_date_str} at skip={skip}", 
                        extra={"symbol": symbol, "date": est_date_str})
            return valid_rows
        
        except Exception as e:
            if attempt < RETRY_ATTEMPTS:
                sleep_time = RETRY_INITIAL * (2 ** (attempt - 1))
                logger.warning(f"Retry {attempt}/{RETRY_ATTEMPTS} for {symbol} on {est_date_str} at skip={skip}: {str(e)}", 
                               extra={"symbol": symbol, "date": est_date_str})
                time.sleep(sleep_time)
            else:
                failed_dates.setdefault((symbol, est_date_str), 0)
                failed_dates[(symbol, est_date_str)] += 1
                logger.error(f"Failed to fetch {limit} ticks for {symbol} on {est_date_str} at skip={skip}: {str(e)}", 
                             extra={"symbol": symbol, "date": est_date_str})
                return []

# Write rows to the temporary CSV file in chunks
def write_to_temp_csv(rows: List[Dict], output_dir: str, symbol: str, date_str: str) -> int:
    if not rows:
        return 0
    
    os.makedirs(output_dir, exist_ok=True)
    temp_csv_file = os.path.join(output_dir, TEMP_CSV)
    
    df = pd.DataFrame(rows)
    mode = "a" if os.path.exists(temp_csv_file) else "w"
    header = not os.path.exists(temp_csv_file)
    df.to_csv(temp_csv_file, mode=mode, header=header, index=False)
    
    logger.info(f"Wrote {len(df)} rows to {temp_csv_file} for {date_str}", 
                extra={"symbol": symbol, "date": date_str})
    return len(df)

# Merge temp CSV into final CSV, deduplicating by symbol and timestamp
def merge_temp_to_final_csv(output_dir: str, symbol: str) -> int:
    temp_csv_file = os.path.join(output_dir, TEMP_CSV)
    final_csv_file = os.path.join(output_dir, FINAL_CSV)
    
    if not os.path.exists(temp_csv_file):
        logger.warning(f"No temp CSV found at {temp_csv_file}")
        return 0
    
    try:
        temp_df = pd.read_csv(temp_csv_file)
        initial_rows = len(temp_df)
        
        # Validate timestamps
        temp_df["timestamp_valid"] = temp_df["timestamp"].apply(lambda x: bool(re.match(TIMESTAMP_REGEX, str(x))))
        invalid_rows = temp_df[~temp_df["timestamp_valid"]]
        if not invalid_rows.empty:
            mode = "a" if os.path.exists(DEBUG_CSV) else "w"
            invalid_rows.to_csv(DEBUG_CSV, mode=mode, header=not os.path.exists(DEBUG_CSV), index=False)
            logger.warning(f"Found {len(invalid_rows)} rows with invalid timestamps in temp CSV; saved to {DEBUG_CSV}")
            temp_df = temp_df[temp_df["timestamp_valid"]]
        
        # Remove nulls and invalid values
        temp_df = temp_df.dropna(subset=["price", "volume"])
        nulls_dropped = initial_rows - len(temp_df)
        if nulls_dropped > 0:
            logger.warning(f"Dropped {nulls_dropped} rows with null price/volume from temp CSV")
        
        initial_rows = len(temp_df)
        temp_df = temp_df[(temp_df["price"] > 0) & (temp_df["volume"] >= 0)]
        invalids_dropped = initial_rows - len(temp_df)
        if invalids_dropped > 0:
            logger.warning(f"Dropped {invalids_dropped} rows with invalid price/volume from temp CSV")
        
        # Read existing final CSV
        if os.path.exists(final_csv_file):
            final_df = pd.read_csv(final_csv_file)
            combined_df = pd.concat([final_df, temp_df], ignore_index=True)
        else:
            combined_df = temp_df
        
        # Deduplicate
        initial_rows = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        duplicates_dropped = initial_rows - len(combined_df)
        if duplicates_dropped > 0:
            logger.warning(f"Dropped {duplicates_dropped} duplicate rows during merge")
        
        # Write to final CSV
        combined_df.drop(columns=["timestamp_valid"], errors="ignore").to_csv(final_csv_file, index=False)
        logger.info(f"Merged {len(combined_df)} rows into {final_csv_file}")
        
        return len(combined_df)
    
    except Exception as e:
        logger.error(f"Failed to merge temp CSV to final CSV: {str(e)}")
        raise

#  Truncate the temp CSV file after merging
def truncate_temp_csv(output_dir: str) -> None:
    temp_csv_file = os.path.join(output_dir, TEMP_CSV)
    try:
        if os.path.exists(temp_csv_file):
            with open(temp_csv_file, "w") as f:
                f.write("")
            logger.info(f"Truncated {temp_csv_file}")
    except Exception as e:
        logger.error(f"Failed to truncate temp CSV: {str(e)}")

# Delete all daily CSV files, keeping temp and final CSVs.
def cleanup_daily_csvs(output_dir: str, symbol: str) -> None:
    pattern = os.path.join(output_dir, f"{symbol}_*.csv")
    temp_csv_file = os.path.join(output_dir, TEMP_CSV)
    final_csv_file = os.path.join(output_dir, FINAL_CSV)
    
    for csv_file in glob.glob(pattern):
        if csv_file not in [temp_csv_file, final_csv_file]:
            try:
                os.remove(csv_file)
                logger.info(f"Deleted daily CSV: {csv_file}")
            except Exception as e:
                logger.error(f"Failed to delete {csv_file}: {str(e)}")

# Validate the final CSV for nulls, invalids, and coverage
def validate_final_csv(output_dir: str, symbol: str, start_date: datetime, end_date: datetime) -> None:
    final_csv_file = os.path.join(output_dir, FINAL_CSV)
    
    if not os.path.exists(final_csv_file):
        logger.warning(f"No final CSV found at {final_csv_file}")
        return
    
    try:
        df = pd.read_csv(final_csv_file)
        total_rows = len(df)
        null_rows = df[["price", "volume"]].isna().any(axis=1).sum()
        invalid_rows = ((df["price"] <= 0) | (df["volume"] < 0)).sum()
        df["timestamp_valid"] = df["timestamp"].apply(lambda x: bool(re.match(TIMESTAMP_REGEX, str(x))))
        invalid_ts_rows = len(df[~df["timestamp_valid"]])
        
        if invalid_ts_rows > 0:
            mode = "a" if os.path.exists(DEBUG_CSV) else "w"
            df[~df["timestamp_valid"]].to_csv(DEBUG_CSV, mode=mode, header=not os.path.exists(DEBUG_CSV), index=False)
            logger.warning(f"Found {invalid_ts_rows} rows with invalid timestamps in final CSV; saved to {DEBUG_CSV}")
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f UTC", errors="coerce")
        unique_timestamps = len(df["timestamp"].drop_duplicates())
        min_ts = df["timestamp"].min()
        max_ts = df["timestamp"].max()
        
        logger.info(f"Validation (final CSV): {total_rows} rows, {null_rows} nulls, {invalid_rows} invalids, "
                    f"{invalid_ts_rows} invalid timestamps, {unique_timestamps} unique timestamps")
        if null_rows > 0:
            logger.error(f"Found {null_rows} rows with null price/volume in final CSV")
        if invalid_rows > 0:
            logger.error(f"Found {invalid_rows} rows with invalid price/volume in final CSV")
        if invalid_ts_rows > 0:
            logger.error(f"Found {invalid_ts_rows} rows with invalid timestamp format in final CSV")
        if total_rows == 0:
            logger.warning("No data in final CSV")
        elif min_ts and max_ts:
            logger.info(f"Final CSV spans {min_ts} to {max_ts}")
    
    except Exception as e:
        logger.error(f"Failed to validate {final_csv_file}: {str(e)}")

# Process all ticks for a single date and write to temp CSV
def process_date(args: tuple) -> None:
    symbol, date, output_dir = args
    date_str = date.strftime("%Y-%m-%d")
    logger.info(f"Processing {date_str} for {symbol}", extra={"symbol": symbol, "date": date_str})

    if is_weekend(date):
        logger.info(f"Skipping weekend date {date_str} for {symbol}", extra={"symbol": symbol, "date": date_str})
        return

    market_open_est = EST.localize(datetime.combine(
        date.astimezone(EST).date(),
        datetime.min.time().replace(hour=MARKET_OPEN_EST[0], minute=MARKET_OPEN_EST[1])
    ))
    market_open_utc = market_open_est.astimezone(UTC)

    total = 0
    skip = 0
    while True:
        batch_rows = fetch_tick_batch(symbol, market_open_utc, skip, BATCH_SIZE)
        if not batch_rows:
            logger.info(f"Finished fetching {total} ticks for {symbol} on {date_str}", 
                        extra={"symbol": symbol, "date": date_str})
            break
        batch_count = write_to_temp_csv(batch_rows, output_dir, symbol, date_str)
        total += batch_count
        skip += BATCH_SIZE

# Main function to fetch NVDA ticks, save to temp CSV, merge to final CSV, and clean up
def main(start_date_str: str, end_date_str: str, output_dir: str, stream: bool = False):
    global total_rows
    start_time = time.time()

    if stream:
        logger.error("Real-time streaming not yet implemented")
        raise NotImplementedError("Streaming mode requires Finnhub websocket integration")

    try:
        start_date = UTC.localize(datetime.strptime(start_date_str, "%Y-%m-%d"))
        end_date = UTC.localize(datetime.strptime(end_date_str, "%Y-%m-%d"))
    except ValueError as e:
        logger.error(f"Invalid date format: {str(e)}")
        raise

    if start_date > end_date:
        logger.error("Start date must be before end date")
        raise ValueError("Invalid date range")

    logger.info(f"Fetching NVDA ticks from {start_date_str} to {end_date_str}, saving to {output_dir}")

    # Process dates in parallel
    symbol = "NVDA"
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    with Pool(processes=4) as pool:
        pool.map(process_date, [(symbol, date, output_dir) for date in dates])

    # Merge temp CSV to final CSV
    final_rows = merge_temp_to_final_csv(output_dir, symbol)
    total_rows = final_rows

    # Truncate temp CSV
    truncate_temp_csv(output_dir)

    # Clean up daily CSVs
    cleanup_daily_csvs(output_dir, symbol)

    # Validate final CSV
    validate_final_csv(output_dir, symbol, start_date, end_date)

    # Log summary
    total_seconds = time.time() - start_time
    logger.info(f"Total rows in final CSV: {total_rows}")
    logger.info(f"Total runtime: {total_seconds / 3600:.2f} hours")
    logger.info(f"Avg rows/sec: {total_rows / total_seconds:.2f}" if total_seconds > 0 else "Avg rows/sec: 0")
    if failed_dates:
        safe_failed_dates = {f"{k[0]}|{k[1]}": v for k, v in failed_dates.items()}
        logger.warning(f"Failed dates: {json.dumps(safe_failed_dates)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NVDA tick data, save to temp CSV, merge to final CSV")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save CSV files")
    parser.add_argument("--stream", action="store_true", help="Run in real-time streaming mode (not implemented)")
    args = parser.parse_args()
    main(args.start_date, args.end_date, args.output_dir, args.stream)