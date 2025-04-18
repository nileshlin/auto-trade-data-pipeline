#!/usr/bin/env python3
"""
Usage:
python aggregator_candles.py <start_time> <end_time> <num_symbols> <max_workers>

Example:
python aggregator_candles.py 2023-01-03T09:00:00+00:00 2023-01-03T10:00:00+00:00 5 4

Purpose:
- Aggregates tick data from historical_tick_data_3.csv into 1-second candles.
- Writes to candles_1s.csv instead of BigQuery for testing.
"""

import logging
import sys
import time
import json
import os
from datetime import datetime, timezone
from typing import List, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil import parser as date_parser

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
INPUT_CSV = "../data/historical_tick_data_3.csv"
OUTPUT_DIR = "../data"
TIMEFRAMES = [("1s", 1)]  # Only 1s timeframe for now

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# LOGGING SETUP (JSON Structured)
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "extra": getattr(record, "extra", {})
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(JsonFormatter())
    logger.addHandler(ch)

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def read_and_filter_csv(csv_file: str, start_time: datetime, end_time: datetime, symbols: List[str] = None) -> pd.DataFrame:
    """
    Reads CSV and filters by time range and optional symbol list.
    Returns empty DataFrame if no data is found or an error occurs.
    """
    if not os.path.exists(csv_file):
        logger.error("Input CSV not found", extra={"csv_file": csv_file})
        print(f"[read_and_filter_csv] Error: {csv_file} does not exist")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_file)
        expected_columns = ["symbol", "timestamp", "price", "volume"]
        if not all(col in df.columns for col in expected_columns):
            logger.error("Invalid CSV schema", extra={"csv_file": csv_file, "columns": df.columns.tolist()})
            print(f"[read_and_filter_csv] Error: CSV must have columns {expected_columns}, got {df.columns.tolist()}")
            return pd.DataFrame()
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if df["timestamp"].isna().any():
            logger.error("Invalid timestamps in CSV", extra={"csv_file": csv_file})
            print(f"[read_and_filter_csv] Error: Some timestamps could not be parsed")
            return pd.DataFrame()
        
        df = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]
        if symbols:
            df = df[df["symbol"].isin(symbols)]
        if df.empty:
            logger.info("No data found in CSV after filter", extra={"csv_file": csv_file})
            print(f"[read_and_filter_csv] No data found in {csv_file} for given filters")
        else:
            logger.info("CSV read successfully", extra={"csv_file": csv_file, "rows": len(df)})
            print(f"[read_and_filter_csv] Read {len(df)} rows from {csv_file}")
        return df
    except Exception as e:
        logger.error("Failed to read CSV", extra={"csv_file": csv_file, "error": str(e)})
        print(f"[read_and_filter_csv] Failed to read {csv_file}: {e}")
        return pd.DataFrame()




# Merges temp CSV into main CSV, deduplicating by symbol+timestamp.
def deduplicate_csv(temp_csv: str, main_csv: str) -> None:
    try:
        if not os.path.exists(temp_csv):
            logger.info("Temp CSV does not exist, skipping deduplication", extra={"temp_csv": temp_csv})
            print(f"[deduplicate_csv] Skipping {temp_csv}: does not exist")
            return
        temp_df = pd.read_csv(temp_csv)
        if temp_df.empty:
            logger.info("Temp CSV is empty, skipping deduplication", extra={"temp_csv": temp_csv})
            print(f"[deduplicate_csv] Skipping {temp_csv}: empty")
            return
        temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"], utc=True)
        
        if os.path.exists(main_csv):
            main_df = pd.read_csv(main_csv)
            main_df["timestamp"] = pd.to_datetime(main_df["timestamp"], utc=True)
            combined_df = pd.concat([main_df, temp_df])
        else:
            combined_df = temp_df
        
        combined_df = combined_df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        combined_df["timestamp"] = combined_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
        combined_df.to_csv(main_csv, index=False)
        logger.info("Deduplicated CSV", extra={
            "main_csv": main_csv,
            "rows_written": len(combined_df)
        })
        print(f"[deduplicate_csv] Deduplicated {main_csv}, rows={len(combined_df)}")
        os.remove(temp_csv)
    except Exception as e:
        logger.error("Deduplication failed", extra={"main_csv": main_csv, "error": str(e)})
        print(f"[deduplicate_csv] Failed for {main_csv}: {e}")
        raise

# Validates candle DataFrame: Returns True if valid, False otherwise.
def validate_candles(df: pd.DataFrame, timeframe: str) -> bool:
    if df.empty:
        return True
    required_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "number_of_trades", "vwap"]
    if not all(col in df.columns for col in required_cols):
        logger.error("Missing required columns", extra={"timeframe": timeframe, "columns": df.columns.tolist()})
        print(f"[validate_candles] Missing columns for {timeframe}: {df.columns.tolist()}")
        return False
    if df[required_cols[:-1]].isnull().any().any():  # Exclude vwap (can be null for zero volume)
        logger.error("Null values found in candles", extra={"timeframe": timeframe})
        print(f"[validate_candles] Null values found for {timeframe}")
        return False
    if (df["high"] < df["low"]).any():
        logger.error("Invalid high/low values", extra={"timeframe": timeframe})
        print(f"[validate_candles] high < low for {timeframe}")
        return False
    if (df["volume"] < 0).any():
        logger.error("Negative volume found", extra={"timeframe": timeframe})
        print(f"[validate_candles] Negative volume for {timeframe}")
        return False
    if (df["number_of_trades"] <= 0).any():
        logger.error("Invalid number_of_trades", extra={"timeframe": timeframe})
        print(f"[validate_candles] number_of_trades <= 0 for {timeframe}")
        return False
    return True

def get_row_count_for_symbol(
    df: pd.DataFrame, symbol: str, start_dt: datetime, end_dt: datetime
) -> int:
    """
    Counts rows for a symbol in the DataFrame within the time range.
    Returns -1 if an error occurs.
    """
    try:
        filtered = df[(df["symbol"] == symbol) & (df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
        return len(filtered)
    except Exception as e:
        logger.error("Count failed", extra={"symbol": symbol, "error": str(e)})
        print(f"[get_row_count_for_symbol] Failed count for {symbol}: {e}")
        return -1

# Aggregates tick or candle data into candles for the given timeframe. Returns empty DataFrame if no data or an error occurs.
def aggregate_candles(
    source_df: pd.DataFrame,
    timeframe_seconds: int,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    use_raw: bool
) -> pd.DataFrame:
    try:
        df = source_df[(source_df["symbol"] == symbol) & (source_df["timestamp"] >= start_dt) & (source_df["timestamp"] <= end_dt)]
        if df.empty:
            logger.info("No data for aggregation", extra={"symbol": symbol, "timeframe_seconds": timeframe_seconds})
            print(f"[aggregate_candles] No data for {symbol}, timeframe={timeframe_seconds}s")
            return pd.DataFrame()

        price_col = "price" if use_raw else "close"
        df["bucket_ts"] = df["timestamp"].dt.floor(f"{timeframe_seconds}s")

        # Group by symbol and bucket timestamp
        grouped = df.groupby(["symbol", "bucket_ts"])
        
        # Compute aggregations
        agg_df = pd.DataFrame({
            "symbol": grouped["symbol"].first(),
            "timestamp": grouped["timestamp"].first(),
            "open": grouped[price_col].first(),
            "high": grouped[price_col].max(),
            "low": grouped[price_col].min(),
            "close": grouped[price_col].last(),
            "volume": grouped["volume"].sum(),
            "number_of_trades": grouped.size(),
            "vwap": grouped.apply(lambda x: (x[price_col] * x["volume"]).sum() / x["volume"].sum() if x["volume"].sum() > 0 else None)
        }).reset_index(drop=True)

        logger.info("Aggregation completed", extra={
            "symbol": symbol,
            "timeframe_seconds": timeframe_seconds,
            "input_rows": len(df),
            "output_rows": len(agg_df)
        })
        print(f"[aggregate_candles] Aggregated {symbol}, timeframe={timeframe_seconds}s, input_rows={len(df)}, output_rows={len(agg_df)}")
        return agg_df
    except Exception as e:
        logger.error("Aggregation failed", extra={"symbol": symbol, "timeframe_seconds": timeframe_seconds, "error": str(e)})
        print(f"[aggregate_candles] Failed for {symbol}, timeframe={timeframe_seconds}s: {e}")
        return pd.DataFrame()

def insert_candles_csv(
    source_df: pd.DataFrame,
    temp_csv: str,
    timeframe_label: str,
    timeframe_seconds: int,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    use_raw: bool
) -> Tuple[bool, bool]:
    """
    Aggregates candles and writes to a temporary CSV.
    Returns (success, no_data).
    """
    row_count = get_row_count_for_symbol(source_df, symbol, start_dt, end_dt)
    if row_count == 0:
        logger.info("No data found for symbol in timeframe; skipping", extra={"symbol": symbol, "timeframe": timeframe_label})
        print(f"[insert_candles_csv] Skipping {symbol}, timeframe={timeframe_label}: No data")
        return True, True
    elif row_count < 0:
        logger.warning("Could not determine row count", extra={"symbol": symbol, "timeframe": timeframe_label})
        print(f"[insert_candles_csv] Could not get row count for {symbol}, timeframe={timeframe_label}. Proceeding")

    logger.info("Executing aggregation", extra={
        "symbol": symbol,
        "timeframe": timeframe_label,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "estimated_rows": row_count if row_count >= 0 else "unknown"
    })
    print(f"[insert_candles_csv] Aggregating {symbol}, timeframe={timeframe_label}, from {start_dt.isoformat()} to {end_dt.isoformat()}")

    try:
        agg_df = aggregate_candles(source_df, timeframe_seconds, symbol, start_dt, end_dt, use_raw)
        if agg_df.empty:
            logger.info("No candles generated; skipping write", extra={"symbol": symbol, "timeframe": timeframe_label})
            print(f"[insert_candles_csv] No candles for {symbol}, timeframe={timeframe_label}")
            return True, True

        if not validate_candles(agg_df, timeframe_label):
            logger.error("Candle validation failed", extra={"symbol": symbol, "timeframe": timeframe_label})
            print(f"[insert_candles_csv] Validation failed for {symbol}, timeframe={timeframe_label}")
            return False, False

        agg_df["timestamp"] = agg_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
        agg_df.to_csv(temp_csv, index=False)
        logger.info("Aggregation completed", extra={
            "symbol": symbol,
            "timeframe": timeframe_label,
            "inserted_rows": len(agg_df)
        })
        print(f"[insert_candles_csv] Completed {symbol}, timeframe={timeframe_label}, rows={len(agg_df)}")
        return True, False
    except Exception as e:
        logger.error("Unexpected error during aggregation", extra={
            "symbol": symbol,
            "timeframe": timeframe_label,
            "error": str(e)
        })
        print(f"[insert_candles_csv] Error for {symbol}, timeframe={timeframe_label}: {e}")
        return False, False

# Process symbol (NVDA) for a specific timeframe
def process_symbol_timeframe(
    source_df: pd.DataFrame,
    symbol: str,
    tf_label: str,
    tf_seconds: int,
    start_time: datetime,
    end_time: datetime,
    source_csv: str,
    symbol_index: int,
    total_symbols: int
) -> bool:
    logger.info("Processing symbol timeframe", extra={
        "symbol": symbol,
        "timeframe": tf_label,
        "symbol_index": symbol_index,
        "total_symbols_in_timeframe": total_symbols
    })
    print(f"[process_symbol_timeframe] Symbol={symbol}, timeframe={tf_label}, index={symbol_index}/{total_symbols}")

    temp_csv = os.path.join(OUTPUT_DIR, f"temp_candles_{tf_label}_{symbol}.csv")
    main_csv = os.path.join(OUTPUT_DIR, f"candles_{tf_label}.csv")
    use_raw = (source_csv == INPUT_CSV)

    success, no_data = insert_candles_csv(
        source_df, temp_csv, tf_label, tf_seconds, symbol, start_time, end_time, use_raw
    )
    if success and not no_data:
        try:
            deduplicate_csv(temp_csv, main_csv)
        except Exception as e:
            logger.error("Deduplication failed", extra={"symbol": symbol, "timeframe": tf_label, "error": str(e)})
            print(f"[process_symbol_timeframe] Deduplication failed for {symbol}, timeframe={tf_label}: {e}")
            return False
    if not success:
        logger.error("Symbol processing failed", extra={"symbol": symbol, "timeframe": tf_label})
        print(f"[process_symbol_timeframe] Error for {symbol}, timeframe={tf_label}")
    return success

# Processes all symbols for a timeframe, reading from source CSV and writing to output CSVs.
def process_timeframe(
    symbols: List[str],
    tf_label: str,
    tf_seconds: int,
    start_time: datetime,
    end_time: datetime,
    source_csv: str,
    max_workers: int,
    timeframe_index: int,
    total_timeframes: int
) -> None:
    logger.info("Starting timeframe processing", extra={
        "timeframe": tf_label,
        "timeframe_index": timeframe_index,
        "total_timeframes": total_timeframes,
        "symbol_count": len(symbols),
        "source_csv": source_csv
    })
    print(f"\n[process_timeframe] Starting timeframe={tf_label} ({timeframe_index}/{total_timeframes}), symbol_count={len(symbols)}")

    t0 = time.time()
    source_df = read_and_filter_csv(source_csv, start_time, end_time, symbols)
    if source_df.empty:
        logger.info("No data in source CSV for timeframe", extra={"timeframe": tf_label, "source_csv": source_csv})
        print(f"[process_timeframe] No data in {source_csv} for timeframe={tf_label}")
        return

    total_symbols = len(symbols)
    processed_symbols = 0
    failed_symbols = 0

    if total_symbols == 0:
        logger.info("No symbols to process for timeframe", extra={"timeframe": tf_label})
        print(f"[process_timeframe] No symbols for timeframe={tf_label}")
        return

    effective_workers = min(max_workers, total_symbols)
    logger.info("Timeframe concurrency setup", extra={"timeframe": tf_label, "effective_workers": effective_workers})
    print(f"[process_timeframe] concurrency={effective_workers} (max_workers={max_workers})")

    if effective_workers <= 1:
        logger.info("Running sequential processing", extra={"timeframe": tf_label})
        print(f"[process_timeframe] Sequential loop for timeframe={tf_label}")
        for i, sym in enumerate(symbols, start=1):
            try:
                success = process_symbol_timeframe(
                    source_df, sym, tf_label, tf_seconds, start_time, end_time, source_csv,
                    symbol_index=i, total_symbols=total_symbols
                )
                processed_symbols += 1
                if not success:
                    failed_symbols += 1
                logger.info(
                    "Symbol processed" if success else "Symbol failed",
                    extra={
                        "symbol": sym,
                        "timeframe": tf_label,
                        "symbol_index": i,
                        "processed": processed_symbols,
                        "failed": failed_symbols,
                        "remaining": total_symbols - processed_symbols
                    }
                )
                print(f"[process_timeframe] Symbol {sym} => {'processed' if success else 'failed'} (i={i}/{total_symbols})")
            except Exception as e:
                logger.error("Task failed", extra={"symbol": sym, "timeframe": tf_label, "error": str(e)})
                print(f"[process_timeframe] Failed for {sym}, timeframe={tf_label}: {e}")
                processed_symbols += 1
                failed_symbols += 1
    else:
        logger.info("Running parallel processing", extra={"timeframe": tf_label})
        print(f"[process_timeframe] Parallel processing for timeframe={tf_label}")
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    process_symbol_timeframe,
                    source_df, sym, tf_label, tf_seconds, start_time, end_time, source_csv,
                    i, total_symbols
                ): sym
                for i, sym in enumerate(symbols, start=1)
            }
            for f in as_completed(futures):
                symbol = futures[f]
                try:
                    success = f.result()
                    processed_symbols += 1
                    if not success:
                        failed_symbols += 1
                    logger.info(
                        "Symbol processed" if success else "Symbol failed",
                        extra={
                            "symbol": symbol,
                            "timeframe": tf_label,
                            "processed": processed_symbols,
                            "failed": failed_symbols,
                            "remaining": total_symbols - processed_symbols
                        }
                    )
                    print(f"[process_timeframe] Symbol {symbol} => {'processed' if success else 'failed'} (processed={processed_symbols})")
                except Exception as e:
                    logger.error("Task failed", extra={"symbol": symbol, "timeframe": tf_label, "error": str(e)})
                    print(f"[process_timeframe] Failed for {symbol}, timeframe={tf_label}: {e}")
                    processed_symbols += 1
                    failed_symbols += 1

    duration = time.time() - t0
    logger.info("Completed timeframe", extra={
        "timeframe": tf_label,
        "processed_symbols": processed_symbols,
        "failed_symbols": failed_symbols,
        "duration_sec": round(duration, 2)
    })
    print(f"[process_timeframe] Completed timeframe={tf_label} in {round(duration, 2)}s (processed={processed_symbols}, failed={failed_symbols})")

def main_partial_parallel(
    start_time: datetime,
    end_time: datetime,
    max_symbols: int,
    max_workers: int
) -> None:
    """
    Main pipeline: reads symbols from CSV, processes 1s timeframe, and writes candles to CSVs.
    """
    pipeline_start = time.time()
    
    logger.info("Reading symbols from CSV", extra={"csv_file": INPUT_CSV})
    print(f"[main_partial_parallel] Reading symbols from {INPUT_CSV}")

    try:
        df = read_and_filter_csv(INPUT_CSV, start_time, end_time)
        if df.empty:
            logger.info("No data found in CSV", extra={"csv_file": INPUT_CSV})
            print("[main_partial_parallel] No data found. Exiting.")
            return
        # Hardcode NVDA for testing, as sample CSV contains only NVDA
        all_symbols = ["NVDA"]  # In production, use: sorted(df["symbol"].unique().tolist())[:max_symbols]
        logger.info("Starting pipeline", extra={
            "found_symbols": len(df["symbol"].unique()),
            "total_symbols_used": len(all_symbols),
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        })
        print(f"[main_partial_parallel] Found {len(df['symbol'].unique())} symbol(s), using {len(all_symbols)} => {all_symbols}")
    except Exception as e:
        logger.error("Failed to read symbols", extra={"csv_file": INPUT_CSV, "error": str(e)})
        print(f"[main_partial_parallel] Error reading symbols: {e}")
        raise

    total_timeframes = len(TIMEFRAMES)
    for idx, (tf_label, tf_seconds) in enumerate(TIMEFRAMES, start=1):
        source_csv = INPUT_CSV  # Always use raw ticks for 1s
        process_timeframe(
            all_symbols, tf_label, tf_seconds, start_time, end_time, source_csv,
            max_workers, timeframe_index=idx, total_timeframes=total_timeframes
        )

    total_runtime = round(time.time() - pipeline_start, 2)
    logger.info("Pipeline completed", extra={
        "runtime_seconds": total_runtime,
        "total_symbols_processed": len(all_symbols),
        "timeframes_processed": total_timeframes
    })
    print(f"[main_partial_parallel] Pipeline completed in {total_runtime}s with {len(all_symbols)} symbol(s) and {total_timeframes} timeframe(s).")

# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python aggregator_candles.py <start_time> <end_time> <num_symbols> <max_workers>")
        sys.exit(1)

    start_time_arg, end_time_arg, num_symbols_arg, max_workers_arg = sys.argv[1:5]
    try:
        st_dt = date_parser.parse(start_time_arg)
        et_dt = date_parser.parse(end_time_arg)
        num_symbols = int(num_symbols_arg)
        max_workers = int(max_workers_arg)
        if num_symbols <= 0 or max_workers <= 0:
            raise ValueError("num_symbols and max_workers must be positive")
    except ValueError as e:
        logger.error("Invalid argument", extra={"error": str(e)})
        print(f"[MAIN] Invalid argument: {e}")
        sys.exit(1)

    # Ensure UTC
    if not st_dt.tzinfo:
        st_dt = st_dt.replace(tzinfo=timezone.utc)
    else:
        st_dt = st_dt.astimezone(timezone.utc)

    if not et_dt.tzinfo:
        et_dt = et_dt.replace(tzinfo=timezone.utc)
    else:
        et_dt = et_dt.astimezone(timezone.utc)

    if st_dt >= et_dt:
        logger.error("Start time must be before end time", extra={
            "start": st_dt.isoformat(),
            "end": et_dt.isoformat()
        })
        print("[MAIN] Error: start_time >= end_time.")
        sys.exit(1)

    logger.info("Starting aggregator run", extra={
        "start_arg": st_dt.isoformat(),
        "end_arg": et_dt.isoformat(),
        "max_symbols": num_symbols,
        "max_workers": max_workers
    })
    print(f"[MAIN] Starting aggregator. Range: {st_dt.isoformat()} to {et_dt.isoformat()} | symbols={num_symbols}, workers={max_workers}")

    try:
        main_partial_parallel(st_dt, et_dt, num_symbols, max_workers)
    except Exception as e:
        logger.error("Execution failed", extra={"error": str(e)})
        print(f"[MAIN] Execution failed: {e}")
        sys.exit(1)