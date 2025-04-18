#!/usr/bin/env python3
"""
Processes NVDA 1s candle data from candles_1s.csv to compute technical indicators,
candle patterns, peaks/valleys, and daily anchors. Writes results to CSV files:
candles_1s_calculated.csv and anchored_vwap_points_1s.csv.

Usage:
  python candle_to_calcs_03.01.2025.py <START_TIME> <END_TIME> [--stream]

Example:
  python candle_to_calcs_03.01.2025.py 2023-01-03T04:00:00-05:00 2023-01-03T05:00:00-05:00
  python candle_to_calcs_03.01.2025.py 2023-01-03T04:00:00-05:00 2023-01-03T05:00:00-05:00 --stream
  
"""

import sys
import logging
import json
import time
from datetime import datetime, timezone, timedelta
from dateutil import parser as date_parser
from typing import Dict
import pandas as pd
import numpy as np
import talib
from scipy.signal import find_peaks
import pytz
import os
import argparse

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
SYMBOL = "NVDA"
TIMEFRAME = "1s"
INPUT_CSV = "../data/candles_1s.csv"
CALCULATED_CSV = "../data/candles_1s_calculated.csv"
ANCHOR_CSV = "../data/anchored_vwap_points_1s.csv"
INVALID_CSV = "../data/invalid_candles.csv"

FLUSH_THRESHOLD = 500_000
ROLLING_BUFFER_SIZE = 10000
ANCHOR_FLUSH_INTERVAL_SECS = 300  # 5 minutes
MAX_ANCHORS_PER_TYPE = 5
NY_TZ = pytz.timezone("America/New_York")
DAILY_ANCHOR_TYPES = ["daily_4am", "daily_930", "daily_4pm", "daily_high", "daily_low"]

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
            "symbol": getattr(record, "symbol", SYMBOL),
            "timeframe": getattr(record, "timeframe", TIMEFRAME),
            "extra": getattr(record, "extra", {})
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)

if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(JsonFormatter())
    logger.addHandler(ch)

# ------------------------------------------------------------------------------
# CANDLE VALIDATION
# ------------------------------------------------------------------------------
def validate_candles(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """Validate candles for nulls, invalid OHLC, and timestamp gaps."""
    if df.empty:
        logger.info("Empty candle dataframe", extra={"symbol": symbol, "timeframe": timeframe})
        return df

    initial_rows = len(df)
    invalid_rows = pd.DataFrame()
    required_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "number_of_trades"]

    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error("Missing required columns", extra={"symbol": symbol, "timeframe": timeframe, "missing_cols": missing_cols})
        return pd.DataFrame()

    # Check nulls
    null_rows = df[required_cols[2:]].isna().any(axis=1)  # Exclude symbol, timestamp
    if null_rows.any():
        invalid_rows = pd.concat([invalid_rows, df[null_rows]], ignore_index=True)
        df = df[~null_rows]
        logger.warning("Dropped null candle rows", extra={"symbol": symbol, "timeframe": timeframe, "count": null_rows.sum()})

    # Check invalid OHLC
    invalid_ohlc = (df["high"] < df["low"]) | (df["open"] <= 0) | (df["close"] <= 0)
    if invalid_ohlc.any():
        invalid_rows = pd.concat([invalid_rows, df[invalid_ohlc]], ignore_index=True)
        df = df[~invalid_ohlc]
        logger.warning("Dropped invalid OHLC rows", extra={"symbol": symbol, "timeframe": timeframe, "count": invalid_ohlc.sum()})

    # Check volume and number_of_trades
    invalid_volume = (df["volume"] < 0) | ((df["number_of_trades"] <= 0) & (df["volume"] > 0))
    if invalid_volume.any():
        invalid_rows = pd.concat([invalid_rows, df[invalid_volume]], ignore_index=True)
        df = df[~invalid_volume]
        logger.warning("Dropped invalid volume/number_of_trades rows", extra={"symbol": symbol, "timeframe": timeframe, "count": invalid_volume.sum()})

    # Check timestamp gaps
    df = df.sort_values("timestamp")
    if len(df) > 1:
        gaps = df["timestamp"].diff().dt.total_seconds()
        gap_mask = gaps > 1.5
        gap_count = gap_mask.sum()
        if gap_count > 0:
            max_gap = gaps[gap_mask].max()
            gap_starts = df["timestamp"][gap_mask].tolist()
            logger.warning("Timestamp gaps detected", extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "gap_count": gap_count,
                "max_gap_seconds": max_gap,
                "gap_starts": gap_starts[:5]  # Log up to 5 for brevity
            })

    # Save invalid rows
    if not invalid_rows.empty:
        invalid_rows.to_csv(INVALID_CSV, mode='a', index=False, header=not os.path.exists(INVALID_CSV))
        logger.info("Saved invalid candles", extra={"symbol": symbol, "timeframe": timeframe, "count": len(invalid_rows), "file": INVALID_CSV})

    logger.info("Validated candles", extra={"symbol": symbol, "timeframe": timeframe, "input_rows": initial_rows, "valid_rows": len(df)})
    return df

# ------------------------------------------------------------------------------
# CSV WRITING
# ------------------------------------------------------------------------------
def write_calculated_candles_partial(df: pd.DataFrame, timeframe: str) -> None:
    """Write calculated candles to CSV in chunks."""
    if df.empty:
        logger.info("Empty calculated candles, skipping write", extra={"timeframe": timeframe})
        return

    total_len = len(df)
    logger.info("Writing calculated candles", extra={"file": CALCULATED_CSV, "rows": total_len, "timeframe": timeframe})

    start_idx = 0
    mode = 'a' if os.path.exists(CALCULATED_CSV) else 'w'
    header = not os.path.exists(CALCULATED_CSV)

    while start_idx < total_len:
        end_idx = min(start_idx + FLUSH_THRESHOLD, total_len)
        chunk = df.iloc[start_idx:end_idx].copy()
        chunk.to_csv(CALCULATED_CSV, mode=mode, header=header, index=False)
        logger.info("Wrote calculated candle chunk", extra={"file": CALCULATED_CSV, "start_idx": start_idx, "end_idx": end_idx - 1, "rows": len(chunk)})
        start_idx = end_idx
        mode = 'a'
        header = False

def write_anchored_vwap_points_partial(df: pd.DataFrame, timeframe: str) -> None:
    """Write deduplicated anchor points to CSV."""
    if df.empty:
        return

    df = deduplicate_anchored_vwap_points(df, timeframe)
    total_len = len(df)
    mode = 'a' if os.path.exists(ANCHOR_CSV) else 'w'
    header = not os.path.exists(ANCHOR_CSV)

    start_idx = 0
    while start_idx < total_len:
        end_idx = min(start_idx + FLUSH_THRESHOLD, total_len)
        chunk = df.iloc[start_idx:end_idx].copy()
        chunk.to_csv(ANCHOR_CSV, mode=mode, header=header, index=False)
        logger.info("Wrote anchor rows", extra={"file": ANCHOR_CSV, "start_idx": start_idx, "end_idx": end_idx - 1, "rows": len(chunk)})
        start_idx = end_idx
        mode = 'a'
        header = False

def deduplicate_anchored_vwap_points(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Deduplicate anchor points by symbol, timeframe, anchor_timestamp."""
    if df.empty:
        return df

    initial_rows = len(df)
    df_dedup = df.sort_values("current_snapshot_timestamp").drop_duplicates(
        subset=["symbol", "timeframe", "anchor_timestamp"], keep="last"
    )
    logger.info("Deduplicated anchor points", extra={"file": ANCHOR_CSV, "initial_rows": initial_rows, "dedup_rows": len(df_dedup)})
    return df_dedup

# ------------------------------------------------------------------------------
# CHUNK GENERATOR
# ------------------------------------------------------------------------------
def generate_date_chunks(start_dt: datetime, end_dt: datetime, chunk_days=30):
    curr_start = start_dt
    while curr_start < end_dt:
        curr_end = curr_start + timedelta(days=chunk_days)
        if curr_end > end_dt:
            curr_end = end_dt
        yield curr_start, curr_end
        curr_start = curr_end

# ------------------------------------------------------------------------------
# ANCHOR MANAGEMENT
# ------------------------------------------------------------------------------
def add_anchor(active_anchors: Dict[str, list], anchor_type: str, idx: int, price: float, ts: pd.Timestamp, max_per_type: int = MAX_ANCHORS_PER_TYPE) -> None:
    if anchor_type not in active_anchors:
        active_anchors[anchor_type] = []

    if anchor_type in DAILY_ANCHOR_TYPES:
        active_anchors[anchor_type] = []  # Keep only one daily anchor
    active_anchors[anchor_type].append({
        "anchor_idx": idx,
        "anchor_ts": ts,
        "anchor_price": price
    })

def remove_all_anchors_of_type(active_anchors: Dict[str, list], anchor_type: str) -> None:
    if anchor_type in active_anchors:
        active_anchors[anchor_type].clear()

def compute_anchor_snapshots(df: pd.DataFrame, symbol: str, timeframe: str, active_anchors: Dict[str, list]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    current_idx = len(df) - 1
    current_ts = df["timestamp"].iloc[current_idx]
    snapshots = []

    for anchor_type, anchors_list in active_anchors.items():
        for anchor_info in anchors_list:
            row = {
                "symbol": symbol,
                "timeframe": timeframe,
                "anchor_type": anchor_type,
                "anchor_timestamp": anchor_info["anchor_ts"],
                "anchor_idx": anchor_info["anchor_idx"],
                "price_at_anchor": anchor_info["anchor_price"],
                "current_snapshot_timestamp": current_ts,
                "current_idx": current_idx,
                "anchored_vwap": None
            }
            snapshots.append(row)

    return pd.DataFrame(snapshots)

# ------------------------------------------------------------------------------
# DAILY KEY-LEVEL MANAGEMENT (EST)
# ------------------------------------------------------------------------------
class DailyAnchorState:
    def __init__(self):
        self.local_date = None
        self.daily_4am_added = False
        self.daily_930_added = False
        self.daily_4pm_added = False
        self.daily_high_val = None
        self.daily_high_idx = None
        self.daily_low_val = None
        self.daily_low_idx = None

def reset_daily_anchors(active_anchors: Dict[str, list], daily_state: DailyAnchorState):
    daily_state.daily_4am_added = False
    daily_state.daily_930_added = False
    daily_state.daily_4pm_added = False
    daily_state.daily_high_val = None
    daily_state.daily_high_idx = None
    daily_state.daily_low_val = None
    daily_state.daily_low_idx = None
    for anchor_type in DAILY_ANCHOR_TYPES:
        remove_all_anchors_of_type(active_anchors, anchor_type)

def check_daily_anchors(row_idx: int, row: pd.Series, active_anchors: Dict[str, list], daily_state: DailyAnchorState) -> None:
    local_date = row["local_date"]
    local_hour = row["local_hour"]
    local_minute = row["local_minute"]

    if daily_state.local_date is None or local_date != daily_state.local_date:
        daily_state.local_date = local_date
        reset_daily_anchors(active_anchors, daily_state)
        daily_state.daily_high_val = row["high"]
        daily_state.daily_high_idx = row_idx
        daily_state.daily_low_val = row["low"]
        daily_state.daily_low_idx = row_idx

    if not daily_state.daily_4am_added and (local_hour == 4 and local_minute == 0):
        add_anchor(active_anchors, "daily_4am", row_idx, row["close"], row["timestamp"])
        daily_state.daily_4am_added = True

    if not daily_state.daily_930_added and (local_hour == 9 and local_minute == 30):
        add_anchor(active_anchors, "daily_930", row_idx, row["close"], row["timestamp"])
        daily_state.daily_930_added = True

    if not daily_state.daily_4pm_added and (local_hour == 16 and local_minute == 0):
        add_anchor(active_anchors, "daily_4pm", row_idx, row["close"], row["timestamp"])
        daily_state.daily_4pm_added = True

    if row["high"] > (daily_state.daily_high_val or -float("inf")):
        daily_state.daily_high_val = row["high"]
        daily_state.daily_high_idx = row_idx
        remove_all_anchors_of_type(active_anchors, "daily_high")
        add_anchor(active_anchors, "daily_high", row_idx, row["high"], row["timestamp"])

    if row["low"] < (daily_state.daily_low_val or float("inf")):
        daily_state.daily_low_val = row["low"]
        daily_state.daily_low_idx = row_idx
        remove_all_anchors_of_type(active_anchors, "daily_low")
        add_anchor(active_anchors, "daily_low", row_idx, row["low"], row["timestamp"])

# ------------------------------------------------------------------------------
# INDICATORS & PATTERNS
# ------------------------------------------------------------------------------
def initialize_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Initialize all output columns with default values."""
    columns = [
        "typical_price", "adx", "di_pos", "di_neg", "di_diff", "macd", "macd_signal", "macd_diff",
        "psar", "psar_trend", "psar_reversal", "atr", "atr_norm", "atr_change", "high_volatility",
        "bb_upper", "bb_lower", "bb_mid", "bb_width", "bb_pos", "bb_breakout", "t3", "t3_slope",
        "is_uptrend", "is_downtrend", "is_no_trend", "candle_pattern_sum", "is_volume_spike",
        "rolling_avg_volume", "is_major_peak", "is_major_valley", "is_minor_peak", "is_minor_valley",
        "is_micro_peak", "is_micro_valley", "is_overnight_early", "is_overnight_late",
        "is_early_morning", "is_premarket_early", "is_premarket_morn", "is_morning",
        "is_late_morning", "is_midday", "is_early_afternoon", "is_late_afternoon", "is_closing",
        "is_afterhours"
    ]
    candle_patterns = [
        "CDLDOJI", "CDLHAMMER", "CDLINVERTEDHAMMER", "CDLHANGINGMAN", "CDLSHOOTINGSTAR",
        "CDLMARUBOZU", "CDLLONGLEGGEDDOJI", "CDLDRAGONFLYDOJI", "CDLGRAVESTONEDOJI",
        "CDLTAKURI", "CDLHIGHWAVE", "CDLSPINNINGTOP", "CDLCLOSINGMARUBOZU", "CDLBELTHOLD",
        "CDLRICKSHAWMAN", "CDLSHORTLINE", "CDLLONGLINE", "CDLHARAMI", "CDLENGULFING",
        "CDLPIERCING", "CDLDARKCLOUDCOVER", "CDLKICKING", "CDLKICKINGBYLENGTH",
        "CDLCOUNTERATTACK", "CDLGAPSIDESIDEWHITE", "CDLSEPARATINGLINES", "CDLONNECK",
        "CDLINNECK", "CDLSTALLEDPATTERN", "CDLMATCHINGLOW", "CDLBREAKAWAY", "CDLHARAMICROSS",
        "CDLTHRUSTING", "CDLUNIQUE3RIVER", "CDLHOMINGPIGEON", "CDLTASUKIGAP",
        "CDL3WHITESOLDIERS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDLEVENINGSTAR",
        "CDLMORNINGSTAR", "CDLDOJISTAR", "CDLABANDONEDBABY", "CDLTRISTAR", "CDLADVANCEBLOCK",
        "CDLSTICKSANDWICH", "CDL3STARSINSOUTH", "CDLMORNINGDOJISTAR", "CDLEVENINGDOJISTAR",
        "CDL3LINESTRIKE", "CDL2CROWS", "CDLIDENTICAL3CROWS", "CDLRISEFALL3METHODS",
        "CDLXSIDEGAP3METHODS", "CDLUPSIDEGAP2CROWS", "CDLLADDERBOTTOM",
        "CDLCONCEALBABYSWALL", "CDLHIKKAKEMOD", "CDLMATHOLD"
    ]
    columns.extend(candle_patterns)

    for col in columns:
        if col not in df.columns:
            df[col] = 0.0 if col not in ["is_no_trend"] else 1.0
    return df

def label_session_binary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return initialize_output_columns(df)

    session_cols = [
        "is_overnight_early", "is_overnight_late", "is_early_morning",
        "is_premarket_early", "is_premarket_morn", "is_morning",
        "is_late_morning", "is_midday", "is_early_afternoon",
        "is_late_afternoon", "is_closing", "is_afterhours"
    ]
    df = initialize_output_columns(df)
    hours = df["local_timestamp"].dt.hour
    minutes = df["local_timestamp"].dt.minute

    df["is_overnight_early"] = ((hours >= 0) & (hours < 2)).astype(int)
    df["is_overnight_late"] = ((hours >= 2) & (hours < 4)).astype(int)
    df["is_early_morning"] = ((hours >= 4) & (hours < 8)).astype(int)
    df["is_premarket_early"] = ((hours >= 8) & (hours < 9)).astype(int)
    df["is_premarket_morn"] = ((hours == 9) & (minutes < 30)).astype(int)
    df["is_morning"] = (((hours == 9) & (minutes >= 30)) | (hours == 10)).astype(int)
    df["is_late_morning"] = ((hours == 11) | ((hours == 12) & (minutes < 30))).astype(int)
    df["is_midday"] = (((hours == 12) & (minutes >= 30)) | (hours == 13)).astype(int)
    df["is_early_afternoon"] = ((hours == 14) | ((hours == 15) & (minutes < 30))).astype(int)
    df["is_late_afternoon"] = (((hours == 15) & (minutes >= 30)) | ((hours == 16) & (minutes < 30))).astype(int)
    df["is_closing"] = (((hours == 16) & (minutes >= 30)) | ((hours == 17) & (minutes < 1))).astype(int)
    df["is_afterhours"] = (((hours == 17) & (minutes >= 1)) | (hours >= 18)).astype(int)

    return df

def apply_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return initialize_output_columns(df)

    df = initialize_output_columns(df)
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

    if len(df) >= 14:
        df["adx"] = talib.ADX(df["high"], df["low"], df["close"], 14).fillna(0)
        df["di_pos"] = talib.PLUS_DI(df["high"], df["low"], df["close"], 14).fillna(0)
        df["di_neg"] = talib.MINUS_DI(df["high"], df["low"], df["close"], 14).fillna(0)
        df["di_diff"] = df["di_pos"] - df["di_neg"]
    else:
        df["adx"] = 0
        df["di_pos"] = 0
        df["di_neg"] = 0
        df["di_diff"] = 0

    macd, macd_signal, macd_hist = talib.MACD(df["close"], 12, 26, 9)
    df["macd"] = macd.fillna(0)
    df["macd_signal"] = macd_signal.fillna(0)
    df["macd_diff"] = macd_hist.fillna(0)

    df["psar"] = talib.SAR(df["high"], df["low"], acceleration=0.02, maximum=0.2).fillna(df["close"])
    df["psar_trend"] = (df["close"] > df["psar"]).astype(int)
    df["psar_reversal"] = df["psar_trend"].diff().fillna(0).abs()

    if len(df) >= 14:
        df["atr"] = talib.ATR(df["high"], df["low"], df["close"], 14).fillna(0)
        df["atr_norm"] = df["atr"] / df["close"].replace(0, np.nan).fillna(0)
        df["atr_change"] = df["atr"].diff().fillna(0)
        df["high_volatility"] = (df["atr_norm"] > df["atr_norm"].rolling(14).mean().fillna(0)).astype(int)
    else:
        df["atr"] = 0
        df["atr_norm"] = 0
        df["atr_change"] = 0
        df["high_volatility"] = 0

    bb_upper, bb_mid, bb_lower = talib.BBANDS(df["close"], 20, 2, 2, 0)
    df["bb_upper"] = bb_upper.fillna(df["close"])
    df["bb_lower"] = bb_lower.fillna(df["close"])
    df["bb_mid"] = bb_mid.fillna(df["close"])
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan).fillna(0)
    df["bb_breakout"] = ((df["close"] > df["bb_upper"]) | (df["close"] < df["bb_lower"])).astype(int)

    return df

def calculate_t3_slope(df: pd.DataFrame, periods: int = 60) -> pd.DataFrame:
    if df.empty or len(df) < periods:
        df = initialize_output_columns(df)
        df["t3"] = df["close"]
        df["t3_slope"] = 0
        return df

    df["t3"] = talib.T3(df["close"], timeperiod=periods).fillna(df["close"])
    df["t3_slope"] = df["t3"].diff(periods).fillna(0)
    return df

def label_t3_trend(df: pd.DataFrame, slope_threshold: float = 0.2) -> pd.DataFrame:
    if df.empty or "t3_slope" not in df.columns:
        df = initialize_output_columns(df)
        df["is_uptrend"] = 0
        df["is_downtrend"] = 0
        df["is_no_trend"] = 1
        return df

    slope = df["t3_slope"]
    df["is_uptrend"] = (slope > slope_threshold).astype(int)
    df["is_downtrend"] = (slope < -slope_threshold).astype(int)
    df["is_no_trend"] = (~(slope > slope_threshold) & ~(slope < -slope_threshold)).astype(int)
    return df

def single_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return initialize_output_columns(df)

    patterns = [
        "CDLDOJI", "CDLHAMMER", "CDLINVERTEDHAMMER", "CDLHANGINGMAN", "CDLSHOOTINGSTAR",
        "CDLMARUBOZU", "CDLLONGLEGGEDDOJI", "CDLDRAGONFLYDOJI", "CDLGRAVESTONEDOJI",
        "CDLTAKURI", "CDLHIGHWAVE", "CDLSPINNINGTOP", "CDLCLOSINGMARUBOZU", "CDLBELTHOLD",
        "CDLRICKSHAWMAN", "CDLSHORTLINE", "CDLLONGLINE"
    ]
    for p in patterns:
        df[p] = getattr(talib, p)(df["open"], df["high"], df["low"], df["close"]).fillna(0)
    return df

def two_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return initialize_output_columns(df)

    patterns = [
        "CDLHARAMI", "CDLENGULFING", "CDLPIERCING", "CDLDARKCLOUDCOVER", "CDLKICKING",
        "CDLKICKINGBYLENGTH", "CDLCOUNTERATTACK", "CDLGAPSIDESIDEWHITE", "CDLSEPARATINGLINES",
        "CDLONNECK", "CDLINNECK", "CDLSTALLEDPATTERN", "CDLMATCHINGLOW", "CDLBREAKAWAY",
        "CDLHARAMICROSS", "CDLTHRUSTING", "CDLUNIQUE3RIVER", "CDLHOMINGPIGEON", "CDLTASUKIGAP"
    ]
    for p in patterns:
        df[p] = getattr(talib, p)(df["open"], df["high"], df["low"], df["close"]).fillna(0)
    return df

def three_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return initialize_output_columns(df)

    patterns = [
        "CDL3WHITESOLDIERS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDLEVENINGSTAR",
        "CDLMORNINGSTAR", "CDLDOJISTAR", "CDLABANDONEDBABY", "CDLTRISTAR",
        "CDLADVANCEBLOCK", "CDLSTICKSANDWICH", "CDL3STARSINSOUTH",
        "CDLMORNINGDOJISTAR", "CDLEVENINGDOJISTAR", "CDL3LINESTRIKE",
        "CDL2CROWS", "CDLIDENTICAL3CROWS"
    ]
    for p in patterns:
        df[p] = getattr(talib, p)(df["open"], df["high"], df["low"], df["close"]).fillna(0)
    return df

def multi_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return initialize_output_columns(df)

    patterns = [
        "CDLRISEFALL3METHODS", "CDLXSIDEGAP3METHODS", "CDLUPSIDEGAP2CROWS",
        "CDLLADDERBOTTOM", "CDLCONCEALBABYSWALL", "CDLHIKKAKEMOD", "CDLMATHOLD"
    ]
    for p in patterns:
        df[p] = getattr(talib, p)(df["open"], df["high"], df["low"], df["close"]).fillna(0)
    return df

def calculate_candle_pattern_sum(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return initialize_output_columns(df)

    cdl_cols = [c for c in df.columns if c.startswith("CDL")]
    df["candle_pattern_sum"] = df[cdl_cols].fillna(0).sum(axis=1)
    return df

def detect_volume_spikes(df: pd.DataFrame, window=60, spike_multiplier=1.5) -> pd.DataFrame:
    if df.empty:
        df = initialize_output_columns(df)
        df["rolling_avg_volume"] = 0
        df["is_volume_spike"] = 0
        return df

    df["rolling_avg_volume"] = df["volume"].rolling(window=window, min_periods=1).mean().fillna(0)
    df["is_volume_spike"] = (df["volume"] > df["rolling_avg_volume"] * spike_multiplier).astype(int)
    return df

def label_peaks_valleys_multi(df: pd.DataFrame, start_idx: int) -> pd.DataFrame:
    if df.empty:
        return initialize_output_columns(df)

    df = initialize_output_columns(df)
    if start_idx >= len(df):
        return df

    df_new = df.iloc[start_idx:].copy()
    major_peaks, _ = find_peaks(df_new["high"], distance=10, prominence=0.9)
    major_valleys, _ = find_peaks(-df_new["low"], distance=10, prominence=0.9)
    minor_peaks, _ = find_peaks(df_new["high"], distance=7, prominence=0.7)
    minor_valleys, _ = find_peaks(-df_new["low"], distance=7, prominence=0.7)
    micro_peaks, _ = find_peaks(df_new["high"], distance=5, prominence=0.5)
    micro_valleys, _ = find_peaks(-df_new["low"], distance=5, prominence=0.5)

    df.iloc[start_idx:, df.columns.get_loc("is_major_peak")] = 0
    df.iloc[start_idx:, df.columns.get_loc("is_major_valley")] = 0
    df.iloc[start_idx:, df.columns.get_loc("is_minor_peak")] = 0
    df.iloc[start_idx:, df.columns.get_loc("is_minor_valley")] = 0
    df.iloc[start_idx:, df.columns.get_loc("is_micro_peak")] = 0
    df.iloc[start_idx:, df.columns.get_loc("is_micro_valley")] = 0

    df.iloc[start_idx + major_peaks, df.columns.get_loc("is_major_peak")] = 1
    df.iloc[start_idx + major_valleys, df.columns.get_loc("is_major_valley")] = 1
    df.iloc[start_idx + minor_peaks, df.columns.get_loc("is_minor_peak")] = 1
    df.iloc[start_idx + minor_valleys, df.columns.get_loc("is_minor_valley")] = 1
    df.iloc[start_idx + micro_peaks, df.columns.get_loc("is_micro_peak")] = 1
    df.iloc[start_idx + micro_valleys, df.columns.get_loc("is_micro_valley")] = 1

    return df

def apply_all_calculations(df: pd.DataFrame, start_idx: int) -> pd.DataFrame:
    if df.empty:
        return initialize_output_columns(df)

    df = label_session_binary(df)
    df = calculate_t3_slope(df, periods=60)
    df = apply_ta_indicators(df)
    df = single_candle_patterns(df)
    df = two_candle_patterns(df)
    df = three_candle_patterns(df)
    df = multi_candle_patterns(df)
    df = calculate_candle_pattern_sum(df)
    df = label_t3_trend(df, slope_threshold=0.2)
    df = detect_volume_spikes(df, window=60, spike_multiplier=1.5)
    df = label_peaks_valleys_multi(df, start_idx)
    return df

# ------------------------------------------------------------------------------
# PROCESSING LOGIC
# ------------------------------------------------------------------------------
def process_symbol_timeframe_logic(symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime) -> None:
    """Process NVDA 1s candles from CSV, compute indicators, and write to CSV."""
    active_anchors = {}
    daily_state = DailyAnchorState()
    last_flush_time = time.time()
    anchor_vwap_buffer = []
    flush_count = 0
    rolling_buffer = pd.DataFrame()
    ta_time = 0
    rows_processed = 0

    # Read input CSV
    try:
        df_all = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"])
        df_all = df_all[df_all["symbol"] == symbol]
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True, errors='coerce')
        if df_all["timestamp"].isna().any():
            logger.error("Invalid timestamps in input CSV", extra={"file": INPUT_CSV})
            return
        df_all = df_all[(df_all["timestamp"] >= start_dt) & (df_all["timestamp"] < end_dt)]
        df_all = df_all.sort_values("timestamp")
    except FileNotFoundError:
        logger.error("Input CSV not found", extra={"file": INPUT_CSV})
        return
    except pd.errors.ParserError:
        logger.error("Failed to parse input CSV", extra={"file": INPUT_CSV})
        return
    except Exception as e:
        logger.error("Failed to read input CSV", extra={"file": INPUT_CSV, "error": str(e)})
        return

    if df_all.empty:
        logger.info("No data in CSV for range", extra={"symbol": symbol, "timeframe": timeframe, "start_dt": start_dt, "end_dt": end_dt})
        return

    logger.info("Input data stats", extra={
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": len(df_all),
        "start_time": df_all["timestamp"].min(),
        "end_time": df_all["timestamp"].max()
    })

    # Clear output CSVs
    for csv_file in [CALCULATED_CSV, ANCHOR_CSV]:
        if os.path.exists(csv_file):
            os.remove(csv_file)

    chunk_ranges = list(generate_date_chunks(start_dt, end_dt, chunk_days=30))
    if not chunk_ranges:
        logger.info("No chunk ranges", extra={"symbol": symbol, "timeframe": timeframe})
        return

    for idx, (chunk_start, chunk_end) in enumerate(chunk_ranges):
        df_new = df_all[(df_all["timestamp"] >= chunk_start) & (df_all["timestamp"] < chunk_end)]
        if df_new.empty and rolling_buffer.empty:
            continue

        df_new = validate_candles(df_new, symbol, timeframe)
        if df_new.empty:
            continue

        df_new["local_timestamp"] = df_new["timestamp"].dt.tz_convert(NY_TZ)
        df_new["local_date"] = df_new["local_timestamp"].dt.date
        df_new["local_hour"] = df_new["local_timestamp"].dt.hour
        df_new["local_minute"] = df_new["local_timestamp"].dt.minute

        combined = pd.concat([rolling_buffer, df_new], ignore_index=True) if not rolling_buffer.empty else df_new.copy()
        if combined.empty:
            continue

        start_idx = len(rolling_buffer) if not rolling_buffer.empty else 0
        ta_start = time.time()
        combined = apply_all_calculations(combined, start_idx)
        ta_time += time.time() - ta_start
        rows_processed += len(combined) - start_idx

        for row_idx in range(start_idx, len(combined)):
            row = combined.iloc[row_idx]
            check_daily_anchors(row_idx, row, active_anchors, daily_state)
            for anchor_type in ["micro_peak", "micro_valley", "minor_peak", "minor_valley", "major_peak", "major_valley"]:
                if row.get(f"is_{anchor_type}", 0) == 1:
                    add_anchor(active_anchors, anchor_type, row_idx, row["high" if "peak" in anchor_type else "low"], row["timestamp"])

            elapsed = time.time() - last_flush_time
            if elapsed >= ANCHOR_FLUSH_INTERVAL_SECS:
                snapshot_df = compute_anchor_snapshots(combined, symbol, timeframe, active_anchors)
                if not snapshot_df.empty:
                    anchor_vwap_buffer.append(snapshot_df)
                if anchor_vwap_buffer:
                    to_write = pd.concat(anchor_vwap_buffer, ignore_index=True)
                    write_anchored_vwap_points_partial(to_write, timeframe)
                    flush_count += 1
                    anchor_vwap_buffer = []
                    active_anchors.clear()
                last_flush_time = time.time()

        snapshot_df = compute_anchor_snapshots(combined, symbol, timeframe, active_anchors)
        if not snapshot_df.empty:
            anchor_vwap_buffer.append(snapshot_df)

        if anchor_vwap_buffer:
            final_to_write = pd.concat(anchor_vwap_buffer, ignore_index=True)
            write_anchored_vwap_points_partial(final_to_write, timeframe)
            flush_count += 1
            anchor_vwap_buffer = []
            active_anchors.clear()

        new_data_slice = combined.iloc[start_idx:].copy()
        write_calculated_candles_partial(new_data_slice, timeframe)

        rolling_buffer = combined.iloc[-ROLLING_BUFFER_SIZE:].copy() if len(combined) > ROLLING_BUFFER_SIZE else combined.copy()
        del combined, df_new  # Free memory

    logger.info("Completed processing", extra={
        "symbol": symbol,
        "timeframe": timeframe,
        "flushes": flush_count,
        "ta_time": round(ta_time, 2),
        "rows_processed": rows_processed
    })

def stream_candles(symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime):
    """Simulate streaming by processing CSV in 1-minute chunks."""
    buffer = pd.DataFrame()
    active_anchors = {}
    daily_state = DailyAnchorState()
    anchor_vwap_buffer = []
    last_flush = time.time()
    flush_count = 0
    ta_time = 0
    rows_processed = 0

    try:
        df_all = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"])
        df_all = df_all[df_all["symbol"] == symbol]
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True, errors='coerce')
        if df_all["timestamp"].isna().any():
            logger.error("Invalid timestamps in input CSV", extra={"file": INPUT_CSV})
            return
        df_all = df_all[(df_all["timestamp"] >= start_dt) & (df_all["timestamp"] < end_dt)]
        df_all = df_all.sort_values("timestamp")
    except FileNotFoundError:
        logger.error("Input CSV not found", extra={"file": INPUT_CSV})
        return
    except pd.errors.ParserError:
        logger.error("Failed to parse input CSV", extra={"file": INPUT_CSV})
        return
    except Exception as e:
        logger.error("Failed to read input CSV", extra={"file": INPUT_CSV, "error": str(e)})
        return

    if df_all.empty:
        logger.info("No data in CSV for range", extra={"symbol": symbol, "timeframe": timeframe})
        return

    logger.info("Input data stats", extra={
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": len(df_all),
        "start_time": df_all["timestamp"].min(),
        "end_time": df_all["timestamp"].max()
    })

    # Clear output CSVs
    for csv_file in [CALCULATED_CSV, ANCHOR_CSV]:
        if os.path.exists(csv_file):
            os.remove(csv_file)

    current_time = start_dt
    while current_time < end_dt:
        next_time = current_time + timedelta(minutes=1)
        if next_time > end_dt:
            next_time = end_dt

        df_new = df_all[(df_all["timestamp"] >= current_time) & (df_all["timestamp"] < next_time)]
        if df_new.empty:
            current_time = next_time
            continue

        df_new = validate_candles(df_new, symbol, timeframe)
        if df_new.empty:
            current_time = next_time
            continue

        df_new["local_timestamp"] = df_new["timestamp"].dt.tz_convert(NY_TZ)
        df_new["local_date"] = df_new["local_timestamp"].dt.date
        df_new["local_hour"] = df_new["local_timestamp"].dt.hour
        df_new["local_minute"] = df_new["local_timestamp"].dt.minute

        buffer = pd.concat([buffer, df_new], ignore_index=True)
        if len(buffer) > ROLLING_BUFFER_SIZE:
            buffer = buffer.iloc[-ROLLING_BUFFER_SIZE:]

        start_idx = len(buffer) - len(df_new)
        combined = buffer.copy()
        ta_start = time.time()
        combined = apply_all_calculations(combined, start_idx)
        ta_time += time.time() - ta_start
        rows_processed += len(combined) - start_idx

        for row_idx in range(start_idx, len(combined)):
            row = combined.iloc[row_idx]
            check_daily_anchors(row_idx, row, active_anchors, daily_state)
            for anchor_type in ["micro_peak", "micro_valley", "minor_peak", "minor_valley", "major_peak", "major_valley"]:
                if row.get(f"is_{anchor_type}", 0) == 1:
                    add_anchor(active_anchors, anchor_type, row_idx, row["high" if "peak" in anchor_type else "low"], row["timestamp"])

        if time.time() - last_flush >= ANCHOR_FLUSH_INTERVAL_SECS:
            snapshot_df = compute_anchor_snapshots(combined, symbol, timeframe, active_anchors)
            if not snapshot_df.empty:
                anchor_vwap_buffer.append(snapshot_df)
            if anchor_vwap_buffer:
                to_write = pd.concat(anchor_vwap_buffer, ignore_index=True)
                write_anchored_vwap_points_partial(to_write, timeframe)
                flush_count += 1
                anchor_vwap_buffer = []
                active_anchors.clear()
            write_calculated_candles_partial(combined.iloc[start_idx:], timeframe)
            last_flush = time.time()

        current_time = next_time
        time.sleep(0.1)  # Simulate real-time delay
        del combined, df_new  # Free memory

    # Final flush
    if not buffer.empty:
        combined = buffer.copy()
        start_idx = 0
        ta_start = time.time()
        combined = apply_all_calculations(combined, start_idx)
        ta_time += time.time() - ta_start
        rows_processed += len(combined)
        snapshot_df = compute_anchor_snapshots(combined, symbol, timeframe, active_anchors)
        if not snapshot_df.empty:
            anchor_vwap_buffer.append(snapshot_df)
        if anchor_vwap_buffer:
            to_write = pd.concat(anchor_vwap_buffer, ignore_index=True)
            write_anchored_vwap_points_partial(to_write, timeframe)
            flush_count += 1
        write_calculated_candles_partial(combined, timeframe)
        del combined

    logger.info("Streaming stopped", extra={
        "symbol": symbol,
        "timeframe": timeframe,
        "flushes": flush_count,
        "ta_time": round(ta_time, 2),
        "rows_processed": rows_processed
    })

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Calculate indicators and anchors for NVDA 1s candles.")
    parser.add_argument("start_time", help="Start time (e.g., 2023-01-03T04:00:00-05:00)")
    parser.add_argument("end_time", help="End time (e.g., 2023-01-03T05:00:00-05:00)")
    parser.add_argument("--stream", action="store_true", help="Run in streaming mode")
    args = parser.parse_args()

    try:
        start_dt = date_parser.parse(args.start_time).astimezone(timezone.utc)
        end_dt = date_parser.parse(args.end_time).astimezone(timezone.utc)
    except ValueError as ve:
        logger.error("Invalid date format", extra={"error": str(ve)})
        sys.exit(1)

    if start_dt >= end_dt:
        logger.error("Start time must be before end time")
        sys.exit(1)

    logger.info("Starting calculations", extra={
        "start_dt": start_dt.isoformat(),
        "end_dt": end_dt.isoformat(),
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "stream": args.stream
    })

    if args.stream:
        stream_candles(SYMBOL, TIMEFRAME, start_dt, end_dt)
    else:
        process_symbol_timeframe_logic(SYMBOL, TIMEFRAME, start_dt, end_dt)

    logger.info("Processing completed")

if __name__ == "__main__":
    main()