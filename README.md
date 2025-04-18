# Financial Analytics Pipeline

This project provides a pipeline for processing financial market data, calculating technical indicators, anchored VWAP (Volume-Weighted Average Price), and recent signal indicators across multiple timeframes.

## Project Structure

- `candle_to_calcs.py`: Generates technical indicators and anchored VWAP points from raw candle data.
- `calculate_recent_ta_indicators03.09.2025.py`: Updates recent and trend-based indicator columns in BigQuery.
- `.gitignore`: Excludes sensitive files, data files, and temporary files.
- `README.md`: Project documentation (this file).

## Prerequisites

- **Python**: 3.8+
- **Dependencies**:
  ```bash
  pip install pandas numpy python-ta-lib scipy pytz python-dateutil