import pandas as pd
from pathlib import Path
import time
import pyarrow as pa
import pyarrow.parquet as pq
import os
import logging; logger = logging.getLogger(__name__)
from factor_research.data.prices import fetch_ticker_prices, _cache_check, validate_ticker_data, append_to_ledger, ValidationResult


def test_fetch_ticker_prices_smoke_test_aapl(tmp_path):
    # Fetch ticker prices df.
    df = fetch_ticker_prices(ticker='AAPL', cache_dir=tmp_path)

    # df must be non-empty.
    if df.empty:
        raise ValueError("df is empty.")
    
    # df has expected columns.
    expected_columns = {'open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividends', 'stock_splits'}
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"Missing columns: {expected_columns - set(df.columns)}")
    
    # Index is DatetimeIndex named 'date', timezone-naive.
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not pd.DatetimeIndex.")
    if not df.index.name == 'date':
        raise ValueError("Index name is not 'date'.")
    if df.index.tz is not None :
        raise ValueError("Index is not timezone-naive.")

    # Index is sorted ascending.
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index is not sorted ascending.")

    # Earliest date is on or before 1985
    if df.index[0] >= pd.Timestamp('1986-1-1'):
        raise ValueError("Earliest date for 'AAPL' is not on or before 1985.")
    
    # All prices are positive. 
    cols_to_check = ['open', 'high', 'low', 'close', 'adj_close']
    has_negative = (df[cols_to_check] < 0).any().any()
    if has_negative:
        raise ValueError("Not all prices are positive.")
    

def test_fetch_ticker_prices_uses_cache(tmp_path):
    # Call fetch_ticker_prices.
    df = fetch_ticker_prices(ticker='AAPL', cache_dir=tmp_path)
    
    # Parquet file exists in cache.
    file_path = Path(tmp_path / 'AAPL.parquet')

    if not file_path.is_file():
        raise ValueError(f"{file_path} does not exist.")
    
    # Record file's mtime.
    mtime_file_path = file_path.stat().st_mtime

    # Sleep for 0.1s for clock resolution.
    time.sleep(0.1)

    # Call fetch_ticker_prices again.
    df = fetch_ticker_prices(ticker='AAPL', cache_dir=tmp_path)

    # File's mtime is unchanged.
    if mtime_file_path != file_path.stat().st_mtime:
        raise ValueError("File's mtime has changed; fetch_ticker_prices may have not loaded file in cache.")
    
    # After round-trip, ensure df structure is sound.
    # df has expected columns.
    expected_columns = {'open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividends', 'stock_splits'}

    if not expected_columns.issubset(df.columns):
        raise ValueError(f"Missing columns: {expected_columns - set(df.columns)}")
    
    # Index is DatetimeIndex named 'date', timezone-naive.
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not pd.DatetimeIndex.")
    if not df.index.name == 'date':
        raise ValueError("Index name is not 'date'.")
    if df.index.tz is not None :
        raise ValueError("Index is not timezone-naive.")
    

def test_fetch_ticker_prices_force_refresh(tmp_path):
    # Call fetch_ticker_prices. 
    fetch_ticker_prices(ticker='AAPL', cache_dir=tmp_path)
    
    # Parquet file exists in cache.
    file_path = Path(tmp_path / 'AAPL.parquet')
    if not file_path.is_file():
        raise ValueError(f"{file_path} does not exist.")
    
    # Record file's mtime.
    mtime_file_path = file_path.stat().st_mtime

    # Sleep for 0.1s for clock resolution.
    time.sleep(0.1)

    # Call fetch_ticker_prices again with force_refresh=True.
    fetch_ticker_prices(ticker='AAPL', cache_dir=tmp_path, force_refresh=True)

    # File's mtime has changed.
    if mtime_file_path == file_path.stat().st_mtime:
        raise ValueError("File's mtime has not changed; fetch_ticker_prices may have bypassed force_refresh.")                                        


def test_fetch_ticker_prices_corruption_recovery(tmp_path):
    # Create corrupt file to tmp_path.
    corrupt_file = tmp_path / "AAPL.parquet"
    corrupt_file.write_bytes(b"this file is corrup :(")

    # Return value is None.
    return_value = _cache_check(tmp_path, 'AAPL', 1)

    if return_value is not None:
        raise ValueError("Corrupt file was not detected.")
    

def test_ttl_plumbing_fetch_ticker_prices(tmp_path):
    # Write a valid parquet file to tmp_path using artificial schema and table.
    schema = pa.schema([
        ('id', pa.int64()),
        ('name', pa.string()),
        ('is_active', pa.bool_())
    ])

    table = pa.Table.from_batches([], schema=schema)
    file_path = tmp_path / "AAPL.parquet"
    pq.write_table(table, file_path)

    # Fetch timestamps from parquet file.
    file_stats = os.stat(file_path)
    current_atime = file_stats.st_atime

    # Set parquet file to have artificially-old mtime (as a Unix timestamp).
    fake_mtime = -22025970
    os.utime(file_path, (current_atime, fake_mtime))

    # Refetch data.
    fetch_ticker_prices('AAPL', cache_dir=tmp_path)

    # Verify that new parquet file with today's date exists. 
    # To assume the times are equal within a reasonable threshold, we 
    # simply check if mtimes are equal within 1 second.
    file_path = tmp_path / "AAPL.parquet"
    file_mtime = os.path.getmtime(file_path)
    tolerance = 1
    current_mtime = time.time()

    if abs(file_mtime - current_mtime) > tolerance:
        raise ValueError("File mtime is invalid.")


def test_validate_ticker_data_empty():
    # Initialize ValidationResult object.
    ticker = 'AAPL'
    prices = pd.DataFrame({})
    result = validate_ticker_data(ticker, prices, pd.NaT, pd.NaT)

    # Verify status.
    if result.status != 'EMPTY':
        raise ValueError("status should be 'EMPTY'.")
    

def test_validate_ticker_data_ticker_reuse():
    # Initialize ValidationResult object.
    ticker = 'AAPL'
    data = {
        'open': [1,2,3],
        'high': [4,5,6],
        'low': [1,2,3],
        'close': [1,2,3],
        'adj_close': [6,7,8],
        'volume': [21,91,420],
        'dividends': [1,2,3],
        'stock_splits': [0,3,1]
    }
    prices = pd.DataFrame(data)
    prices_index = [
        pd.Timestamp('2014-01-01'),
        pd.Timestamp('2014-01-02'),
        pd.Timestamp('2014-01-03')
    ]
    prices.index = prices_index
    spell_end = pd.Timestamp('2008-09-29')
    result = validate_ticker_data(ticker, prices, pd.NaT, spell_end)

    # Verify status.
    if result.status != 'TICKER_REUSE':
        raise ValueError("status should be 'TICKER_REUSE'.")


def test_validate_ticker_data_partial_history():
    # Initialize ValidationResult object.
    ticker = 'AAPL'
    data = {
        'open': [1,2,3],
        'high': [4,5,6],
        'low': [1,2,3],
        'close': [1,2,3],
        'adj_close': [6,7,8],
        'volume': [21,91,420],
        'dividends': [1,2,3],
        'stock_splits': [0,3,1]
    }
    prices = pd.DataFrame(data)
    prices_index = [
        pd.Timestamp('2010-01-01'),
        pd.Timestamp('2010-01-02'),
        pd.Timestamp('2010-01-03')
    ]
    prices.index = prices_index
    spell_start = pd.Timestamp('2005-01-01')
    result = validate_ticker_data(ticker, prices, spell_start, pd.NaT)

    # Verify status.
    if result.status != 'PARTIAL_HISTORY':
        raise ValueError("status should be 'PARTIAL_HISTORY'.")
    

def test_validate_ticker_data_happy_path():
    # Initialize ValidationResult object.
    # Embed 1985 start date into index of prices df.
    ticker = 'AAPL'
    data = {
        'open': [1,2,3],
        'high': [4,5,6],
        'low': [1,2,3],
        'close': [1,2,3],
        'adj_close': [6,7,8],
        'volume': [21,91,420],
        'dividends': [1,2,3],
        'stock_splits': [0,3,1]
    }
    prices = pd.DataFrame(data)
    prices_index = [
        pd.Timestamp('1985-01-01'),
        pd.Timestamp('1985-01-02'),
        pd.Timestamp('1985-01-03')
    ]
    prices.index = prices_index
    spell_start = pd.Timestamp('1990-01-01')
    result = validate_ticker_data(ticker, prices, spell_start, pd.NaT)

    # Verify status.
    if result.status != 'OK':
        raise ValueError("status should be 'OK'.")
    

def test_validate_ticker_data_NaT_spell_start():
    # Initialize ValidationResult object.
    ticker = 'AAPL'
    data = {
        'open': [1,2,3],
        'high': [4,5,6],
        'low': [1,2,3],
        'close': [1,2,3],
        'adj_close': [6,7,8],
        'volume': [21,91,420],
        'dividends': [1,2,3],
        'stock_splits': [0,3,1]
    }
    prices = pd.DataFrame(data)
    prices_index = [
        pd.Timestamp('1985-01-01'),
        pd.Timestamp('1985-01-02'),
        pd.Timestamp('1985-01-03')
    ]
    prices.index = prices_index
    spell_start = pd.NaT
    result = validate_ticker_data(ticker, prices, spell_start, pd.NaT)

    # Verify status.
    if result.status != 'OK':
        raise ValueError("status should be 'OK'.")
    

def test_validate_ticker_data_tolerance_boundary():
    # Initialize ValidationResult object.
    # prices index date starts exactly tolerance_days + 1 after spell_start.
    ticker = 'AAPL'
    data = {
        'open': [1,2,3],
        'high': [4,5,6],
        'low': [1,2,3],
        'close': [1,2,3],
        'adj_close': [6,7,8],
        'volume': [21,91,420],
        'dividends': [1,2,3],
        'stock_splits': [0,3,1]
    }
    prices_index = [
        pd.Timestamp('2020-1-18'),
        pd.Timestamp('2020-1-19'),
        pd.Timestamp('2020-1-20')
    ]
    prices = pd.DataFrame(data)
    prices.index = prices_index
    spell_start = pd.Timestamp('2020-01-10')
    tolerance_days = 7
    result = validate_ticker_data(ticker, prices, spell_start, pd.NaT, tolerance_days=tolerance_days)

    # Verify status.
    if result.status != 'PARTIAL_HISTORY':
        raise ValueError("status should be 'PARTIAL_HISTORY'.")
    
    # Now, prices index data starts exactly tolerance_days - 1 after spell_start.
    spell_start = pd.Timestamp('2020-1-12')
    result = validate_ticker_data(ticker, prices, spell_start, pd.NaT, tolerance_days=tolerance_days)
    
    # Verify status.
    if result.status != 'OK':
        raise ValueError("status should be 'OK'.")
    

def test_append_to_ledger_creation(tmp_path):
    # Initialize ValidationResult object.
    yf_start=pd.Timestamp('2020-01-01')
    yf_end = pd.NaT
    result = ValidationResult(
        'OK',
        'Price history appears to be correct.',
        yf_start,
        yf_end, 
        1
    )

    # Call append_to_ledger.
    append_to_ledger(tmp_path, 'AAPL', yf_start, yf_end, result)

    # File exists.
    file_path = tmp_path / '_ledger.parquet'
    if not file_path.exists():
        raise ValueError("Ledger was not created to expected directory.")
    
    # Ledger has only one row.
    ledger = pd.read_parquet(file_path)
    if len(ledger) != 1:
        raise ValueError("Ledger should have one row.")
    
    # Ledger has correct schema.
    expected_columns = {
        'ticker',
        'attempted_at',
        'spell_start',
        'spell_end',
        'yf_start',
        'yf_end',
        'status',
        'reason',
        'n_rows'
    }
    if not expected_columns.issubset(ledger.columns):
        raise ValueError("Ledger schema is invalid.")


def test_append_to_ledger_existing(tmp_path):
    # Write one entry.
    yf_start=pd.Timestamp('2020-01-01')
    yf_end = pd.NaT
    result = ValidationResult(
        'OK',
        'Price history appears to be correct.',
        yf_start,
        yf_end, 
        1
    )
    append_to_ledger(tmp_path, 'AAPL', yf_start, yf_end, result)

    # Write another entry. Ensure second call happens after the first.
    time.sleep(0.001)
    append_to_ledger(tmp_path, 'AAPL', yf_start, yf_end, result) 

    # Ledger has 2 rows.
    ledger_file_path = tmp_path / '_ledger.parquet'
    ledger = pd.read_parquet(ledger_file_path)
    if len(ledger) != 2:
        raise ValueError("Ledger should have two rows.")
    
    # Ledger rows are in chronological order.
    if not (ledger['attempted_at'].iloc[0] < ledger['attempted_at'].iloc[1]):
        raise ValueError("Ledger rows are not in chronological order.")
    
    # Data types in ledger should not be tuples.
    if ledger['yf_start'].iloc[0] != pd.Timestamp('2020-01-01'):
        raise ValueError("Data types in row may be tuples.")


def test_append_to_ledger_corruption_recovery(tmp_path):
    # Create corrupt parquet file.
    corrupt_file = tmp_path / '_ledger.parquet'
    corrupt_file.write_bytes(b"My head hearts a litle bit :/")

    # Call append_to_ledger.
    ticker = 'AAPL'
    yf_start=pd.Timestamp('2020-01-01')
    yf_end = pd.NaT
    result = ValidationResult(
        'OK',
        'Price history appears to be correct.',
        yf_start,
        yf_end, 
        1
    )
    append_to_ledger(tmp_path, ticker, yf_start, yf_end, result)

    # New ledger should have been created (with valid schema).
    new_file = tmp_path / '_ledger.parquet'
    try:
        ledger = pd.read_parquet(new_file)
        expected_columns = {
            'ticker',
            'attempted_at',
            'spell_start',
            'spell_end',
            'yf_start',
            'yf_end',
            'status',
            'reason',
            'n_rows'
        }
        if not expected_columns.issubset(ledger.columns):
            raise ValueError("Ledger schema is invalid.")
    except (OSError, ValueError, pa.lib.ArrowException) as e:
        logger.warning(f"Ledger file still corrupt: ({e!r})")
        raise e
    if ledger['yf_start'].iloc[0] != pd.Timestamp('2020-01-01'):
        raise ValueError("Data types in row may be tuples.")

