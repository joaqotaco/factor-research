import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf
import time
import requests
from dataclasses import dataclass
import logging; logger = logging.getLogger(__name__)


# Custom exception class(es)
class PriceFetchError(Exception):
    pass


# Constants
_MIN_SECONDS_BETWEEN_FETCHES = 1.5
TRANSIENT_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.HTTPError,   # 429, 5xx, etc.
)


def fetch_ticker_prices(
    ticker: str,
    cache_dir: Path,
    force_refresh: bool = False,
    max_age_days: int = 1,
) -> pd.DataFrame:
    """
    Fetch full price history for a single ticker from yfinance.

    On a cache miss (or force_refresh, or stale cache), fetches the
    entire available history (period="max") from yfinance using a
    curl_cffi-impersonated Chrome session, with retry-and-backoff
    for transient errors and a rate-limit sleep between calls. The
    result is cached as cache_dir/{ticker}.parquet for subsequent
    reads within max_age_days.

    Notes on returned schema
    ------------------------
    Columns are lowercased + snake_case for consistency with our
    data layer convention. Index is a DatetimeIndex named 'date',
    timezone-naive (we drop tz here; price bars are end-of-day).

    Returns
    -------
    pd.DataFrame indexed by 'date' (DatetimeIndex), with columns:
        - open, high, low, close (raw, unadjusted prices)
        - adj_close             (yfinance's split+dividend-adjusted close)
        - volume                (raw share volume)
        - dividends             (cash dividend on ex-date, else 0)
        - stock_splits          (split ratio on split date, else 0)

    Raises
    ------
    PriceFetchError
        If after max_retries the fetch still fails with a transient error.
    
    Empty results are NOT treated as an error in Phase 1 — the function
    returns an empty DataFrame. Phase 2 will introduce explicit validity
    checks against the universe spell table.
    """
    # Create new directory (if one does not already exist).
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache check.
    if not force_refresh:
        cached = _cache_check(cache_dir, ticker, max_age_days)
        if cached is not None:
            return cached 
    
    # Initialize curl_cffi session.
    session = _make_session()

    # Fetch ticker data from yfinance.
    ticker_data = _fetch_from_yfinance_with_retry(ticker, session=session)

    # Normalize ticker_data.
    ticker_data.columns = ticker_data.columns.str.replace(r'[-\s]+', '_', regex=True).str.lower() # Lowercase + snake_case columns.
    ticker_data.index.name = 'date'                                     # Index name: Date -> date.
    ticker_data.index = pd.to_datetime(ticker_data.index)               # Ensure Index is pd.Timestamp.
    ticker_data.index = ticker_data.index.tz_localize(None)             # Drop timezone.
    ticker_data_sorted = ticker_data.sort_index()                       # Sort by index ascending. 

    # Cache file, then return.
    _save_cache(ticker_data_sorted, cache_dir=cache_dir, ticker=ticker)
    return ticker_data_sorted


@dataclass(frozen=True)
class ValidationResult:
    status: str       # 'OK' | 'EMPTY' | 'TICKER_REUSE' | 'PARTIAL_HISTORY'
    reason: str       # human-readable detail
    yf_start: pd.Timestamp | None
    yf_end: pd.Timestamp | None
    n_rows: int


def validate_ticker_data(
    ticker: str,
    prices: pd.DataFrame,
    spell_start: pd.Timestamp | None,    # NaT also accepted
    spell_end: pd.Timestamp | None,      # NaT also accepted (currently in)
    tolerance_days: int = 7,
) -> ValidationResult:
    """
    Verify yfinance's price series is consistent with what universe.py
    says about this ticker's index membership window.

    Decision tree (in order):
      1. prices is empty            -> EMPTY
      2. spell_end known AND yf_start > spell_end + tolerance
                                    -> TICKER_REUSE
      3. spell_start known AND yf_start > spell_start + tolerance
                                    -> PARTIAL_HISTORY
      4. otherwise                  -> OK

    Notes
    -----
    - "spell_start known" means not NaT/None. Same for spell_end.
    - If spell_start is NaT, we skip the partial-history check (we don't
      know when the ticker entered the index, so we can't measure
      history adequacy).
    - The TICKER_REUSE check is the most important: it catches the
      WB->Weibo case where a recycled ticker returns data for a
      different company.
    - tolerance_days is in CALENDAR days, not business days. 7 calendar
      days handles weekends, holidays, and small reporting quirks.
    """
    # Iterate through decision tree logic.
    if prices.empty:
        result = ValidationResult(
            'EMPTY',
            'Price history does not exist on yfinance.',
            pd.NaT,
            pd.NaT,
            0
        )
        
        return result 

    yf_start = prices.index[0]
    yf_end = prices.index[-1]
    n_rows = len(prices)

    if not pd.isna(spell_end) and (yf_start > spell_end + pd.Timedelta(days=tolerance_days)):
        status = 'TICKER_REUSE'
        reason = f"{ticker} was reused -> price history cannot be trusted."
    elif not pd.isna(spell_start) and (yf_start > spell_start + pd.Timedelta(days=tolerance_days)):
        status = 'PARTIAL_HISTORY'
        reason = 'yfinance only has partial price history.'
    else:
        status = 'OK'
        reason = 'Price history appears to be correct.'

    result = ValidationResult(
        status,
        reason,
        yf_start,
        yf_end,
        n_rows
    )

    return result


def append_to_ledger(
    ledger_path: Path,
    ticker: str,
    spell_start: pd.Timestamp | None,
    spell_end: pd.Timestamp | None,
    result: ValidationResult,
) -> None:
    """
    Append one row to the missing-data ledger at ledger_path.

    Schema
    ------
      - ticker
      - attempted_at  (pd.Timestamp.now())
      - spell_start
      - spell_end
      - yf_start
      - yf_end
      - status
      - reason
      - n_rows

    Behavior
    --------
    - If ledger_path does not exist: create a new parquet file with this row.
    - If ledger_path exists: read, append, write back.

    Note: this is fine for our scale (~500 ticker writes per pipeline run).
    For higher volumes, partitioned/append-friendly storage is preferred.

    Every fetch attempt writes a row, including OK ones. The ledger is
    a complete audit trail; aggregations (% missing, etc.) are derived
    at memo-write time.
    """
    # Initialize row_dict.
    row_dict = {
        'ticker': ticker,
        'attempted_at': pd.Timestamp.now(),
        'spell_start': spell_start,
        'spell_end': spell_end,
        'yf_start': result.yf_start,
        'yf_end': result.yf_end,
        'status': result.status,
        'reason': result.reason,
        'n_rows': result.n_rows
    }   

    # If ledger exists and is not corrupted, append row_dict to existing ledger.
    # Otherwise, create new ledger to ledger_path.
    ledger_file_path = ledger_path / '_ledger.parquet'
    try: 
        ledger = pd.read_parquet(ledger_file_path)
        ledger = pd.concat([ledger, pd.DataFrame([row_dict])])
        ledger = ledger.sort_values(by=['attempted_at'], ascending=True)  
        ledger.to_parquet(ledger_file_path, index=False)
        logger.info(f"Ledger appended and written back to {ledger_path}.")
    except (OSError, ValueError, pa.lib.ArrowException) as e:
        logger.warning(f"Ledger file does not exist or appears to be corrupted at {ledger_path} ({e!r}); creating new ledger.")
        ledger = pd.DataFrame([row_dict])
        ledger.to_parquet(ledger_file_path, index=False)
        logger.info(f"Ledger saved to {ledger_path}.")


def fetch_universe_prices(
    spells: pd.DataFrame,
    cache_dir: Path,
    ledger_path: Path,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    progress_log_every: int = 25,
) -> pd.DataFrame:
    """
    Build the survivorship-aware long-format price panel for the
    full universe.

    For each unique ticker in `spells`:
      1. Fetch its full price history via fetch_ticker_prices (cached).
      2. For each spell of that ticker, validate against the spell
         window using validate_ticker_data.
      3. Append a row to the ledger for every (ticker, spell) attempt,
         OK or otherwise.
      4. If validation is OK or PARTIAL_HISTORY: slice the price series
         to [max(spell_start, yf_start), min(spell_end, yf_end)],
         tag with the ticker, and add to the output panel.
      5. If EMPTY or TICKER_REUSE: do NOT add to the panel. The ledger
         records the exclusion.

    Ticker-level errors (PriceFetchError after retries, unexpected
    exceptions) are caught, logged to the ledger with status='ERROR',
    and the loop continues. One bad ticker MUST NOT kill the run.

    Returns
    -------
    pd.DataFrame in long format with columns:
        - date (Timestamp)
        - ticker (str)
        - open, high, low, close, adj_close (float)
        - volume (int or float)
        - dividends, stock_splits (float)
    
    Sorted by (ticker, date) ascending. Optional [start_date, end_date]
    filter applied at the very end.

    Notes
    -----
    - The first run for a fresh cache will take ~15-30 minutes for the
      full S&P 500 universe (rate limit + network). Subsequent runs are
      seconds (everything cached).
    - PARTIAL_HISTORY is included with a warning; the ledger is the
      source of truth on coverage gaps.
    - This function is intentionally serial. Parallelizing would require
      proper rate-limit coordination and is deferred.
    """
    # Fetch tickers from spells and iterate.
    tickers = spells['ticker'].unique()
    for ticker in tickers:
        # Fetch ticker-specific spells, prices, and ValidationResult.
        ticker_spells = spells[spells['ticker'] == ticker]
        ticker_prices = fetch_ticker_prices(ticker, cache_dir)
        ticker_validation = validate_ticker_data(ticker, ticker_prices, )

        
    

def _cache_check(cache_dir: Path, ticker: str, max_age_days: int) -> pd.DataFrame | None:
    """
    Return most recent cached parquet per ticker
    or None if no cache exists.
    """
    # Fetch latest file in cache.
    cache_file = cache_dir / f"{ticker}.parquet"
    if not cache_file.exists():
        logger.warning(f"No cache found for {ticker}. Fetching new data.")
        return None

    # Calculate age of latest cache file (in days).
    unix_seconds = cache_file.stat().st_mtime
    cache_date = pd.to_datetime(unix_seconds, unit='s')
    current_date = pd.Timestamp.now()
    time_difference = current_date - cache_date
    cache_age_days = time_difference.days

    # If latest_cache is too old, fetch new data.
    if cache_age_days > max_age_days:
        logger.warning(f"Latest cache file is at least {max_age_days} old. Fetching new data.")
        return None

    # Verify lastest cache file is not corrupted.
    try:
        cache_parquet = pd.read_parquet(cache_file)
    except (OSError, ValueError, pa.lib.ArrowException) as e:
        logger.warning(f"Cache file at {cache_file} appears to be corrupted ({e!r}); refetching.")
        return None
    
    return cache_parquet


def _fetch_from_yfinance_with_retry(
    ticker: str,
    session,                       # curl_cffi session
    max_retries: int = 3,
    base_delay_seconds: float = 2.0,
) -> pd.DataFrame:
    """
    Fetch full history for a single ticker, retrying transient failures
    with exponential backoff (base_delay * 2**attempt).
    """
    # Sleep for _MIN_SECONDS_BETWEEN_FETCHES seconds before calling yfinance.
    time.sleep(_MIN_SECONDS_BETWEEN_FETCHES)

    # Loop up to max_retries times.
    attempt = 0
    while attempt < max_retries:
        # Call yf.Ticker API.
        try:
            ticker_df = yf.Ticker(ticker, session=session).history(period='max', auto_adjust=False)
            return ticker_df
        except TRANSIENT_EXCEPTIONS:
            logger.warning(f"Network or rate-limit exception reached; sleeping for {base_delay_seconds * 2**attempt} seconds, then retrying.")
            time.sleep(base_delay_seconds * 2**attempt)
            attempt += 1
    
    raise PriceFetchError(ticker, " could not be fetched from yfinance.")


def _make_session():
    """
    Return a curl_cffi session that impersonates Chrome to bypass
    Yahoo's TLS-fingerprint rate limiting.
    """
    from curl_cffi import requests as curl_requests
    return curl_requests.Session(impersonate="chrome")


def _save_cache(df: pd.DataFrame, cache_dir: Path, ticker: str) -> None:
    """
    Write df to cache_dir as <ticker>.parquet.
    """
    cache_filename = cache_dir / f"{ticker}.parquet"
    df.to_parquet(cache_filename)
    logger.info(f"Cached {ticker} to: {cache_filename}")

