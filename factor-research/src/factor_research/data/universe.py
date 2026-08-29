import pandas as pd
from pathlib import Path
from io import StringIO
import requests
import logging; logger = logging.getLogger(__name__)
from datetime import datetime, timezone
import pyarrow as pa

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

_VALID_PREFIXES = ('current', 'changes')

class WikipediaScrapeError(Exception): pass

def fetch_current_sp500_members(
    cache_dir: Path,
    force_refresh: bool = False,
    max_age_days: int=7,
) -> pd.DataFrame:
    """
    Fetch the current S&P 500 members from Wikipedia.

    On the first call (or when force_refresh=True), scrapes Wikipedia
    and caches the result as a parquet file in cache_dir, with a
    date-stamped filename for provenance. On subsequent calls, returns
    the cached DataFrame without hitting the network.

    Parameters
    ----------
    cache_dir: Path
        Path to where the files are cached.
    force_refresh: bool
        If True, data will be fetched from Wikipedia; otherwise, check for latest historical changes log in cache file path.
    max_age_days: int
        Determines the TTL of the changes log. Default age is 7 days.

    Returns
    -------
    pd.DataFrame with columns:
        - ticker (str)           : Yahoo-compatible ticker symbol.
        - security (str)         : Company name.
        - gics_sector (str)      : GICS sector.
        - date_added (Timestamp) : Date the company joined the S&P 500
                                  (NaT if Wikipedia doesn't list one).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not force_refresh:
        cached = _load_latest_cache(cache_dir, prefix='current', max_age_days=max_age_days)
        if cached is not None:
            return cached

    df = _fetch_sp500_data_from_wikipedia(table_index=0)

    df = df.rename(columns={
        'Symbol': 'ticker',
        'Security': 'security',
        'GICS Sector': 'gics_sector',
        'Date added': 'date_added',
    })[['ticker', 'security', 'gics_sector', 'date_added']]

    df['ticker'] = df['ticker'].apply(_normalize_ticker)
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

    # Assertions
    if df.empty:
        raise WikipediaScrapeError("Wikipedia returned no data--perhaps page structure has changed?")
    expected_columns = {'ticker', 'security', 'gics_sector', 'date_added'}
    if not expected_columns.issubset(df.columns):
         raise ValueError(f"Missing columns: {expected_columns - set(df.columns)}")

    _save_cache(df, cache_dir, prefix='current')
    return df


def fetch_sp500_changes(
    cache_dir: Path,
    force_refresh: bool = False,
    max_age_days: int=7
) -> pd.DataFrame:
    """
    Fetch the historical changes log of the S&P 500 from Wikipedia.

    Each row in Wikipedia's 'Selected changes' table represents one
    event date and may contain BOTH an added and a removed ticker
    (a substitution) OR only one of them. This function normalizes
    the table into a long-format DataFrame with one row per
    (date, action, ticker) event, sorted ascending by date.

    Parameters
    ----------
    cache_dir: Path
        Path to where the files are cached.
    force_refresh: bool
        If True, data will be fetched from Wikipedia; otherwise, check for latest historical changes log in cache file path.
    max_age_days: int
        Determines the TTL of the changes log. Default age is 7 days.

    Returns
    -------
    pd.DataFrame with columns:
      - date (Timestamp) : Effective date of the change.
      - action (str)     : 'add' or 'remove'.
      - ticker (str)     : Yahoo-compatible ticker.
      - security (str)   : Company name (best-effort).
      - reason (str)     : Reason given by Wikipedia (may be empty).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not force_refresh:
        cached = _load_latest_cache(cache_dir, prefix='changes', max_age_days=max_age_days)
        if cached is not None:
            return cached

    df = _fetch_sp500_data_from_wikipedia(table_index=1)

    # Flatten multi-index columns if present.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map(" ".join).str.strip()

    # Split df into two dfs: one with only "adds" and one with only "removes".
    adds_df = _initialize_cols_for_changes_df(df.copy(), "add", "Added Ticker", "Added Security")
    removes_df = _initialize_cols_for_changes_df(df.copy(), "remove", "Removed Ticker", "Removed Security")

    # Concatanate adds and removes dfs into final df.
    final_df = (
        pd.concat([adds_df, removes_df], ignore_index=True)
        .assign(
            date=lambda x: pd.to_datetime(x["date"]), # Ensure date col is DateTime
            reason=lambda x: x["reason"].str.replace(r"\[.*?\]", "", regex=True) # Remove footnotes from Wikipedia
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Assertions
    if df.empty:
        raise WikipediaScrapeError("Wikipedia returned no data--perhaps page structure has changed?")
    expected_columns = {'date', 'action', 'ticker', 'security', 'reason'}
    if not expected_columns.issubset(final_df.columns):
         raise ValueError(f"Missing columns: {expected_columns - set(final_df.columns)}")

    # Cache final_df and return
    _save_cache(final_df, cache_dir, prefix="changes")
    return final_df


def build_membership_spells(
    current_members: pd.DataFrame,
    changes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the spell-based membership table for the S&P 500.

    Parameters
    ----------
    current_members : pd.DataFrame
        Output of fetch_current_sp500_members (today's constituents).
    changes : pd.DataFrame
        Output of fetch_sp500_changes (chronological add/remove log).

    Returns
    -------
    pd.DataFrame with columns:
      - ticker (str)
      - start_date (Timestamp or NaT) : When the ticker entered the
            index. NaT means "ticker was already in the index before
            our changes log starts."
      - end_date (Timestamp or NaT) : When the ticker left the index.
            NaT means "ticker is still currently in the index."

    Each ticker may appear multiple times (one row per continuous
    spell of membership). Spells are sorted by (ticker, start_date),
    with NaT start_dates sorted last.

    Notes
    -----
    Ticker reuse is handled correctly: if 'WB' was Wachovia until
    2008 and is now Weibo (since 2014), this function returns two
    rows for 'WB' — one closed spell ending in 2008, one open spell
    starting in 2014.
    """
    # Initialize dict that maps ticker to known end_date of its currently-open spell.
    active = dict.fromkeys(current_members['ticker'], pd.NaT)

    # Maintain list closed_spells: list[tuple[str, Timestamp, Timestamp]].
    closed_spells = []

    # Sort changes df descending by date.
    changes_sorted = changes.sort_values(by='date', ascending=False)

    # Iterate over sorted changes df.
    for row in changes_sorted.itertuples():
        # Fetch date, action, and ticker values.
        event_date = row.date
        action = row.action
        ticker = row.ticker

        if action == 'add':
            # If an 'add' event is not in active, skip event.
            if ticker not in active:
                logger.warning("Ticker was added in changes but not present current members; skipping event.")
                continue
            closed_spells.append((ticker, event_date, active[ticker]))
            del active[ticker]
        elif action == 'remove':
            # If a 'remove' event is already in active, skip event. 
            if ticker in active:
                logger.warning("Ticker seems to have been added to changes twice; skipping event.")
                continue
            active[ticker] = event_date 

    # Emit one spell per remaining ticker in active.
    for ticker in active:
        closed_spells.append((ticker, pd.NaT, active[ticker]))
    
    # Convert closed_spells to df with desire columns, sort by (ticker, start_date), and return.
    closed_spells_df = pd.DataFrame(closed_spells, columns=['ticker', 'start_date', 'end_date'])
    closed_spells_df_sorted = closed_spells_df.sort_values(by=['ticker', 'start_date'])

    return closed_spells_df_sorted

def get_constituents_on_date(
    spells: pd.DataFrame,
    as_of: pd.Timestamp,
) -> list[str]:
    """
    Return the tickers that were in the S&P 500 on a given date.

    Parameters
    ----------
    spells: pd.DataFrame
        df fetched from build_membership_spells API.
    as_of: pd.Timestamp
        Date used to fetch constituents.
    
    Returns
    -------
    list[str] of constituents in the S&P 500 on as_of date.

    Notes
    -----
    Add dates are inclusive, remove dates are not inclusive. 
    """
    # Initialize empty constituents list.
    constituents = []

    # Iterate over spells df.
    for spell in spells.itertuples():
        # Fetch ticker, start_date, and end_date values.
        ticker = spell.ticker
        start_date = spell.start_date
        end_date = spell.end_date

        # Add spell if start_date and end_date values respect as_of date.
        is_valid_start_date = _is_valid_date(start_date, as_of, is_start_date=True)
        is_valid_end_date = _is_valid_date(end_date, as_of, is_start_date=False)
        is_in_index = is_valid_start_date and is_valid_end_date

        if is_in_index:
            constituents.append(ticker)
        
    return constituents 


def _fetch_sp500_data_from_wikipedia(table_index: int) -> pd.DataFrame:
    """
    Fetch a table from the S&P 500 Wikipedia page.

    Parameters
    ----------
    table_index: int
        Set to 0 to fetch members df; set to 1 to fetch changes df.
    
    Returns
    -------
    pd.DataFrame of desired table (according to table_index) from Wikipedia.

    """
    response = requests.get(
        WIKI_SP500_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    return tables[table_index]

def _initialize_cols_for_changes_df(df: pd.DataFrame, action: str, ticker_col: str, security_col: str) -> pd.DataFrame:
    """
    Initalizes the columns for the dfs that will be merged in the fetch_sp500_changes API.

    Parameters
    ----------
    df: pd.DataFrame
        Pulled from changes API to change column structure.
    action: str
        Must be set to either "add" or "remove" 
    ticker_col: str
    security_col: str

    Returns
    -------
    pd.DataFrame with corruct column structure. 
    """

    # Assert valid action values.
    valid_action_values = ['add', 'remove']
    if action not in valid_action_values:
        raise ValueError("'action' paramater must be set to 'add' or 'remove'.")

    return (
        df.rename(
            columns={
                "Effective Date Effective Date": "date",
                ticker_col: "ticker",
                security_col: "security",
                "Reason Reason": "reason",
            }
        )
        .assign(action=action)
        [["date", "action", "ticker", "security", "reason"]]
        .dropna(subset=["ticker"])
    )


def _normalize_ticker(wiki_ticker: str) -> str:
    """
    Normalize a Wikipedia ticker for yfinance compatibility
    (e.g., 'BRK.B' -> 'BRK-B').

    Parameters
    ----------
    wiki_ticker: str
        Ticker str pulled from Wikipedia.
    
    Returns
    -------
    str that is compatible with yfinace.
    """
    return str(wiki_ticker).replace('.', '-')


def _load_latest_cache(cache_dir: Path, prefix: str, max_age_days: int=7) -> pd.DataFrame | None:
    """
    Return the most recent cached parquet for the given prefix,
    or None if no cache exists.

    Parameters
    ----------
    cache_dir: Path
        Path to where the files are cached.
    prefix: str
        Must be either 'current' or 'changes'.
    max_age_days: int
        Determines the TTL of the changes log. Default age is 7 days.

    Returns
    -------
    pd.Dataframe of most recent cached parquet.

    """
    if prefix not in _VALID_PREFIXES:
        raise ValueError(f"Prefix must be one of {_VALID_PREFIXES}.")

    cache_files = list(cache_dir.glob(f"sp500_{prefix}_*.parquet"))
    if not cache_files:
        logger.warning(f"No cache found for S&P 500 {prefix}. Fetching data from Wikipedia.")
        return None

    latest_cache = max(cache_files, key=lambda f: f.stat().st_mtime)

    # Check time of latest cache file, convert to datetime, and calculate age of latest cache file (in days).
    unix_latest_cache_date = latest_cache.stat().st_mtime
    latest_cache_date = datetime.fromtimestamp(unix_latest_cache_date, tz=timezone.utc)
    current_date = datetime.now(timezone.utc)
    time_difference = current_date - latest_cache_date
    latest_cache_age_days = time_difference.days

    if latest_cache_age_days >= max_age_days:
        logger.warning(f"Latest cache file is at least {max_age_days} old. Fetching new data.")
        return None

    logger.info(f"Loading S&P 500 {prefix} from cache: {latest_cache}")

    try:
        cache_parquet = pd.read_parquet(latest_cache)
    except (OSError, ValueError, pa.lib.ArrowException) as e:
        logger.warning(f"Cache file at {latest_cache} appears corrupted ({e!r}); refetching.")
        return None

    return cache_parquet


def _save_cache(df: pd.DataFrame, cache_dir: Path, prefix: str) -> None:
    """
    Write df to cache_dir as sp500_<prefix>_<YYYY-MM-DD>.parquet.

    Parameters
    ----------
    df: pd.DataFrame
        df to be converted to parquet file.
    cache_dir: Path
        Path to where the files are cached.
    prefix: str
        Must be either 'current' or 'changes'.
    """
    if prefix not in _VALID_PREFIXES: 
        raise ValueError(f"Prefix must be one of {_VALID_PREFIXES}.")

    cache_filename = cache_dir / f"sp500_{prefix}_{pd.Timestamp.now().strftime('%Y-%m-%d')}.parquet"
    df.to_parquet(cache_filename, index=False)
    logger.info(f"Cached S&P 500 {prefix} to: {cache_filename}")

def _is_valid_date(date_to_check, as_of_date: pd.Timestamp, *, is_start_date: bool):
    """
    Helper function used in get_constituents_on_date to check if start or end date is in index, relative to as_of date.

    Parameters
    ----------
    date_to_check: pd.Timestamp
        start or end date to check.
    as_of_date: pd.Timestamp
        as_of pulled from constituents API.
    is_start_date: bool
        True if date_to_check is start_date, False otherwise.

    Returns
    -------
    bool specifying if date_to_check is in index.
    """
    # If date_to_check is pd.NaT, always return True.
    if pd.isna(date_to_check):
        return True 
    
    if is_start_date:
        return date_to_check <= as_of_date 
    return date_to_check > as_of_date