import pandas as pd
import os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from factor_research.data.universe import fetch_current_sp500_members, fetch_sp500_changes, _load_latest_cache, build_membership_spells, get_constituents_on_date


def test_fetch_current_members_returns_valid_schema(tmp_path):
    # Fetch df.
    df = fetch_current_sp500_members(cache_dir=tmp_path)

    # Specify expected columns of df.
    expected_columns = {'ticker', 'security', 'gics_sector', 'date_added'}

    # Assertions.
    if not expected_columns.issubset(df.columns):
         raise ValueError(f"Missing columns: {expected_columns - set(df.columns)}")
    
    if len(df) < 480: 
        raise ValueError(f"Too few constituents: {len(df)} (expected at least 480)")
    
    if not df['ticker'].str.isupper().all(): 
        raise ValueError("All tickers should be uppercase")
    
    if not df['ticker'].str.contains(r'^[A-Z0-9\.\-]+$').all(): 
        raise ValueError("Tickers should only contain valid characters")
    
    if not (df['ticker'].str.len() <= 6).all(): 
        raise ValueError("Tickers should be 6 characters or fewer")
    

def test_load_latest_cache_TTL(tmp_path):
     # Write a valid parquet file to tmp_path using artificial schema and table.
    schema = pa.schema([
        ('id', pa.int64()),
        ('name', pa.string()),
        ('is_active', pa.bool_())
    ])
    table = pa.Table.from_batches([], schema=schema)
    file_path = tmp_path / "sp500_current_1969-04-20.parquet"
    pq.write_table(table, file_path)

    # Fetch timestamps from parquet file.
    file_stats = os.stat(file_path)
    current_atime = file_stats.st_atime

    # Set parquet file to have artificially-old mtime (as a Unix timestamp).
    fake_mtime = -22025970
    os.utime(file_path, (current_atime, fake_mtime))

    # Call _load_latest_cache API and store return value.
    return_value = _load_latest_cache(tmp_path, 'current', max_age_days=1)

    # Assert that return value is None.
    if return_value is not None:
        raise ValueError("API does not respect TTL.")


def test_corruption_recovery(tmp_path):
    # Create corrupt file.
    corrupt_file = tmp_path / f"sp500_current_{pd.Timestamp.now().strftime('%Y-%m-%d')}.parquet"
    corrupt_file.write_bytes(b"this is not a parquet file :(")

    # Assert that return value is None.
    return_value = _load_latest_cache(tmp_path, 'current')
    if return_value is not None:
        raise ValueError("API does not detect corrputed file.")
    

def test_ttl_plumbing(tmp_path):
    # Write a valid parquet file to tmp_path using artificial schema and table.
    schema = pa.schema([
        ('id', pa.int64()),
        ('name', pa.string()),
        ('is_active', pa.bool_())
    ])
    table = pa.Table.from_batches([], schema=schema)
    file_path = tmp_path / "sp500_current_1969-04-20.parquet"
    pq.write_table(table, file_path)

    # Fetch timestamps from parquet file.
    file_stats = os.stat(file_path)
    current_atime = file_stats.st_atime

    # Set parquet file to have artificially-old mtime (as a Unix timestamp).
    fake_mtime = -22025970
    os.utime(file_path, (current_atime, fake_mtime))

    # Refetch data.
    fetch_current_sp500_members(cache_dir=tmp_path, max_age_days=1)

    # Verify that new parquet file with today's date exists. 
    new_file_path = tmp_path / f"sp500_current_{pd.Timestamp.now().strftime('%Y-%m-%d')}.parquet"
    if not new_file_path.is_file():
        raise ValueError("New cache file with today's date does not exist.")
    

def test_build_membership_spells_smoke_test():
    # Construct tiny current_members df.
    current_members_data = {
        'ticker': ['AAPL', 'MSFT'],
        'security': ['Apple Inc.', 'Microsoft'],
        'gics_sector': ['Information Technology', 'Systems Software'],
        'date_added': [pd.Timestamp('1982-11-30'), pd.Timestamp('1994-6-1')]
    }
    current_members = pd.DataFrame(current_members_data)

    # Construct tiny changes df.
    changes_data = {
        'date': [ pd.Timestamp('2020-1-15'), pd.Timestamp('2022-6-30')],
        'action': ['add', 'remove'],   
        'ticker': ['XYZ', 'XYZ'],
        'security': ['XYZ Inc.', 'XYZ Inc'],
        'reason': ['Need money :)', 'Bankrupt :(']
    }
    changes = pd.DataFrame(changes_data)

    # Call build_membership_spells.
    membership_spells = build_membership_spells(current_members=current_members, changes=changes)
    # Verify df has the follwing 3 rows: 
    # AAPL (NaT, NaT), MSFT (NaT, NaT), XYZ (2020-01-15, 2022-06-30).
    if len(membership_spells) != 3:
        raise ValueError("df length is invalid.")

    expected_tickers = ['AAPL', 'MSFT', 'XYZ']
    expected_start = [pd.NaT, pd.NaT, pd.Timestamp('2020-1-15')]
    expected_end = [pd.NaT, pd.NaT, pd.Timestamp('2022-6-30')]

    for index in range(len(membership_spells)):
        if membership_spells.iloc[index]['ticker'] != expected_tickers[index]:
            raise ValueError("Ticker values in membership spells table are invalid.")
        if not _dates_are_equal(membership_spells.iloc[index]['start_date'], expected_start[index]):
            raise ValueError("start_date values in membership spells table are invalid.")
        if not _dates_are_equal(membership_spells.iloc[index]['end_date'], expected_end[index]):
            raise ValueError("end_date values in membership spells table are invalid.")
        

def test_build_membership_spells_ticker_reuse():
    # Construct tiny current_members df.
    current_members_data = {
        'ticker': ['WB'],
        'security': ['Weibo Corp'],
        'gics_sector': ['Information Technology'],
        'date_added': [pd.Timestamp('2014-4-21')]
    }
    current_members = pd.DataFrame(current_members_data)

    # Construct tiny changes df.
    changes_data = {
        'date': [pd.Timestamp('2008-9-29'), pd.Timestamp('2014-4-21')],
        'action': ['remove', 'add'],
        'ticker': ['WB', 'WB'],
        'security': ['Wachovia.', 'WB Corp'],
        'reason': ['Bankrupt :(', 'CHINA #1!!!!']
    }
    changes = pd.DataFrame(changes_data)

    # Call build_memebership_spells.
    membership_spells = build_membership_spells(current_members=current_members, changes=changes)

    # Verify df has the follwing 2 rows: 
    # WB (NaT, 2008-09-29), WB (2014-4-21, NaT).
    if len(membership_spells) != 2:
        raise ValueError("df length is invalid.")

    expected_tickers = ['WB', 'WB']
    expected_start = [pd.Timestamp('2014-4-21'), pd.NaT]
    expected_end = [pd.NaT, pd.Timestamp('2008-9-29')]

    for index in range(len(membership_spells)):
        if membership_spells.iloc[index]['ticker'] != expected_tickers[index]:
            raise ValueError("Ticker values in membership spells table are invalid.")
        if not _dates_are_equal(membership_spells.iloc[index]['start_date'], expected_start[index]):
            raise ValueError("start_date values in membership spells table are invalid.")
        if not _dates_are_equal(membership_spells.iloc[index]['end_date'], expected_end[index]):
            raise ValueError("end_date values in membership spells table are invalid.")
        
def test_build_membership_spells_currently_in():
    # Construct tiny current_members df.
    current_members_data = {
        'ticker': ['TSLA'],
        'security': ['Tesla, Inc.'],
        'gics_sector': ['Automobile Manufacturers'],
        'date_added': [pd.Timestamp('2020-12-21')]
    }
    current_members = pd.DataFrame(current_members_data)

    # Construct tiny changes df.
    changes_data = {
        'date': [pd.Timestamp('2020-12-21')],
        'action': ['add'],
        'ticker': ['TSLA'],
        'security': ['Tesla, Inc.'],
        'reason': ['Elon Musky']
    }
    changes = pd.DataFrame(changes_data)

    # Call membership_spells.
    membership_spells = build_membership_spells(current_members=current_members, changes=changes)

    # Verify resulting row: (TSLA, 2020-12-21, NaT).
    if not membership_spells.loc[membership_spells['ticker'] == 'TSLA', 'ticker'].item() == 'TSLA':
        raise ValueError('Ticker is invalid.')
    if not _dates_are_equal(membership_spells.loc[membership_spells['ticker'] == 'TSLA', 'start_date'].item(), pd.Timestamp('2020-12-21')):
        raise ValueError('start_date is invalid.')
    if not _dates_are_equal(membership_spells.loc[membership_spells['ticker'] == 'TSLA', 'end_date'].item(), pd.NaT):
        raise ValueError('end_date is invalid.')


def test_build_membership_spells_real_data_sanity_check(tmp_path):
    # Fetch current members and changes dfs.
    current_members = fetch_current_sp500_members(tmp_path)
    changes = fetch_sp500_changes(tmp_path)

    # Call build_membership_spells.
    membership_spells = build_membership_spells(current_members=current_members, changes=changes)

    # Assert: there is exactly one closed spell for LEH with end_date==2008-09-16 and start_date either NaT or before 2008.
    leh_count = (membership_spells['ticker'] == 'LEH').sum()
    if leh_count != 1:
        raise ValueError("There is not exactly one spell for LEH.")
    leh_start_date = membership_spells[membership_spells['ticker'] == 'LEH']['start_date'].iloc[0]
    if (not _dates_are_equal(leh_start_date, pd.NaT)) and (leh_start_date >= pd.Timestamp('2008-1-1')):
        raise ValueError("start_date for LEH is invalid.")
    leh_end_date = membership_spells[membership_spells['ticker'] == 'LEH']['end_date'].iloc[0]
    if not _dates_are_equal(leh_end_date, pd.Timestamp('2008-9-16')):
        raise ValueError("end_date for LEH is invalid.")

    # Assert: there is at least one open spell for AAPL (end date is NaT)
    aapl_end_date = membership_spells[membership_spells['ticker'] == 'AAPL']['end_date'].iloc[0]
    if not _dates_are_equal(aapl_end_date, pd.NaT):
        raise ValueError("AAPL spell is not open.")

    # Assert: there is exactly one spell for WB.
    wb_count = (membership_spells['ticker'] == 'WB').sum()
    if wb_count != 1:
        print(wb_count)
        raise ValueError("There are not exactly two spells for WB.")


def test_get_constituents_on_date_synthetic():
    # Construct tiny spells df.
    data = {
        'ticker': ['AAPL'],
        'start_date': [pd.Timestamp('2020-1-1')],
        'end_date': [pd.Timestamp('2025-1-1')]
    }

    spells = pd.DataFrame(data)

    # Query date in middle of ticker's spell --> ticker should be present.
    as_of = pd.Timestamp('2022-1-1')
    constituents = get_constituents_on_date(spells=spells, as_of=as_of)
    if not constituents:
        raise ValueError("Constituents list should not be empty.")
    
    # Query before ticker's start date --> ticker should not be present.
    as_of = pd.Timestamp('1969-4-20')
    constituents = get_constituents_on_date(spells=spells, as_of=as_of)
    if constituents:
        raise ValueError("Constituents list should be empty.")
    

def test_get_constituents_on_date_boundary():
    # Construct tiny spells df.
    data = {
        'ticker': ['AAPL'],
        'start_date': [pd.Timestamp('2020-1-1')],
        'end_date': [pd.Timestamp('2025-1-1')]
    }

    spells = pd.DataFrame(data)

    # Query date on start date --> ticker should be present.
    as_of = pd.Timestamp('2020-1-1')
    constituents = get_constituents_on_date(spells=spells, as_of=as_of)
    if not constituents:
        raise ValueError("Constituents list should not be empty.")

    # Query date on end date --> ticker should not be present.
    as_of = pd.Timestamp('2025-1-1')
    constituents = get_constituents_on_date(spells=spells, as_of=as_of)
    if constituents:
        raise ValueError("Constituents list should be empty.")
    

def test_get_constituents_on_date_real_data_sanity(tmp_path):
    # Call full pipeline.
    current = fetch_current_sp500_members(tmp_path)
    changes = fetch_sp500_changes(tmp_path)
    spells = build_membership_spells(current, changes)
    constituents = get_constituents_on_date(spells, pd.Timestamp('2008-6-30'))

    # Verify constituents length is between 480 and 520
    n = len(constituents)
    if n < 480 or n > 520:
        raise ValueError("Constituents list size is invalid.")
    
    # Verify 'LEH' ticker is in constituents.
    if 'LEH' not in constituents:
        raise ValueError("'LEH' ticker should be in constituents list.")
    
    # Verify 'TSLA' ticker is NOT in constituents.
    if 'TSLA' in constituents:
        raise ValueError("'TSLA' ticker should not be in constituents list.")
    
def test_get_constituents_on_date_with_always_in_ticker():
    """A ticker with NaT start_date (always-in before log) should be
    returned for any as_of date <= its end_date."""
    spells = pd.DataFrame({
        'ticker': ['ALWAYSIN', 'NEWBIE'],
        'start_date': [pd.NaT, pd.Timestamp('2020-01-01')],
        'end_date': [pd.NaT, pd.NaT],
    })
    # Query a date in 1995 — long before NEWBIE was added, but
    # ALWAYSIN was already in the index before our log started.
    result = get_constituents_on_date(spells, pd.Timestamp('1995-06-15'))
    assert result == ['ALWAYSIN'], f"Expected ['ALWAYSIN'], got {result}"

def test_get_constituents_on_date_future_returns_only_current():
    """A future as_of should return only currently-open spells
    (NaT end_date), not closed historical spells."""
    spells = pd.DataFrame({
        'ticker': ['LEH', 'AAPL'],
        'start_date': [pd.NaT, pd.NaT],
        'end_date': [pd.Timestamp('2008-09-16'), pd.NaT],
    })
    # Query a date in 2050 — Lehman has been gone for 42 years.
    result = get_constituents_on_date(spells, pd.Timestamp('2050-01-01'))
    assert result == ['AAPL'], f"Expected ['AAPL'], got {result}"
    

def _dates_are_equal(date1, date2):
    """
    Helper function used to determine if two dates are equal.
    Safely handles if date1 == pd.NaT or date2 == pd.NaT.
    """
    # If both are NaT, consider them equal
    if pd.isna(date1) and pd.isna(date2):
        return True
    # If only one is NaT, they are not equal
    elif pd.isna(date1) or pd.isna(date2):
        return False
    # Otherwise, compare the values normally
    return date1 == date2