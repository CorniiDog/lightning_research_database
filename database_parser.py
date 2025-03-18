import os
import datetime
import sqlite3
from pyproj import Transformer
import pandas as pd


def get_dat_files_paths(lightning_data_folder, data_extension):
    """Return full paths of files with the given extension in the specified folder."""
    return [
        os.path.join(lightning_data_folder, f)
        for f in os.listdir(lightning_data_folder)
        if f.endswith(data_extension)
    ]


# Default station mask order (each character represents a station in order)
DEFAULT_STATION_MASK_ORDER = "NMLKJIHGFEDC3A"

# Initialize a transformer to convert from WGS84 (lat,lon,alt in EPSG:4979) to ECEF (EPSG:4978)
transformer = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


def _decode_station_mask(mask_str, station_mask_order=DEFAULT_STATION_MASK_ORDER):
    """
    Convert a hexadecimal mask string to a comma-separated list of station names.
    Uses the provided station_mask_order (defaults to DEFAULT_STATION_MASK_ORDER).
    """
    mask_int = int(mask_str.strip(), 16)
    stations = []
    for i, station in enumerate(station_mask_order):
        if mask_int & (1 << i):
            stations.append(station)
    return ",".join(stations)


def _add_to_database(cursor, event):
    """
    Inserts an event record into the events table.
    'event' is a tuple containing:
    (time_unix, lat, lon, alt, reduced_chi2, num_stations, power_db, power, mask, stations, x, y, z)
    """
    cursor.execute(
        """
        INSERT INTO events (
            time_unix, lat, lon, alt, reduced_chi2, num_stations, power_db, power, mask, stations, x, y, z
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        event,
    )


def _create_database_if_not_exist(DB_PATH: str = "lylout_db.db"):
    """
    Creates the database and events table if they do not exist.
    Returns a sqlite3 connection object.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time_unix FLOAT,
            lat FLOAT,
            lon FLOAT,
            alt FLOAT,
            reduced_chi2 FLOAT,
            num_stations INTEGER,
            power_db FLOAT,
            power FLOAT,
            mask TEXT,
            stations TEXT,
            x FLOAT,
            y FLOAT,
            z FLOAT
        )
    """
    )

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_time_unix ON events(time_unix)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_num_stations ON events(num_stations)"
    )
    conn.commit()
    return conn


def _executesql(query, params=None, DB_PATH="lylout_db.db", fetch=True):
    """
    Executes a SQL query on the specified database.

    Parameters:
      query (str): SQL query to execute.
      params (list/tuple, optional): Parameters for a parameterized query.
      DB_PATH (str): Path to the SQLite database file.
      fetch (bool): If True, returns fetched results; otherwise commits changes.

    Returns:
      list: List of sqlite3.Row objects if fetch is True, else None.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enables accessing rows as dictionaries.
    cursor = conn.cursor()

    if params is None:
        params = []
    cursor.execute(query, params)

    if fetch:
        results = cursor.fetchall()
        conn.close()
        return results
    else:
        conn.commit()
        conn.close()


def _build_where_clause(filters):
    """
    Constructs a SQL WHERE clause from filters provided as a dict or a list.

    For a dict, each key is a column name with an equality filter.
    For a list, each element is either a tuple (column, operator, value)
    or a dict with keys "column", "operator", and "value".

    Parameters:
      filters (dict or list): Filter specifications.

    Returns:
      tuple: (where_clause (str), params (list))
    """
    conditions = []
    params = []

    if isinstance(filters, dict):
        for col, val in filters.items():
            conditions.append(f"{col} = ?")
            params.append(val)
    elif isinstance(filters, list):
        for filt in filters:
            if isinstance(filt, tuple) and len(filt) == 3:
                col, op, val = filt
                conditions.append(f"{col} {op} ?")
                params.append(val)
            elif isinstance(filt, dict):
                col = filt.get("column")
                op = filt.get("operator", "=")
                val = filt.get("value")
                if col is None or val is None:
                    raise ValueError(
                        "Each filter dict must have 'column' and 'value' keys."
                    )
                conditions.append(f"{col} {op} ?")
                params.append(val)
            else:
                raise ValueError(
                    "Filters must be tuples (column, operator, value) or dicts."
                )
    else:
        raise ValueError("Filters must be either a dict or a list.")

    clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return clause, params


def query_events(filters, DB_PATH="lylout_db.db"):
    """
    Queries the 'events' table using filters provided as a dict or list.

    Parameters:
      filters (dict or list): Filter conditions to apply.
      DB_PATH (str): Path to the SQLite database.

    Returns:
      list: Query result rows.
    """
    where_clause, params = _build_where_clause(filters)
    query = f"SELECT * FROM events {where_clause}"
    return _executesql(query, params, DB_PATH)


def query_events_as_dataframe(filters, DB_PATH="lylout_db.db"):
    """
    Queries the 'events' table using filters and returns results as a pandas DataFrame.

    Parameters:
      filters (dict or list): Filter conditions to apply.
      DB_PATH (str): Path to the SQLite database.

    Returns:
      pandas.DataFrame: Query results.
    """
    results = query_events(filters, DB_PATH)  # Get results as a list of sqlite3.Row
    df = pd.DataFrame(results, columns=get_headers(DB_PATH))  # Convert to DataFrame
    return df


def get_headers(DB_PATH="lylout_db.db") -> list:
    """
    Retrieves the column names (headers) from the 'events' table in the SQL database.

    Parameters:
      DB_PATH (str): Path to the SQLite database.

    Returns:
      list: A list of column names from the 'events' table.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(events)")
    headers = [row[1] for row in cursor.fetchall()]
    conn.close()
    return headers


def _parse_dat_extension(lylout_path: str, DB_PATH: str = "lylout_db.db"):
    if not lylout_path.lower().endswith(".dat"):
        raise Exception("File must be a .dat file")

    with open(lylout_path, "r") as f:
        lines = f.readlines()

    # Check for an optional station mask order override in the header
    station_mask_order = DEFAULT_STATION_MASK_ORDER
    for line in lines:
        if line.startswith("Station mask order:"):
            station_mask_order = line.split("Station mask order:")[1].strip()
            break

    # Extract the base date from header (format: "Data start time: MM/DD/YY HH:MM:SS")
    base_date = None
    for line in lines:
        if line.startswith("Data start time:"):
            parts = line.split("Data start time:")[1].strip()
            base_date = datetime.datetime.strptime(parts, "%m/%d/%y %H:%M:%S")
            base_date = base_date.replace(tzinfo=datetime.timezone.utc)
            break
    if base_date is None:
        raise Exception("Base date not found in header.")

    # Find the beginning of data
    data_start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("*** data ***"):
            data_start_index = i + 1
            break
    if data_start_index is None:
        raise Exception("Data section not found.")

    # Create the database and events table if they don't exist
    conn = _create_database_if_not_exist(DB_PATH)
    cursor = conn.cursor()

    # Process each data line
    for line in lines[data_start_index:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 8:
            continue  # Skip incomplete lines

        # Parse fields from the line
        ut_sec = float(parts[0])
        lat = float(parts[1])
        lon = float(parts[2])
        alt = float(parts[3])
        reduced_chi2 = float(parts[4])
        num_stations = int(parts[5])

        # The power value in the file is in dBW.
        power_db = float(parts[6])
        # Convert dBW to linear watts: P (watts) = 10^(dBW/10)
        power = 10 ** (power_db / 10)

        mask_str = parts[7]

        # Convert UT seconds (since midnight UTC) to Unix timestamp
        midnight = base_date.replace(hour=0, minute=0, second=0, microsecond=0)
        event_time = midnight + datetime.timedelta(seconds=ut_sec)
        time_unix = event_time.timestamp()

        # Decode the station bitmask using the (possibly overridden) station_mask_order
        stations_list = _decode_station_mask(mask_str, station_mask_order)

        # Convert geodetic coordinates to ECEF using pyproj
        x, y, z = transformer.transform(lon, lat, alt)

        event = (
            time_unix,
            lat,
            lon,
            alt,
            reduced_chi2,
            num_stations,
            power_db,
            power,
            mask_str,
            stations_list,
            x,
            y,
            z,
        )
        _add_to_database(cursor, event)

    conn.commit()
    conn.close()


def parse_lylout(lylout_path: str, DB_PATH: str = "lylout_db.db"):
    if lylout_path.lower().endswith(".dat"):
        _parse_dat_extension(lylout_path, DB_PATH)
