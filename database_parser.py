import os
import datetime
import sqlite3
from pyproj import Transformer

def get_dat_files_paths(lightning_data_folder, data_extension):
    """Return full paths of files with the given extension in the specified folder."""
    return [
        os.path.join(lightning_data_folder, f)
        for f in os.listdir(lightning_data_folder)
        if f.endswith(data_extension)
    ]

# Predefined station order (adjust as needed)
STATION_MASK_ORDER = ['STN1', 'STN2', 'STN3', 'STN4', 'STN5', 'STN6', 'STN7', 'STN8']

# Initialize a transformer to convert from WGS84 (lat,lon,alt in EPSG:4979) to ECEF (EPSG:4978)
transformer = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)

def _decode_station_mask(mask_str):
    """Convert a hexadecimal mask string to a comma-separated list of station names."""
    mask_int = int(mask_str.strip(), 16)
    stations = []
    for i, station in enumerate(STATION_MASK_ORDER):
        if mask_int & (1 << i):
            stations.append(station)
    return ",".join(stations)

def _add_to_database(cursor, event):
    """
    Inserts an event record into the events table.
    'event' is a tuple containing:
    (time_unix, lat, lon, alt, reduced_chi2, num_stations, power, mask, stations, x, y, z)
    """
    cursor.execute("""
        INSERT INTO events (
            time_unix, lat, lon, alt, reduced_chi2, num_stations, power, mask, stations, x, y, z
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, event)

def _create_database_if_not_exist(DB_PATH: str = "lylout_db.db"):
    """
    Creates the database and events table if they do not exist.
    Returns a sqlite3 connection object.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            time_unix FLOAT,
            lat FLOAT,
            lon FLOAT,
            alt FLOAT,
            reduced_chi2 FLOAT,
            num_stations INTEGER,
            power FLOAT,
            mask TEXT,
            stations TEXT,
            x FLOAT,
            y FLOAT,
            z FLOAT
        )
    """)
    conn.commit()
    return conn

def _parse_dat_extension(lylout_path: str, DB_PATH: str = "lylout_db.db"):
    if not lylout_path.lower().endswith(".dat"):
        raise Exception("File must be a .dat file")
    
    with open(lylout_path, "r") as f:
        lines = f.readlines()
    
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
        power = float(parts[6])
        mask_str = parts[7]
        
        # Convert UT seconds (since midnight UTC) to Unix timestamp
        midnight = base_date.replace(hour=0, minute=0, second=0, microsecond=0)
        event_time = midnight + datetime.timedelta(seconds=ut_sec)
        time_unix = event_time.timestamp()
        
        # Decode the station bitmask into a comma-separated list
        stations_list = _decode_station_mask(mask_str)
        
        # Convert geodetic coordinates to ECEF using pyproj
        x, y, z = transformer.transform(lon, lat, alt)
        
        event = (time_unix, lat, lon, alt, reduced_chi2, num_stations, power,
                 mask_str, stations_list, x, y, z)
        _add_to_database(cursor, event)
    
    conn.commit()
    conn.close()

def parse_lylout(lylout_path: str, DB_PATH: str = "lylout_db.db"):
  if lylout_path.lower().endswith(".dat"):
    _parse_dat_extension(lylout_path, DB_PATH)
