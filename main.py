import os
import database_parser
import logger
import datetime

lightning_data_folder = "lylout_files"
data_extension = ".dat"

os.makedirs(lightning_data_folder, exist_ok=True) # Ensure that it exists

dat_file_paths = database_parser.get_dat_files_paths(lightning_data_folder, data_extension)

for file_path in dat_file_paths:
  if not logger.is_logged(file_path):
    print(file_path, "not appropriately added to the database. Adding...")
    database_parser.parse_lylout(file_path)
    logger.log_file(file_path) # Mark as logged and unmodified
  else:
    print(file_path, "was parsed and added to the database already")

headers = database_parser.get_headers()
print("Headers:", headers)

# Column/Header descriptions:
# 'time_unix'    -> Seconds (Unix timestamp, UTC)
# 'lat'          -> Degrees (WGS84 latitude)
# 'lon'          -> Degrees (WGS84 longitude)
# 'alt'          -> Meters (Altitude above sea level)
# 'reduced_chi2' -> Reduced chi-square goodness-of-fit metric
# 'num_stations' -> Count (Number of contributing stations)
# 'power_db'     -> Decibels (dBW) (Power of the detected event in decibel-watts)
# 'power'        -> Watts (Linear power, converted from power_db using 10^(power_db / 10))
# 'mask'         -> Hexadecimal bitmask (Indicates contributing stations)
# 'stations'     -> Comma-separated string (Decoded station names from the mask)
# 'x'            -> Meters (ECEF X-coordinate in WGS84)
# 'y'            -> Meters (ECEF Y-coordinate in WGS84)
# 'z'            -> Meters (ECEF Z-coordinate in WGS84)


start_time = datetime.datetime(2022, 7, 12, 1, 3, tzinfo=datetime.timezone.utc).timestamp()
end_time = datetime.datetime(2022, 9, 2, 19, 3, tzinfo=datetime.timezone.utc).timestamp()

# Build filter list for time_unix boundaries.
filters = [
    ("time_unix", ">=", start_time),
    ("time_unix", "<=", end_time),
    ("reduced_chi2", "<", 1.0),
    ("num_stations", ">=", 6)
]

events = database_parser.query_events_as_dataframe(filters)
print(events)

