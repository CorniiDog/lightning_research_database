import os
import database_parser
import lightning_bucketer
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
    ("reduced_chi2", "<", 2.0),
    ("num_stations", ">=", 6),
    ("alt", "<=", 20000), # 20 km = 20000m
    ("alt", ">", 0), # Above ground
    ("power_db",  ">", -4), # dBW
    ("power_db",  "<", 50), # dBW

]

events = database_parser.query_events_as_dataframe(filters)
print(events)

params = {
  "max_lightning_dist": 50000, # meters
  "max_lightning_speed": 299792.458, # m/s
  "min_lightning_speed": 0, # m/s
  "min_lightning_points": 300, # The minimum number of points to pass the minimum amount
  "max_lightning_time_threshold": 0.2 # seconds between points
}

lightning_bucketer.USE_CACHE = True # Generate cache of result to save time for future requests
bucketed_strikes_indeces = lightning_bucketer.bucket_dataframe_lightnings(events, params=params)


