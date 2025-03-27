# Run in background: python main.py > output.log 2>&1 & disown
# List all files in directory './' and sizes: du -h --max-depth=1 ./ | sort -hr

import os
import database_parser
import lightning_bucketer
import lightning_plotters
import lightning_stitcher
import logger
import datetime
import shutil
import time

####################################################################################
 #
  # About: A top-down view of what is going on
 #
####################################################################################
"""
This program processes LYLOUT data files, such as "LYLOUT_20220712_pol.exported.dat"

1. It first reads through all the files, and then puts all of the points into an 
SQLite database

2. Then, with user-specified filters, the user extracts a pandas DataFrame 
(DataFrame "events") from the SQLite database that meets all of the 
filter criteria. 

3. Afterwards, with user-specified parameters, the lightning_bucketer processes 
all of the "events" data to return a list of lightning strikes, which each 
lightning strike is simply a list of indices for the "events" DataFrame
(a list of lists).

4. You can use the events with the lightning strikes data to plot data or analyze 
the data. Examples in the code and comments below show how to do so.
"""
####################################################################################
 #
  # About: How to run
 #
####################################################################################
"""
Look at README.md for further details on how to install prerequisited and also run
the file.
"""
####################################################################################

# Mark process tart time
process_start_time = time.time()

# For certain situations, the program may use multiple CPU's to speed up the processing.
# You can designate a percentage, or explicitly define in num_cores
CPU_PCT = 0.9  # Percentage of CPU's to use for multiprocessing when necessary
NUM_CORES = int(max(CPU_PCT * os.cpu_count(), 1))

####################################################################################
 #
  # Retreiving ".dat" files from "lylout_files" directory and adding to the SQLite Database
 #
####################################################################################

# This retreives all file paths from "lylout_files" directory
lightning_data_folder = "lylout_files"  # Should put all "LYLOUT..." files within this directory (drag/drop into it)
data_extension = ".dat"
os.makedirs(lightning_data_folder, exist_ok=True)  # Ensure that it exists

dat_file_paths = database_parser.get_dat_files_paths(lightning_data_folder, data_extension)

# The SQLite database file
# (This file doesn't need to exist. Will auto-generate a new one)
cache_dir = "cache_dir"
os.makedirs(cache_dir, exist_ok=True)

DB_PATH = os.path.join(cache_dir, "lylout_db.db")
logger.LOG_FILE = os.path.join(cache_dir, "file_log.json")

# NOTE: If you want to delete the lightning database, delete "lylout_db.db"
# and "file_log.json" if they exist in the Python project directory

for file_path in dat_file_paths:
    # If the file is not already processed into the SQLite database
    # (basically meaning non-identical, or non-existing hashes determined by "file_log.json")
    if not logger.is_logged(file_path):
        print(file_path, "not appropriately added to the database. Adding...")
        database_parser.parse_lylout(file_path, DB_PATH)
        logger.log_file(file_path)  # Log the file for no redundant re-processing into the database

    # If the file is already processed, do nothing
    else:
        print(file_path, "was parsed and added to the database already")

####################################################################################
 #
  # List of headers (These headers are variables that can be applied to the filters list)
 #
####################################################################################
print("Headers:", database_parser.get_headers(DB_PATH))

# Column/Header descriptions:
# 'time_unix'    -> float   Seconds (Unix timestamp, UTC)
# 'lat'          -> float   Degrees (WGS84 latitude)
# 'lon'          -> float   Degrees (WGS84 longitude)
# 'alt'          -> float   Meters (Altitude above sea level)
# 'reduced_chi2' -> float   Reduced chi-square goodness-of-fit metric
# 'num_stations' -> int     Count (Number of contributing stations)
# 'power_db'     -> float   Decibels (dBW) (Power of the detected event in decibel-watts)
# 'power'        -> float   Watts (Linear power, converted from power_db using 10^(power_db / 10))
# 'mask'         -> str     Hexadecimal bitmask (Indicates contributing stations)
# 'stations'     -> str     Comma-separated string (Decoded station names from the mask)
# 'x'            -> float   Meters (ECEF X-coordinate in WGS84)
# 'y'            -> float   Meters (ECEF Y-coordinate in WGS84)
# 'z'            -> float   Meters (ECEF Z-coordinate in WGS84)

####################################################################################
 #
  # Filter params for extracting data points from the SQLite database
 #
####################################################################################

start_time = datetime.datetime(
    2022, 7, 12, 0, 0, tzinfo=datetime.timezone.utc
).timestamp()  # Timestamp converts to unix (float)

end_time = datetime.datetime(
    2022, 7, 12, 23, 0, tzinfo=datetime.timezone.utc
).timestamp()  # Timestamp converts to unix (float)

# Build filter list for time_unix boundaries.
# Look at "List of headers" above for additional
# Filterings
filters = [
    ("time_unix", ">=", start_time),  # In unix
    ("time_unix", "<=", end_time),  # In unix
    ("reduced_chi2", "<", 2.0,),  # The chi^2 (reliability index) value to accept the data
    ("num_stations", ">=", 7),  # Number of stations that have visibly seen the strike
    ("alt", "<=", 18000),  # alt is in meters. Therefore 20 km = 20000m
    ("alt", ">", 0),  # Above ground
    ("power_db", ">", -4),  # In dBW
    ("power_db", "<", 50),  # In dBW
]

# Events is a pandas DataFrame that represents all results determined from the filters above
events = database_parser.query_events_as_dataframe(filters, DB_PATH)
print(events)

####################################################################################
 #
  # Identifying the lightning strikes
 #
####################################################################################

# Additional parameters that determines "What points make up a single lightning strike"
# They are explicitly defined
params = {
    "max_lightning_dist": 5000,  # max distance between two points to determine it being involved in the same strike
    "max_lightning_speed": 299792.458,  # max speed between two points in m/s (essentially dx/dt)
    "min_lightning_speed": 0,  # min speed between two points in m/s (essentially dx/dt)
    "min_lightning_points": 500,  # The minimum number of points to pass the system as a "lightning strike"
    "max_lightning_time_threshold": 0.08,  # max number of seconds between points 
    "max_lightning_duration": 20, # max seconds that define an entire lightning strike. This is essentially a "time window" for all of the points to fill the region that determines a "lightning strike"
}

lightning_bucketer.USE_CACHE = True  # Generate cache of result to save time for future identical (one-to-one exact) requests
lightning_bucketer.RESULT_CACHE_FILE = os.path.join(cache_dir, "result_cache.pkl")
# To delete the cache of all of the save data (as it accumulates over time), run:
# ```
# lightning_bucketer.delete_result_cache()
# ```

# The parameters above will be passed to return bucketed_strikes_indices, defined by type list[list[int]]
# which is a list of all lightning stikes, such that each lightning strike is a list of all indexes
bucketed_strikes_indices = lightning_bucketer.bucket_dataframe_lightnings(events, **params)

# Example: To get a Pandas DataFrame of the first strike in the list, you do:
# ```
# first_strikes = events.iloc[bucketed_strikes_indices[0]]
# ```
#
# Example 2: Iterating through all lightning strikes:
# ```
# for i in range(len(bucketed_strikes_indices)):
#   sub_strike = events.iloc[bucketed_strikes_indices[i]]
#   # Process the dataframe however you please of the designated lightning strike
# ```


# Sort the bucketed strikes indices by the length of each sublist in descending order.
# (Strikes with most points first)
bucketed_strikes_indices_sorted_by_len = sorted(bucketed_strikes_indices, key=len, reverse=True)

# Example: To get a Pandas DataFrame of the first strike in the list, you do:
# ```
# first_strikes = events.iloc[bucketed_strikes_indices_sorted[0]]
# ```
#
# Example 2: Iterating through all lightning strikes:
# ```
# for i in range(len(bucketed_strikes_indices_sorted)):
#   sub_strike = events.iloc[bucketed_strikes_indices_sorted[i]]
#   # Process the dataframe however you please of the designated lightning strike
# ```


print(f"Number of strikes matching criteria: {len(bucketed_strikes_indices_sorted_by_len)}")

# Stop the program if the data is too restrained
if len(bucketed_strikes_indices_sorted_by_len) == 0:
    print("Data too restrained.")

# Print each bucket with its length to terminal
for i, strike in enumerate(bucketed_strikes_indices_sorted_by_len):
    start_time_unix = events.iloc[strike[0]]["time_unix"]
    print(f"Bucket {i}: Length = {len(strike)}: Time = {start_time_unix}")

print("Created buckets of nodes that resemble a lightning strike")

print("Stitching lightning strikes")
bucketed_lightning_correlations = lightning_stitcher.stitch_lightning_strikes(bucketed_strikes_indices_sorted_by_len, events, **params)

print("Finished generating stitchings of the lightning strike")

####################################################################################
 #
  # Plotting and exporting
 #
####################################################################################
print("Exporting CSV data")

csv_dir = "strikes_csv_files"

if os.path.exists(csv_dir):
    shutil.rmtree(csv_dir)

os.makedirs(csv_dir, exist_ok=True)

lightning_bucketer.export_as_csv(bucketed_strikes_indices_sorted_by_len, events, output_dir=csv_dir)

print("Finished exporting as CSV")

export_dir = "export"
os.makedirs(export_dir, exist_ok=True)

# Exporting a chart of strikes over time
print("Plotting strike points over time")
export_path = os.path.join(export_dir, "strike_pts_over_time")
lightning_plotters.plot_strikes_over_time(bucketed_strikes_indices_sorted_by_len, events, output_filename=export_path+".png")

# Exporting most points
print("Exporting largest instance")
largest_strike = bucketed_strikes_indices_sorted_by_len[0]
export_path = os.path.join(export_dir, "most_pts")
lightning_plotters.plot_avg_power_map(largest_strike, events, output_filename=export_path+".png", transparency_threshold=-1)
lightning_plotters.generate_strike_gif(largest_strike, events, output_filename=export_path+".gif", transparency_threshold=-1)

print("Exporting largest stitched instance")

export_path = os.path.join(export_dir, "most_pts_stitched")
lightning_plotters.plot_lightning_stitch(bucketed_lightning_correlations[0], events, export_path+".png")
lightning_plotters.plot_lightning_stitch_gif(bucketed_lightning_correlations[0], events, output_filename=export_path+".gif")


strike_dir = "strikes"

# Remove the strikes directory if it exists
# This prevents existing plots from intertwining
# with the existing plots.
if os.path.exists(strike_dir):
    shutil.rmtree(strike_dir)

os.makedirs(strike_dir, exist_ok=True)

print("Plotting strikes as a heatmap")
lightning_plotters.plot_all_strikes(bucketed_strikes_indices_sorted_by_len, events, strike_dir, NUM_CORES, sigma=1.5, transparency_threshold=-1)
lightning_plotters.plot_all_strikes(bucketed_strikes_indices_sorted_by_len, events, strike_dir, NUM_CORES, as_gif=True, sigma=1.5, transparency_threshold=-1)

print("Plotting all strike stitchings")

strike_stitchings_dir = "strike_stitchings"

# Remove the strike_stitchings directory if it exists
# This prevents existing plots from intertwining
# with the existing plots.
if os.path.exists(strike_stitchings_dir):
    shutil.rmtree(strike_stitchings_dir)

lightning_plotters.plot_all_strike_stitchings(bucketed_lightning_correlations, events, strike_stitchings_dir, NUM_CORES)
lightning_plotters.plot_all_strike_stitchings(bucketed_lightning_correlations, events, strike_stitchings_dir, NUM_CORES, as_gif=True)

print("Finished generating plots")

process_time = time.time() - process_start_time
print(f"Process time: {process_time:.2f} seconds.")