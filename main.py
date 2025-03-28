# Run in background: python main.py > output.log 2>&1 & disown
# List all files in directory './' and sizes: du -h --max-depth=1 ./ | sort -hr

# GOES 3D data stitching

import os
import database_parser
import lightning_bucketer
import lightning_plotters
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

files = os.listdir(lightning_data_folder)
if len(files) == 0:
    print(f"Please put lightning LYLOUT files in the directory '{lightning_data_folder}'")
    exit()

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

# Mark process start time
process_start_time = time.time()

####################################################################################
 #
  # Filter params for extracting data points from the SQLite database
 #
####################################################################################

start_time = datetime.datetime(
    2020, 4, 29, 0, 0, tzinfo=datetime.timezone.utc
).timestamp()  # Timestamp converts to unix (float)

end_time = datetime.datetime(
    2020, 4, 29, 23, 59, tzinfo=datetime.timezone.utc
).timestamp()  # Timestamp converts to unix (float)

# Build filter list for time_unix boundaries.
# Look at "List of headers" above for additional
# Filterings
filters = [
    ("time_unix", ">=", start_time),  # In unix
    ("time_unix", "<=", end_time),  # In unix
    ("reduced_chi2", "<", 5.0,),  # The chi^2 (reliability index) value to accept the data
    ("num_stations", ">=", 5),  # Number of stations that have visibly seen the strike
    ("alt", "<=", 24000),  # alt is in meters. Therefore 20 km = 20000m
    ("alt", ">", 0),  # Above ground
    ("power_db", ">", -4),  # In dBW
    ("power_db", "<", 50),  # In dBW
]

# Events is a pandas DataFrame that represents all results determined from the filters above
events = database_parser.query_events_as_dataframe(filters, DB_PATH)
print("Events:", events)

if events.empty:
    print("Data too restrained")

####################################################################################
 #
  # Identifying the lightning strikes
 #
####################################################################################

# Additional parameters that determines "What points make up a single lightning strike"
# They are explicitly defined
params = {
    # Creating an initial lightning strike
    "max_lightning_dist": 30000,  # Max distance between two points to determine it being involved in the same strike
    "max_lightning_speed": 1.4e8,  # Max speed between two points in m/s (essentially dx/dt)
    "min_lightning_speed": 0,  # Min speed between two points in m/s (essentially dx/dt)
    "min_lightning_points": 300,  # The minimum number of points to pass the system as a "lightning strike"
    "max_lightning_time_threshold": 0.3,  # Max number of seconds between points 
    "max_lightning_duration": 40, # Max seconds that define an entire lightning strike. This is essentially a "time window" for all of the points to fill the region that determines a "lightning strike"

    # Combining intercepting lightning strike data filtering
    "combine_strikes_with_intercepting_times": True, # Set to true to ensure that strikes with intercepting times get combined. 
    "intercepting_times_extension_buffer": 3, # Number of seconds of additional overlap to allow an additional strike to be involved
    "intercepting_times_extension_max_distance": 200000, # The max distance between the start point of one lightning strike and at least one from the entirety of another lightning strike's points
}

lightning_bucketer.USE_CACHE = True  # Generate cache of result to save time for future identical (one-to-one exact) requests
lightning_bucketer.RESULT_CACHE_FILE = os.path.join(cache_dir, "result_cache.pkl")
lightning_bucketer.NUM_CORES = NUM_CORES # Set number of CPU cores for faster processing
lightning_bucketer.NUM_CHUNKS = 40 # Combine the (possibly) thousands of buckets into 'n' chunks for multiprocessing
# To delete the cache of all of the save data (as it accumulates over time), run:
# ```
# lightning_bucketer.delete_result_cache()
# ```

# The parameters above will be passed to return bucketed_strikes_indices, defined by type list[list[int]]
# which is a list of all lightning stikes, such that each lightning strike is a list of all indexes
bucketed_strikes_indices, bucketed_lightning_correlations = lightning_bucketer.bucket_dataframe_lightnings(events, **params)

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

process_time = time.time() - process_start_time
print(f"Process time: {process_time:.2f} seconds.")

# Stop the program if the data is too restrained
if len(bucketed_strikes_indices) == 0:
    print("Data too restrained.")
    exit()

# Print each bucket with its length to terminal
for i, strike in enumerate(bucketed_strikes_indices):
    start_time_unix = events.iloc[strike[0]]["time_unix"]
    print(f"Bucket {i}: Length = {len(strike)}: Time = {start_time_unix}")

print("Created buckets of nodes that resemble a lightning strike")

print("Finished generating stitchings of the lightning strike")

####################################################################################
 #
  # Plotting and exporting
 #
####################################################################################

# Only export plot data with more than n datapoints
MAKE_PLOTS_WITH_MORE_PTS_THAN = 1000
bucketed_strikes_indices = [lst for lst in bucketed_strikes_indices if len(lst) > MAKE_PLOTS_WITH_MORE_PTS_THAN]
bucketed_lightning_correlations = [lst for lst in bucketed_lightning_correlations if len(lst) > MAKE_PLOTS_WITH_MORE_PTS_THAN]
#####

print("Exporting CSV data")

csv_dir = "strikes_csv_files"

if os.path.exists(csv_dir):
    shutil.rmtree(csv_dir)

os.makedirs(csv_dir, exist_ok=True)

lightning_bucketer.export_as_csv(bucketed_strikes_indices, events, output_dir=csv_dir)

print("Finished exporting as CSV")

export_dir = "export"
os.makedirs(export_dir, exist_ok=True)

# Exporting a chart of strikes over time
print("Plotting strike points over time")
export_path = os.path.join(export_dir, "strike_pts_over_time")
lightning_plotters.plot_strikes_over_time(bucketed_strikes_indices, events, output_filename=export_path+".png")

# Exporting most points
print("Exporting largest instance")
export_path = os.path.join(export_dir, "most_pts")
bucketed_strikes_indices_largest = max(bucketed_strikes_indices, key=len)
lightning_plotters.plot_avg_power_map(bucketed_strikes_indices_largest, events, output_filename=export_path+".png", transparency_threshold=-1)
lightning_plotters.generate_strike_gif(bucketed_strikes_indices_largest, events, output_filename=export_path+".gif", transparency_threshold=-1)

print("Exporting largest stitched instance")

export_path = os.path.join(export_dir, "most_pts_stitched")
bucketed_lightning_correlations_largest = max(bucketed_lightning_correlations, key=len)
lightning_plotters.plot_lightning_stitch(bucketed_lightning_correlations_largest, events, export_path+".png")
lightning_plotters.plot_lightning_stitch_gif(bucketed_lightning_correlations_largest, events, output_filename=export_path+".gif")

# Exporting entirely
print("Exporting all strikes")
export_path = os.path.join(export_dir, "all_pts")
bucketed_strikes_indices_combined = [index for strike in bucketed_strikes_indices for index in strike]
lightning_plotters.plot_avg_power_map(bucketed_strikes_indices_combined, events, output_filename=export_path+".png", transparency_threshold=-1)
lightning_plotters.generate_strike_gif(bucketed_strikes_indices_combined, events, output_filename=export_path+".gif", transparency_threshold=-1)

print("Number of points within timeframe:", len(bucketed_strikes_indices_combined))


# Commented out: Not really that optimized to use yet
# print("Exporting all stitched instances")
# export_path = os.path.join(export_dir, "all_pts_stitched")
# bucketed_lightning_correlations_combined = [index for strike in bucketed_lightning_correlations for index in strike] 
# lightning_plotters.plot_lightning_stitch(bucketed_lightning_correlations_combined, events, output_filename=export_path+".png")
# lightning_plotters.plot_lightning_stitch_gif(bucketed_lightning_correlations_combined, events, output_filename=export_path+".gif")

strike_dir = "strikes"

# Remove the strikes directory if it exists
# This prevents existing plots from intertwining
# with the existing plots.
if os.path.exists(strike_dir):
    shutil.rmtree(strike_dir)

os.makedirs(strike_dir, exist_ok=True)

print("Plotting strikes as a heatmap")
lightning_plotters.plot_all_strikes(bucketed_strikes_indices, events, strike_dir, NUM_CORES, sigma=1.5, transparency_threshold=-1)
lightning_plotters.plot_all_strikes(bucketed_strikes_indices, events, strike_dir, NUM_CORES, as_gif=True, sigma=1.5, transparency_threshold=-1)

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