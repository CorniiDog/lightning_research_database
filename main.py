import os
import database_parser
import lightning_bucketer
import lightning_plotters
import logger
import datetime
import shutil

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
lightning strike is simply a list of indeces for the "events" DataFrame
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
CPU_PCT = 0.7  # Percentage of CPU's to use for multiprocessing when necessary
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

dat_file_paths = database_parser.get_dat_files_paths(
    lightning_data_folder, data_extension
)

# The SQLite database file
# (This file doesn't need to exist. Will auto-generate a new one)
DB_PATH = "lylout_db.db"

# NOTE: If you want to delete the lightning database, delete "lylout_db.db"
# and "file_log.json" if they exist in the Python project directory

for file_path in dat_file_paths:
    # If the file is not already processed into the SQLite database
    # (basically meaning non-identical, or non-existing hashes determined by "file_log.json")
    if not logger.is_logged(file_path):
        print(file_path, "not appropriately added to the database. Adding...")
        database_parser.parse_lylout(file_path, DB_PATH)
        logger.log_file(
            file_path
        )  # Log the file for no redundant re-processing into the database

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
    ("alt", "<=", 17000),  # alt is in meters. Therefore 20 km = 20000m
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
    "max_lightning_dist": 50000,  # meters
    "max_lightning_speed": 299792.458,  # m/s
    "min_lightning_speed": 0,  # m/s
    "min_lightning_points": 300,  # The minimum number of points to pass the minimum amount
    "max_lightning_time_threshold": 0.15,  # seconds between points
}

lightning_bucketer.USE_CACHE = True  # Generate cache of result to save time for future identical (one-to-one exact) requests

# To delete the cache of all of the save data (as it accumulates over time), run:
# ```
# lightning_bucketer.delete_result_cache()
# ```

# The parameters above will be passed to return bucketed_strikes_indeces, defined by type list[list[int]]
# which is a list of all lightning stikes, such that each lightning strike is a list of all indexes
bucketed_strikes_indeces = lightning_bucketer.bucket_dataframe_lightnings(
    events, params=params
)
# Example: To get a Pandas DataFrame of the first strike in the list, you do:
# ```
# first_strikes = events.iloc[bucketed_strikes_indeces[0]]
# ```
#
# Example 2: Iterating through all lightning strikes:
# ```
# for i in range(len(bucketed_strikes_indeces)):
#   sub_strike = events.iloc[bucketed_strikes_indeces[i]]
#   # Process the dataframe however you please of the designated lightning strike
# ```


# Sort the bucketed strikes indices by the length of each sublist in descending order.
# (Strikes with points first)
bucketed_strikes_indeces_sorted = sorted(
    bucketed_strikes_indeces, key=len, reverse=True
)
# Example: To get a Pandas DataFrame of the first strike in the list, you do:
# ```
# first_strikes = events.iloc[bucketed_strikes_indeces_sorted[0]]
# ```
#
# Example 2: Iterating through all lightning strikes:
# ```
# for i in range(len(bucketed_strikes_indeces_sorted)):
#   sub_strike = events.iloc[bucketed_strikes_indeces_sorted[i]]
#   # Process the dataframe however you please of the designated lightning strike
# ```

print(f"Number of strikes matching criteria: {len(bucketed_strikes_indeces_sorted)}")

# Stop the program if the data is too restrained
if len(bucketed_strikes_indeces_sorted) == 0:
    print("Data too restrained. ")
    exit()

# Print each bucket with its length to terminal
for i, strike in enumerate(bucketed_strikes_indeces_sorted):
    start_time_unix = events.iloc[strike[0]]["time_unix"]
    print(f"Bucket {i}: Length = {len(strike)}: Time = {start_time_unix}")

####################################################################################
 #
  # Plotting and exporting
 #
####################################################################################

print("Plotting strike points over time")
lightning_plotters.plot_strikes_over_time(bucketed_strikes_indeces_sorted, events)

print("Plotting all strikes into a readable heatmap.")
strike_dir = "strikes"

# Remove the strikes directory if it exists
if os.path.exists(strike_dir):
    shutil.rmtree(strike_dir)

os.makedirs(strike_dir, exist_ok=True)
lightning_plotters.plot_all_strikes(
    bucketed_strikes_indeces_sorted, events, strike_dir, NUM_CORES
)

print("Exporting largest instance on file")
# Just plot the largest instance
lightning_plotters.plot_avg_power_map(bucketed_strikes_indeces_sorted[0], events)

print("Finished generating plots")
