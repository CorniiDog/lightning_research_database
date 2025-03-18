import os
import database_parser
import lightning_bucketer
import lightning_plotters
import logger
import datetime
import shutil

lightning_data_folder = "lylout_files"
data_extension = ".dat"

CPU_PCT = 0.7 # Percentage of CPU's to use for multiprocessing when necessary

os.makedirs(lightning_data_folder, exist_ok=True)  # Ensure that it exists

dat_file_paths = database_parser.get_dat_files_paths(
    lightning_data_folder, data_extension
)

for file_path in dat_file_paths:
    if not logger.is_logged(file_path):
        print(file_path, "not appropriately added to the database. Adding...")
        database_parser.parse_lylout(file_path)
        logger.log_file(file_path)  # Mark as logged and unmodified
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


start_time = datetime.datetime(
    2022, 7, 12, 0, 0, tzinfo=datetime.timezone.utc
).timestamp()
end_time = datetime.datetime(
    2022, 7, 12, 23, 0, tzinfo=datetime.timezone.utc
).timestamp()

# Build filter list for time_unix boundaries.
filters = [
    ("time_unix", ">=", start_time),
    ("time_unix", "<=", end_time),
    ("reduced_chi2", "<", 2.0),
    ("num_stations", ">=", 7),
    ("alt", "<=", 17000),  # 20 km = 20000m
    ("alt", ">", 0),  # Above ground
    ("power_db", ">", -4),  # dBW
    ("power_db", "<", 50),  # dBW
]

events = database_parser.query_events_as_dataframe(filters)
print(events)

params = {
    "max_lightning_dist": 50000,  # meters
    "max_lightning_speed": 299792.458,  # m/s
    "min_lightning_speed": 0,  # m/s
    "min_lightning_points": 300,  # The minimum number of points to pass the minimum amount
    "max_lightning_time_threshold": 0.15,  # seconds between points
}

lightning_bucketer.USE_CACHE = (
    True  # Generate cache of result to save time for future requests
)
bucketed_strikes_indeces = lightning_bucketer.bucket_dataframe_lightnings(
    events, params=params
)

# Sort the bucketed strikes indices by the length of each sublist in descending order.
bucketed_strikes_indeces_sorted = sorted(
    bucketed_strikes_indeces, key=len, reverse=True
)

len_strikes = len(bucketed_strikes_indeces_sorted)
print(f"Number of strikes matching criteria: {len_strikes}")

if len(bucketed_strikes_indeces_sorted) == 0:
    print("Data too restrained. ")
else:
    # Optionally, print each bucket with its length.
    for i, strike in enumerate(bucketed_strikes_indeces_sorted):
        start_time_unix = events.iloc[strike[0]]["time_unix"]
        print(f"Bucket {i}: Length = {len(strike)}: Time = {start_time_unix}")

    print("Plotting strike points over time")
    lightning_plotters.plot_strikes_over_time(bucketed_strikes_indeces_sorted, events)

    print("Plotting all strikes into a readable heatmap.")
    strike_dir = "strikes"
    num_cores = int(max(CPU_PCT*os.cpu_count(), 1)) # calculate the number of cores to use

    # Remove the strikes directory if it exists
    if os.path.exists(strike_dir):
        shutil.rmtree(strike_dir)
    
    os.makedirs(strike_dir, exist_ok=True)
    lightning_plotters.plot_all_strikes(bucketed_strikes_indeces_sorted, events, strike_dir,num_cores)

    print("Exporting largest instance on file")
    # Just plot the largest instance
    lightning_plotters.plot_avg_power_map(bucketed_strikes_indeces_sorted[0], events)


    print("Finished generating plots")
