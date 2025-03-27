try:
  import cupy as cp
except ModuleNotFoundError:
    print(
        "CuPy is not installed.\n"
        "Please install the appropriate version for your system.\n"
        "Refer to the project's README for detailed instructions:\n"
        "    ./README.md"
    )
    exit(1)
    
import numpy as np
import pandas as pd
import pickle as pkl
import os
import hashlib
import re
import datetime
from tqdm import tqdm


def _bucket_dataframe_lightnings(
    df: pd.DataFrame,
    max_time_threshold,
    max_lightning_duration,
    max_dist_between_pts,
    max_speed,
    min_speed=0,
    min_pts=0,
) -> list[list[int]]:
    """
    Buckets the dataframe into groups of lightning strikes based on temporal and spatial constraints,
    now incorporating a maximum lightning duration to prevent grouping events over an unrealistic time span.

    This function performs the following steps:
      1. Sorts the dataframe by 'time_unix'.
      2. Converts the time values to a GPU array and computes the difference between consecutive time points.
      3. Uses a vectorized cumulative sum to group events that are close in time (i.e., where the time difference
         is below the max_time_threshold).
      4. Iterates through each time group and further clusters events into potential lightning strikes based on:
         - Minimum number of points in a group (min_pts).
         - Spatial proximity (max_dist_between_pts).
         - Speed constraints (min_speed and max_speed).
         - Maximum lightning duration (max_lightning_duration): If the duration of a subgroup exceeds this value,
           it is finalized and a new subgroup is started.
      5. Returns a list of lists, where each sublist contains indices from the original dataframe corresponding to
         a detected lightning strike.

    Parameters:
      df (pandas.DataFrame): DataFrame containing lightning event data.
      max_time_threshold (float): Maximum allowed time difference between consecutive points (in seconds).
      max_lightning_duration (float): Maximum duration (in seconds) for a lightning strike.
      max_dist_between_pts (float): Maximum allowed spatial distance (in meters) between points.
      max_speed (float): Maximum allowed speed (in m/s) between points.
      min_speed (float, optional): Minimum allowed speed (in m/s) between points. Defaults to 0.
      min_pts (int, optional): Minimum number of points required for a valid lightning strike. Defaults to 0.

    Returns:
      list[list[int]]: A list where each sublist contains the indices of the dataframe representing a lightning strike.
    """
    df.sort_values(by="time_unix", inplace=True)
    time_unix_gpu = cp.asarray(df["time_unix"].values)
    delta_t = cp.diff(time_unix_gpu)

    # Group events by time threshold using cumulative sum.
    time_groups = cp.concatenate(
        (
            cp.array([0], dtype=cp.int32),
            cp.cumsum((delta_t > max_time_threshold).astype(cp.int32)),
        )
    )
    print(time_groups)
    print("Processing the buckets.")

    group_ids = cp.asnumpy(time_groups)
    unique_groups = np.unique(group_ids)
    total_unique_groups = len(unique_groups)

    lightning_strikes = []

    for group in tqdm(unique_groups, total=total_unique_groups):
        group_indices = np.where(group_ids == group)[0]

        # Skip groups with fewer points than required.
        if len(group_indices) < min_pts:
            continue

        group_df = df.iloc[group_indices]

        x_vals = group_df["x"].values
        y_vals = group_df["y"].values
        z_vals = group_df["z"].values
        unix_vals = group_df["time_unix"].values

        # Initialize list to hold subgroups (potential lightning strikes) for this time group.
        sub_groups = []

        for j in range(len(x_vals)):
            event_x = x_vals[j]
            event_y = y_vals[j]
            event_z = z_vals[j]
            event_unix = unix_vals[j]

            # Before processing the current event, finalize subgroups that have exceeded max_lightning_duration.
            new_sub_groups = []
            for sg in sub_groups:
                # Check if current event time exceeds the allowed duration from subgroup's first event.
                delta = event_unix - sg["unix"][0]
                if delta > max_lightning_duration:
                    # Finalize subgroup if it meets the minimum points requirement.
                    if len(sg["indices"]) >= min_pts:
                        lightning_strikes.append([group_indices[idx] for idx in sg["indices"]])
                    # Do not retain this subgroup.
                else:
                    new_sub_groups.append(sg)
            sub_groups = new_sub_groups

            found = False
            # Attempt to add the event to an existing subgroup.
            for sg in sub_groups:
                # Avoid square rooting  to improve the calculations
                distances_squared = ((event_x - sg["x"]) ** 2 + (event_y - sg["y"]) ** 2 + (event_z - sg["z"]) ** 2)

                max_dist_between_pts_squared =  (max_dist_between_pts ** 2)
                mask = (distances_squared <= max_dist_between_pts_squared)
                if cp.any(mask):
                    # Check speed constraints only for points within spatial threshold.
                    dt = cp.abs(event_unix - sg["unix"])
                    dt = cp.where(dt == 0, 1e-6, dt)
                    speeds_squared = distances_squared / (dt ** 2)
                    
                    min_speed_squared = (min_speed ** 2)
                    max_speed_squared = (max_speed ** 2)
                    if cp.any((speeds_squared >= min_speed_squared) & (speeds_squared <= max_speed_squared)):
                        # Ensure adding the event doesn't violate max_lightning_duration.
                        if event_unix - sg["unix"][0] <= max_lightning_duration:
                            sg["indices"].append(j)
                            sg["x"] = cp.concatenate([sg["x"], cp.array([event_x])])
                            sg["y"] = cp.concatenate([sg["y"], cp.array([event_y])])
                            sg["z"] = cp.concatenate([sg["z"], cp.array([event_z])])
                            sg["unix"] = cp.concatenate([sg["unix"], cp.array([event_unix])])
                            found = True
                            break

            # If event did not fit in any subgroup, start a new subgroup.
            if not found:
                sub_groups.append(
                    {
                        "indices": [j],
                        "x": cp.array([event_x]),
                        "y": cp.array([event_y]),
                        "z": cp.array([event_z]),
                        "unix": cp.array([event_unix]),
                    }
                )

        # After processing all events in the group, finalize remaining subgroups.
        for sg in sub_groups:
            if len(sg["indices"]) >= min_pts:
                final_subgroup = [group_indices[idx] for idx in sg["indices"]]
                lightning_strikes.append(final_subgroup)

    print("Passed groups:", len(lightning_strikes))
    return lightning_strikes



# The result cache file, as a pkl
RESULT_CACHE_FILE = "result_cache.pkl"


def _compute_cache_key(df: pd.DataFrame, params: dict) -> str:
    """
    Compute a unique cache key for the dataframe and parameters.

    The key is based on:
      - The shape of the dataframe.
      - The minimum and maximum 'time_unix' values in the dataframe.
      - The sorted parameters provided.

    Parameters:
      df (pandas.DataFrame): DataFrame containing lightning event data.
      params (dict): Dictionary of parameters used for bucketing.

    Returns:
      str: An MD5 hash string representing the unique cache key.
    """
    key_str = f"{df.shape}_{df['time_unix'].min()}_{df['time_unix'].max()}_{sorted(params.items())}"
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def delete_result_cache():
    """
    Delete the cached result file from the filesystem.

    This function removes the file specified by 'RESULT_CACHE_FILE' if it exists.

    Returns:
      None
    """
    os.remove(RESULT_CACHE_FILE)


def _get_result_cache(df, params) -> list[list[int]] | None:
    """
    Retrieve the cached bucketing result if available.

    This function computes a cache key based on the dataframe and parameters, then checks if
    a cached result exists in the cache file. If a matching cache is found, it is returned.

    Parameters:
      df (pandas.DataFrame): DataFrame containing lightning event data.
      params (dict): Dictionary of parameters used for bucketing.

    Returns:
      list of list of int or None: Cached bucketing result if available; otherwise, None.
    """
    key = _compute_cache_key(df, params)
    if os.path.exists(RESULT_CACHE_FILE):
        try:
            with open(RESULT_CACHE_FILE, "rb") as f:
                cache = pkl.load(f)
            if key in cache:
                print("Cache hit.")
                return cache[key]
        except Exception as e:
            print(f"Cache load error: {e}")
    return None


def save_result_cache(df, params, result):
    """
    Save the bucketing result in the cache with a computed key.

    This function computes a cache key from the dataframe and parameters, updates the cache,
    and writes the cache to disk.

    Parameters:
      df (pandas.DataFrame): DataFrame containing lightning event data.
      params (dict): Dictionary of parameters used for bucketing.
      result (list of list of int): The bucketing result to be cached.

    Returns:
      None
    """
    key = _compute_cache_key(df, params)
    cache = {}
    if os.path.exists(RESULT_CACHE_FILE):
        try:
            with open(RESULT_CACHE_FILE, "rb") as f:
                cache = pkl.load(f)
        except Exception as e:
            print(f"Cache load error: {e}")
            cache = {}
    cache[key] = result
    with open(RESULT_CACHE_FILE, "wb") as f:
        pkl.dump(cache, f)


# Set to True to use the cache. Otherwise, set to False
USE_CACHE = True


def bucket_dataframe_lightnings(df: pd.DataFrame, **params) -> list[list[int]]:
    """
    Bucket lightning strikes in the dataframe using provided temporal and spatial parameters.

    The function groups lightning events into strikes by:
      1. Dividing events based on time differences (using max_lightning_time_threshold).
      2. Clustering events within each group based on spatial distance (max_lightning_dist) and speed constraints
         (max_lightning_speed and min_lightning_speed).
      3. Filtering clusters that do not meet the minimum number of points (min_lightning_points).

    Optionally, the computed results are cached to speed up future identical computations.

    Expected parameters in 'params':
      - max_lightning_dist (float): Maximum allowed distance between consecutive points (meters).
      - max_lightning_speed (float): Maximum allowed speed (m/s).
      - min_lightning_speed (float): Minimum allowed speed (m/s).
      - min_lightning_points (int): Minimum number of points required to qualify as a lightning strike.
      - max_lightning_time_threshold (float): Maximum allowed time difference between consecutive points (seconds).
      - max_lightning_duration (float): Max time allowed for a "lightning strike" window as a whole (seconds). (Lightning world record is roughly 18s)
    Parameters:
      df (pandas.DataFrame): DataFrame containing lightning event data with required headers.
      **params: Additional keyword arguments for controlling bucketing behavior.

    Returns:
      list[list[int]]: A list where each sublist contains the indices of the dataframe representing a lightning strike.
    """

    if USE_CACHE:
        result = _get_result_cache(df, params)
        if result:
            print("Using cached result from earlier")
            return result
        
    result = _bucket_dataframe_lightnings(
        df,
        max_time_threshold=params.get("max_lightning_time_threshold", 1),
        max_lightning_duration=params.get("max_lightning_duration", 20.0),
        max_dist_between_pts=params.get("max_lightning_dist", 50000),
        max_speed=params.get("max_lightning_speed", 299792.458),
        min_speed=params.get("min_lightning_speed", 0),
        min_pts=params.get("min_lightning_points", 300),  
    )

    save_result_cache(df, params, result)

    return result


def export_as_csv(bucketed_strike_indices: list[list[int]], events: pd.DataFrame, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for indices in bucketed_strike_indices:
        # Extract the events corresponding to this lightning strike and sort by time
        strike_df = events.iloc[indices].sort_values(by="time_unix")
        
        # Convert the first event's time to a datetime object for a meaningful file name
        start_time_unix = strike_df.iloc[0]["time_unix"]
        start_time_dt = datetime.datetime.fromtimestamp(
        start_time_unix, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Generate a safe file name using regex to remove forbidden characters
        safe_start_time = re.sub(r'[<>:"/\\|?*]', '_', str(start_time_dt))
        
        
        # Construct the output file path and ensure unique names if needed
        output_filename = os.path.join(output_dir, f"{safe_start_time}.csv")
        counter = 1
        while os.path.exists(output_filename):
            output_filename = os.path.join(output_dir, f"{safe_start_time}_{counter}.csv")
            counter += 1
        
        # Export the lightning strike data to CSV without the DataFrame index
        strike_df.to_csv(output_filename, index=False)
        print(f"Exported lightning strike CSV to {output_filename}")