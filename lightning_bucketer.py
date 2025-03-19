import cupy as cp
import numpy as np
import pandas as pd
import pickle as pkl
import os
import hashlib


def _bucket_dataframe_lightnings(
    df: pd.DataFrame,
    max_time_threshold,
    max_dist_between_pts,
    max_speed,
    min_speed=0,
    min_pts=0,
) -> list[list[int]]:
    """
    Buckets the dataframe into groups of lightning strikes based on temporal and spatial constraints.

    This function performs the following steps:
      1. Sorts the dataframe by 'time_unix'.
      2. Converts the time values to a GPU array and computes the difference between consecutive time points.
      3. Uses a vectorized cumulative sum to group events that are close in time (i.e., where the time difference
         is below the max_time_threshold).
      4. Iterates through each time group and further clusters events into potential lightning strikes based on:
         - Minimum number of points in a group (min_pts).
         - Spatial proximity (max_dist_between_pts).
         - Speed constraints (min_speed and max_speed).
      5. Returns a list of lists, where each sublist contains indices from the original dataframe corresponding to
         a detected lightning strike.

    Parameters:
      df (pandas.DataFrame): DataFrame containing lightning event data with headers including
                             ['id', 'time_unix', 'lat', 'lon', 'alt', 'reduced_chi2', 'num_stations',
                              'power_db', 'power', 'mask', 'stations', 'x', 'y', 'z'].
      max_time_threshold (float): Maximum allowed time difference between consecutive points (in seconds) to group them together.
      max_dist_between_pts (float): Maximum allowed spatial distance (in meters) between points to consider them part of the same strike.
      max_speed (float): Maximum allowed speed (in m/s) between points.
      min_speed (float, optional): Minimum allowed speed (in m/s) between points. Defaults to 0.
      min_pts (int, optional): Minimum number of points required for a group to be considered a valid lightning strike. Defaults to 0.

    Returns:
      list[list[int]]: A list where each sublist contains the indices of the dataframe representing a lightning strike.
    """
    df.sort_values(by="time_unix", inplace=True)
    time_unix_gpu = cp.asarray(df["time_unix"].values)
    delta_t = cp.diff(time_unix_gpu)

    # Compute group indices vectorized:
    # cumsum basically makes a running total that increments every time delta_t is surpassed
    # A highly optimized method at grouping by time threshold
    time_groups = cp.concatenate(
        (
            cp.array([0], dtype=cp.int32),
            cp.cumsum((delta_t > max_time_threshold).astype(cp.int32)),
        )
    )
    print(time_groups)
    print("Done")

    group_ids = cp.asnumpy(time_groups)
    unique_groups = np.unique(group_ids)
    total_unique_groups = len(unique_groups)

    lightning_strikes = []

    for i, group in enumerate(unique_groups):
        group_indices = np.where(group_ids == group)[0]

        # Ensure minimum number of points
        if len(group_indices) < min_pts:
            continue

        pct = 100 * (i + 1) / total_unique_groups
        print(
            f"Progress: {pct:.2f}% ({i+1}/{total_unique_groups}). Currently processing {len(group_indices)} points."
        )

        # Retrieve the group events.
        group_df = df.iloc[group_indices]

        x_vals = group_df["x"].values
        y_vals = group_df["y"].values
        z_vals = group_df["z"].values
        unix_vals = group_df["time_unix"].values

        # Example modification for caching subgroup GPU arrays
        sub_groups = []  # List to hold dictionaries for each subgroup

        for j in range(len(x_vals)):
            # Create an event record
            event_x = x_vals[j]
            event_y = y_vals[j]
            event_z = z_vals[j]
            event_unix = unix_vals[j]

            found = False
            if sub_groups:
                for sg in sub_groups:
                    # Compute distances using already cached GPU arrays
                    distances = cp.sqrt(
                        (event_x - sg["x"]) ** 2
                        + (event_y - sg["y"]) ** 2
                        + (event_z - sg["z"]) ** 2
                    )
                    if cp.any(distances <= max_dist_between_pts):
                        # Check speeds only for points within the distance threshold
                        dt = cp.abs(event_unix - sg["unix"])
                        dt = cp.where(dt == 0, 1e-6, dt)
                        speeds = distances / dt
                        if cp.any((speeds >= min_speed) & (speeds <= max_speed)):
                            sg["indices"].append(j)
                            # Update subgroup GPU arrays by concatenating the new event
                            sg["x"] = cp.concatenate([sg["x"], cp.array([event_x])])
                            sg["y"] = cp.concatenate([sg["y"], cp.array([event_y])])
                            sg["z"] = cp.concatenate([sg["z"], cp.array([event_z])])
                            sg["unix"] = cp.concatenate(
                                [sg["unix"], cp.array([event_unix])]
                            )
                            found = True
                            break

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

        for sub_group in sub_groups:
            if len(sub_group["indices"]) < min_pts:
                continue

            final_subgroup = []
            for idx in sub_group["indices"]:
                final_subgroup.append(group_indices[idx])
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
        max_dist_between_pts=params.get("max_lightning_dist", 50000),
        max_speed=params.get("max_lightning_speed", 299792.458),
        min_speed=params.get("min_lightning_speed", 0),
        min_pts=params.get("min_lightning_points", 300),
    )

    save_result_cache(df, params, result)

    return result
