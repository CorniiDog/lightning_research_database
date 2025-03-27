import pandas as pd
import cupy as cp
from collections import deque
from typing import Tuple
from tqdm import tqdm

def stitch_lightning_strike(strike_indeces: list[int], events: pd.DataFrame, **params) -> list[Tuple[(int, int)]]:
    """
    Builds a chain of lightning strike nodes by connecting each strike to the closest preceding strike,
    but only if the connection satisfies thresholds on time, distance, and speed.
    
    Parameters:
        strike_indeces (list[int]): List of indices corresponding to lightning strike events.
        events (pd.DataFrame): DataFrame containing event data with columns like "time_unix", "x", "y", and "z".
        **params: Additional filtering parameters:
            max_lightning_time_threshold (float): Maximum allowed time difference between consecutive points (seconds). Default is 1.
            max_lightning_dist (float): Maximum allowed distance between consecutive points (meters). Default is 50000.
            max_lightning_speed (float): Maximum allowed speed (m/s). Default is 299792.458.
            min_lightning_speed (float): Minimum allowed speed (m/s). Default is 0.
    
    Returns:
        Correlations (list[Tuple[(int, int)]]): A list of correlations between indices
    """
    # Retrieve filtering parameters.
    max_time_threshold = params.get("max_lightning_time_threshold", 1)
    max_dist_between_pts = params.get("max_lightning_dist", 50000)
    max_speed = params.get("max_lightning_speed", 299792.458)
    min_speed = params.get("min_lightning_speed", 0)


    # Sort the strike indices chronologically (using "time_unix").
    strike_indeces: list[int] = sorted(strike_indeces, key=lambda idx: events.loc[idx, "time_unix"])
    
    # Create a Series DataFrame for only the selected strikes.
    strike_series_df: pd.Series = events.iloc[strike_indeces]

    # Cache the cupy arrays for the data columns.
    all_x = cp.array(strike_series_df["x"].values)
    all_y = cp.array(strike_series_df["y"].values)
    all_z = cp.array(strike_series_df["z"].values)
    all_times = cp.array(strike_series_df["time_unix"].values)

    # List to store nodes corresponding to each strike.
    parsed_indices: list[int] = []
    correlations: list[Tuple[(int, int)]] = []

    for i in range(len(strike_indeces)):
        current_indice = strike_indeces[i]

        if len(parsed_indices) > 0:
            # Get the current strike's coordinates and time.
            x1, y1, z1 = all_x[i], all_y[i], all_z[i]
            current_time = all_times[i]

            x_pre = all_x[:i]
            y_pre = all_y[:i]
            z_pre = all_z[:i]
            times_pre = all_times[:i]
            current_coords = cp.array([x1, y1, z1])

            # Compute squared Euclidean distances.
            # We don't sqrt for optimization purposes.
            # We just do our math in squareds
            distances_squared = ((x_pre - current_coords[0])**2 +
                                (y_pre - current_coords[1])**2 +
                                (z_pre - current_coords[2])**2)

            # Compute time differences (seconds).
            dt = current_time - times_pre
            dt = cp.where(dt == 0, 1e-6, dt)  # Avoid divide-by-zero

            dt_squared = (dt ** 2)

            # Compute squared speeds (m²/s²).
            speeds_squared = distances_squared / dt_squared

            # Precompute squared thresholds.
            max_dist_squared = max_dist_between_pts ** 2
            max_speed_squared = max_speed ** 2
            min_speed_squared = min_speed ** 2
            max_time_threshold_squared = max_time_threshold ** 2

            # Apply filtering mask using squared comparisons.
            # (dt_squared <= max_time_threshold_squared) & \
            mask = (distances_squared <= max_dist_squared) & \
                (speeds_squared <= max_speed_squared) & \
                (speeds_squared >= min_speed_squared)

            # mask = (dt <= max_time_threshold) & (distances <= max_dist_between_pts) & (speeds <= max_speed) & (speeds >= min_speed)

            valid_indices = cp.where(mask)[0]

            if valid_indices.size > 0:
                # Select the candidate with the minimum distance among those valid.
                valid_distances_squared = distances_squared[valid_indices]
                min_valid_idx = int(cp.argmin(valid_distances_squared).get())
                candidate_idx = int(valid_indices[min_valid_idx].get())
                parent_indice = parsed_indices[candidate_idx]

                correlations.append((parent_indice, current_indice))

        # Save the current node.
        parsed_indices.append(current_indice)


    return correlations

def stitch_lightning_strikes(bucketed_strike_indices: list[list[int]], events: pd.DataFrame, **params) -> list[list[Tuple[int, int]]]:
    """
    Processes multiple groups of lightning strike indices sequentially with a progress bar.
    
    Parameters:
      bucketed_strike_indices (list[list[int]]): A list where each element is a list of strike indices representing a group.
      events (pd.DataFrame): DataFrame containing event data.
      **params: Additional parameters passed to stitch_lightning_strike.
    
    Returns:
      A list containing correlations for each group.
    """
    results = []
    for strike_indices in tqdm(bucketed_strike_indices, total=len(bucketed_strike_indices)):
        result = stitch_lightning_strike(strike_indices, events, **params)
        if len(result) > 0:
            results.append(result)

    return results


