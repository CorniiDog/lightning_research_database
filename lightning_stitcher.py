import pandas as pd
#import cupy as cp
import numpy as np
from collections import deque
from typing import Tuple
from tqdm import tqdm

def filter_correlations_by_chain_size(correlations, min_pts):
    # Build an undirected graph from the correlations.
    graph = {}
    for parent, child in correlations:
        graph.setdefault(parent, set()).add(child)
        graph.setdefault(child, set()).add(parent)
    
    visited = set()
    valid_nodes = set()
    
    # Use DFS to find connected components.
    for node in graph:
        if node not in visited:
            stack = [node]
            component = set()
            while stack:
                current = stack.pop()
                if current not in component:
                    component.add(current)
                    stack.extend(graph.get(current, []))
            visited |= component
            if len(component) >= min_pts:
                valid_nodes |= component
    
    # Filter correlations: both parent and child must be in a valid chain.
    return [(p, c) for (p, c) in correlations if p in valid_nodes and c in valid_nodes]



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
    min_pts = params.get("min_lightning_points", 300)


    # Sort the strike indices chronologically (using "time_unix").
    strike_indeces: list[int] = sorted(strike_indeces, key=lambda idx: events.loc[idx, "time_unix"])
    
    # Create a Series DataFrame for only the selected strikes.
    strike_series_df: pd.Series = events.iloc[strike_indeces]

    # Cache the cupy arrays for the data columns.
    all_x = np.array(strike_series_df["x"].values)
    all_y = np.array(strike_series_df["y"].values)
    all_z = np.array(strike_series_df["z"].values)
    all_times = np.array(strike_series_df["time_unix"].values)

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
            current_coords = np.array([x1, y1, z1])

            # Compute squared Euclidean distances.
            # We don't sqrt for optimization purposes.
            # We just do our math in squareds
            distances_squared = ((x_pre - current_coords[0])**2 +
                                (y_pre - current_coords[1])**2 +
                                (z_pre - current_coords[2])**2)

            # Compute time differences (seconds).
            dt = current_time - times_pre
            dt = np.where(dt == 0, 1e-6, dt)  # Avoid divide-by-zero

            dt_squared = (dt ** 2)

            # Compute squared speeds (m²/s²).
            speeds_squared = distances_squared / dt_squared

            # Precompute squared thresholds.
            max_dist_squared = max_dist_between_pts ** 2
            max_speed_squared = max_speed ** 2
            min_speed_squared = min_speed ** 2
            max_time_threshold_squared = max_time_threshold ** 2

            # Apply filtering mask using squared comparisons.
            mask = (distances_squared <= max_dist_squared)
            mask &= (speeds_squared <= max_speed_squared) 
            mask &= (speeds_squared >= min_speed_squared)
            mask &= (dt_squared <= max_time_threshold_squared)

            valid_indices = np.where(mask)[0]

            if valid_indices.size > 0:
                # Select the candidate with the minimum distance among those valid.
                valid_distances_squared = distances_squared[valid_indices]
                min_valid_idx = int(np.argmin(valid_distances_squared))
                candidate_idx = int(valid_indices[min_valid_idx])
                parent_indice = parsed_indices[candidate_idx]

                correlations.append((parent_indice, current_indice))

        # Save the current node.
        parsed_indices.append(current_indice)

    # Filter out correlations that are not connected to a lightning strike that contains min_pts pts
    correlations_filtered = filter_correlations_by_chain_size(correlations, min_pts)

    return correlations_filtered



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

    bucketed_correlations = []
    for strike_indices in tqdm(bucketed_strike_indices, desc="Stitching Lightning Strikes",total=len(bucketed_strike_indices)):
        correlations = stitch_lightning_strike(strike_indices, events, **params)
        bucketed_correlations.append(correlations)

    combine_strikes_with_intercepting_times = params.get("combine_strikes_with_intercepting_times", True)
    intercepting_times_extension_buffer = params.get("intercepting_times_extension_buffer", 10)
    intercepting_times_extension_max_distance = params.get("intercepting_times_extension_max_distance", 15000)
    intercepting_times_extension_max_distance_squared = intercepting_times_extension_max_distance ** 2 # For faster distance equation

    if combine_strikes_with_intercepting_times:
        temp_bucketed_correlations = []
        for correlations in tqdm(bucketed_correlations, desc="Grouping Intercepting Lightning Strikes",total=len(bucketed_correlations)):
            

            if len(correlations) == 0:
                continue

            sorted_correlations = sorted(correlations, key=lambda corr: events.loc[corr[0], "time_unix"])

            correlations_start_row = events.iloc[sorted_correlations[0][0]]
            correlations_start_time = correlations_start_row["time_unix"]

            x1, y1, z1 = correlations_start_row["x"], correlations_start_row["y"], correlations_start_row["z"]
            
            result_found = False
            for i, temp_correlations in enumerate(temp_bucketed_correlations):

                start_time = events.iloc[temp_correlations[0][0]]["time_unix"]
                end_time = events.iloc[temp_correlations[-1][1]]["time_unix"] + intercepting_times_extension_buffer

                if start_time < correlations_start_time < end_time:
                    
                    unique_idx_list = set()
                    for corr_start_idx, corr_end_idx in temp_correlations:
                        unique_idx_list.add(corr_start_idx)
                        unique_idx_list.add(corr_end_idx)

                    other_strike_data = events.iloc[unique_idx_list]

                    x_vals = other_strike_data["x"].values
                    y_vals = other_strike_data["y"].values
                    z_vals = other_strike_data["z"].values

                    distances_squared = (x_vals - x1) ** 2 + (y_vals - y1) ** 2 (z_vals - z1) ** 2

                    mask = (distances_squared <= intercepting_times_extension_max_distance_squared)
                    if np.any(mask):                        

                        result_found = True
                        temp_correlations += sorted_correlations
                        temp_bucketed_correlations[i] = sorted(temp_correlations, key=lambda corr: events.loc[corr[0], "time_unix"])
                        break
            
            if not result_found:
                temp_bucketed_correlations.append(sorted_correlations)
        bucketed_correlations = temp_bucketed_correlations

    return bucketed_correlations


