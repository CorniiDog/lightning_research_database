# Install your designated CUDA:
# https://developer.nvidia.com/cuda-downloads

# Specified cupy version: https://pypi.org/project/cupy/


import cupy as cp
import numpy as np
import pandas as pd

def bucket_dataframe_lightnings(df:pd.DataFrame, max_time_threshold, max_dist_between_pts, max_speed, min_speed=0, min_pts=0) -> list[list[int]]:
    # Returns a list of all indexes that represent a lightning strike
    # A dataframe has the following headers: ['id', 'time_unix', 'lat', 'lon', 'alt', 'reduced_chi2', 'num_stations', 'power_db', 'power', 'mask', 'stations', 'x', 'y', 'z']
    # It contains roughly 2 million rows, so use cupy when necessary
    df.sort_values(by="time_unix", inplace=True)
    time_unix_gpu = cp.asarray(df['time_unix'].values)
    delta_t = cp.diff(time_unix_gpu)

    # Compute group indices vectorized: 
    # cumsum basically makes a running total that increments every time delta_t is surpassed
    # A highly optimized method at grouping by time threshold
    time_groups = cp.concatenate(
        (cp.array([0], dtype=cp.int32), cp.cumsum((delta_t > max_time_threshold).astype(cp.int32)))
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
        
        pct = 100*(i+1)/total_unique_groups
        print(f"Progress: {pct:.2f}% ({i+1}/{total_unique_groups}). Currently processing {len(group_indices)} points.")
      
        # Retrieve the group events.
        group_df = df.iloc[group_indices]

        x_vals = group_df['x'].values
        y_vals = group_df['y'].values
        z_vals = group_df['z'].values
        unix_vals = group_df['time_unix'].values

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
                  distances = cp.sqrt((event_x - sg['x'])**2 +
                                      (event_y - sg['y'])**2 +
                                      (event_z - sg['z'])**2)
                  if cp.any(distances <= max_dist_between_pts):
                      # Check speeds only for points within the distance threshold
                      dt = cp.abs(event_unix - sg['unix'])
                      dt = cp.where(dt == 0, 1e-6, dt)
                      speeds = distances / dt
                      if cp.any((speeds >= min_speed) & (speeds <= max_speed)):
                          sg['indices'].append(j)
                          # Update subgroup GPU arrays by concatenating the new event
                          sg['x'] = cp.concatenate([sg['x'], cp.array([event_x])])
                          sg['y'] = cp.concatenate([sg['y'], cp.array([event_y])])
                          sg['z'] = cp.concatenate([sg['z'], cp.array([event_z])])
                          sg['unix'] = cp.concatenate([sg['unix'], cp.array([event_unix])])
                          found = True
                          break

            if not found:
                sub_groups.append({
                    'indices': [j],
                    'x': cp.array([event_x]),
                    'y': cp.array([event_y]),
                    'z': cp.array([event_z]),
                    'unix': cp.array([event_unix])
                })

        
        for sub_group in sub_groups:
          if len(sub_group['indices']) < min_pts:
              continue
          
          final_subgroup = []
          for idx in sub_group['indices']:
              final_subgroup.append(group_indices[idx])
          lightning_strikes.append(final_subgroup)

   
    print("Passed groups", len(lightning_strikes))

    return lightning_strikes

        
        


