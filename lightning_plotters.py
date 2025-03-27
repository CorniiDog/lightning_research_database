import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
import os
import multiprocessing
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import imageio
from PIL import Image
import re
from typing import Tuple

def plot_strikes_over_time(
    bucketed_strikes_indices_sorted: list[list[int]],
    events: pd.DataFrame,
    output_filename="strike_points_over_time.png",
    _export_fig=True
):
    """
    Generate a scatter plot of lightning strike points over time and save it as an image.

    The function extracts the start time (as a timezone-aware datetime) and the number of strike points
    from each bucket in the sorted list of lightning strikes. It then creates a scatter plot with lines
    connecting the points using Plotly, and finally saves the plot to a specified file.

    Parameters:
      bucketed_strikes_indices_sorted (list of list of int): Sorted list of lightning strike indices,
                                                             where each sublist corresponds to a strike.
      events (pandas.DataFrame): DataFrame containing lightning event data, including a 'time_unix' column.
      output_filename (str): The filename to save the resulting plot. Defaults to "strike_points_over_time.png".

    Returns:
      str: The output filename where the image is saved.
    """
    # Prepare data: For each bucket, extract the start time (as a timezone-aware datetime) and the number of strike points.
    plot_data = []
    for strike in bucketed_strikes_indices_sorted:
        start_time_unix = events.iloc[strike[0]]["time_unix"]
        dt = datetime.datetime.fromtimestamp(start_time_unix, tz=datetime.timezone.utc)
        plot_data.append({"Time": dt, "Strike Points": len(strike)})

    df_plot = pd.DataFrame(plot_data)
    # Sort the DataFrame by time.
    df_plot.sort_values(by="Time", inplace=True)

    # Compute global start time (earliest strike bucket) for display.
    global_start_time = df_plot["Time"].min().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Create a scatter plot with lines connecting the points.
    fig = px.scatter(
        df_plot,
        x="Time",
        y="Strike Points",
        title=f"Number of Strike Points Over Time ({global_start_time})",
        template="plotly_white",
        labels={"Time": "Time (UTC)", "Strike Points": "Number of Strike Points"},
    )
    fig.update_traces(
        mode="lines+markers",
        marker=dict(size=3, color="red"),
        line=dict(color="darkblue", width=2),
    )
    fig.update_layout(
        title_font_size=18,
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    if _export_fig:
        # Save as svg
        fig.write_image(output_filename, scale=3)

    return fig


def plot_avg_power_map(
    strike_indices: list[int],
    events: pd.DataFrame,
    lat_bins: int = 500,
    lon_bins: int = 500,
    sigma: float = 1.0,
    transparency_threshold:float = 0.01,
    output_filename: str = "strike_avg_power_map.png",
    _export_fig=True,
    _range=None,
    _bar_range=None
):
    """
    Generate a heatmap of average power (in dBW) over latitude/longitude for a specified lightning strike.

    This function bins the strike event data into a 2D grid and calculates the mean power in each bin.
    It then applies a Gaussian blur to smooth the binned data and creates a heatmap using Plotly.

    Parameters:
      strike_indices (list of int): List of indices corresponding to rows in the 'events' DataFrame for a specific strike.
      events (pandas.DataFrame): DataFrame containing at least 'lat', 'lon', and 'power_db' columns.
      lat_bins (int): Number of bins for latitude. Defaults to 500.
      lon_bins (int): Number of bins for longitude. Defaults to 500.
      sigma (float): Standard deviation for the Gaussian kernel applied for smoothing. 1.0.
      transparency_threshold (float): A number such that if the power_db is below this threshold, it becomes transparent. Set to -1 to disable
      output_filename (str): Filename for the output image. Defaults to "strike_avg_power_map.png".

    Returns:
      str: The output filename where the heatmap image is saved.
    """

    strike_events = events.iloc[strike_indices]

    # Get the strike's start time from the first event.
    start_time_unix = strike_events.iloc[-1]["time_unix"]
    start_time_dt = datetime.datetime.fromtimestamp(start_time_unix, tz=datetime.timezone.utc)
    frac = int(start_time_dt.microsecond / 10000)  # Convert microseconds to hundredths (0-99)
    start_time_dt = start_time_dt.strftime(f"%Y-%m-%d %H:%M:%S.{frac:02d} UTC")

    # Extract lat, lon, and power for binning.
    lat = strike_events["lat"].values
    lon = strike_events["lon"].values
    power = strike_events["power_db"].values

    # Determine the min/max for lat/lon.
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()

    # Use binned_statistic_2d to compute mean power in each lat/lon bin.
    stat, lat_edges, lon_edges, _ = binned_statistic_2d(
        lat,
        lon,
        power,
        statistic="mean",
        bins=[lat_bins, lon_bins],
        range=_range or [[lat_min, lat_max], [lon_min, lon_max]],
    )

    # Replace NaNs with 0 (or any default) so they appear in the heatmap.
    stat_filled = np.nan_to_num(stat, nan=0.0)

    # Apply Gaussian blur to smooth the binned data.
    # Increase or decrease `sigma` depending on how much smoothing you want.
    blurred_stat = gaussian_filter(stat_filled, sigma=sigma)

    # Remove areas below the transparency threshold by masking them as NaN.
    blurred_stat = np.where(blurred_stat < transparency_threshold, np.nan, blurred_stat)

    # Compute bin centers for plotting.
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    if _bar_range:
        _bar_min = _bar_range[0]
        _bar_max = _bar_range[1]
        _zauto = False
    else:
        _bar_min = None
        _bar_max = None
        _zauto = True


    # Create heatmap trace; x-axis = longitude, y-axis = latitude.
    heatmap = go.Heatmap(
        x=lon_centers,
        y=lat_centers,
        z=blurred_stat,
        colorscale="ice",
        colorbar=dict(title="Average Power (dBW)"),
        zauto=_zauto,
        zmin = _bar_min,
        zmax = _bar_max,
        reversescale = False
    )

    # Build the figure with layout settings.
    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=f"Smoothed (Gaussian) Average Power Heatmap (dBW) ({start_time_dt})",
        xaxis=dict(title="Longitude", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="Latitude", showgrid=True, gridcolor="lightgray"),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    # Export the figure to file (SVG/PNG).
    if _export_fig:
        fig.write_image(output_filename, scale=3)

    return fig, np.nanmax(blurred_stat)

def generate_strike_gif(
    strike_indices: list[int],
    events: pd.DataFrame,
    lat_bins: int = 500,
    lon_bins: int = 500,
    sigma: float = 1.0,
    num_frames: int = 30,
    transparency_threshold: float = 0.01,
    output_filename: str = "strike_power_map_animation.gif",
    duration: float = 3000,
    looped:bool = True,
) -> str:
    """
    Generate a GIF animation of a lightning strike heatmap evolving over a specified number of frames.

    This function sorts the lightning strike events by time, divides the total number of events into 'num_frames'
    segments, and generates a heatmap for each cumulative segment using `plot_avg_power_map` (with PNG export omitted).
    The frames are then compiled into a GIF using imageio.

    Parameters:
      strike_indices (list of int): List of indices for lightning strike events.
      events (pd.DataFrame): DataFrame with at least 'lat', 'lon', 'power_db', and 'time_unix' columns.
      lat_bins (int): Number of bins for latitude. Defaults to 500.
      lon_bins (int): Number of bins for longitude. Defaults to 500.
      sigma (float): Standard deviation for the Gaussian kernel used in smoothing. 1.0.
      num_frames (int): Number of frames in the resulting GIF. Defaults to 40.
      transparency_threshold (float): A number such that if the power_db is below this threshold, it becomes transparent. Set to -1 to disable
      output_filename (str): Filename for the output GIF. Defaults to "lightning_strike_animation.gif".
      frame_duration (float): Duration (in milliseconds) for each frame in the GIF. Defaults to 3000 milliseconds.
      looped (bool): The gif will loop if set to True

    Returns:
      str: The filename where the GIF animation is saved.
    """

    # Preprocess: sort indices by time and extract corresponding times into a NumPy array.
    sorted_indices = sorted(strike_indices, key=lambda idx: events.loc[idx, "time_unix"])
    sorted_times = np.array([events.loc[idx, "time_unix"] for idx in sorted_indices])

    # Determine the overall time span among the selected events.
    min_time = events.loc[sorted_indices[0], "time_unix"]
    max_time = events.loc[sorted_indices[-1], "time_unix"]
    time_interval = (max_time - min_time) / num_frames

    strike_events = events.iloc[strike_indices]

    # Extract lat, lon, and power for binning.
    lat = strike_events["lat"].values
    lon = strike_events["lon"].values

    # Determine the min/max for lat/lon.
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()

    _range = [[lat_min, lat_max], [lon_min, lon_max]]
    # Use binned_statistic_2d to compute mean power in each lat/lon bin.


    _, max_stat = plot_avg_power_map(
            sorted_indices,
            events,
            lat_bins=lat_bins,
            lon_bins=lon_bins,
            sigma=sigma,
            _export_fig=False,
            _range=_range,
            transparency_threshold=transparency_threshold
        )

    frames = []
    # Generate frames based on time intervals.
    for frame in range(1, num_frames + 1):
        current_time_threshold = min_time + frame * time_interval
        # Filter events up to the current time threshold.
        
        # Quickly find the cutoff position using np.searchsorted.
        pos = np.searchsorted(sorted_times, current_time_threshold, side='right')
        frame_indices = sorted_indices[:pos]   
        fig, _ = plot_avg_power_map(
            frame_indices,
            events,
            lat_bins=lat_bins,
            lon_bins=lon_bins,
            sigma=sigma,
            transparency_threshold=transparency_threshold,
            _export_fig=False,
            _range=_range,
            _bar_range=[0, max_stat],
        )
        
        # Convert the Plotly figure to an image.
        img_bytes = fig.to_image(format="png", scale=3)
        img = Image.open(BytesIO(img_bytes))
        frames.append(np.array(img))

    # Logic to set to 0 (means indefinitely)
    # Else loop once
    looped = 0 if looped else 1

    # Split the gif's duration to the number of frames
    frame_duration = duration/num_frames

    # Save all frames as a GIF.
    imageio.mimsave(output_filename, frames, duration=frame_duration, loop=looped)
    return output_filename

def _plot_strike(args):
    """
    Helper function to generate and save an average power heatmap for a single lightning strike.

    This function is designed for parallel processing. It unpacks the input arguments, computes the strike's start time,
    constructs an output filename, and calls plot_avg_power_map to generate and save the heatmap.

    Parameters:
      args (tuple): A tuple containing:
          - strike_indices (list of int): List of indices representing a lightning strike.
          - events (pandas.DataFrame): DataFrame containing the lightning event data.
          - strike_dir (str): Directory where the heatmap image should be saved.

    Returns:
      None
    """
    strike_indices, events, strike_dir, as_gif, sigma, transparency_threshold = args

    # Get the start time
    start_time_unix = events.iloc[strike_indices[0]]["time_unix"]
    start_time_dt = datetime.datetime.fromtimestamp(
        start_time_unix, tz=datetime.timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    safe_start_time = re.sub(r'[<>:"/\\|?*]', '_', str(start_time_dt))

    if not as_gif:
        output_filename = os.path.join(strike_dir, f"{safe_start_time}.png")
        plot_avg_power_map(strike_indices, events, output_filename=output_filename, sigma=sigma, transparency_threshold=transparency_threshold)
    else:
        output_filename = os.path.join(strike_dir, f"{safe_start_time}.gif")
        generate_strike_gif(strike_indices, events, output_filename=output_filename, sigma=sigma, transparency_threshold=transparency_threshold)


def plot_all_strikes(
    bucketed_strike_indices, events, strike_dir="strikes", num_cores=1, as_gif=False, sigma=1.0, transparency_threshold=0.01
):
    """
    Generate and save heatmaps for all detected lightning strikes using parallel processing.

    This function prepares argument tuples for each lightning strike and utilizes a multiprocessing pool
    to generate average power heatmaps concurrently. A progress bar is displayed to indicate processing status.

    Parameters:
      bucketed_strike_indices (list of list of int): List where each sublist contains indices corresponding to a lightning strike.
      events (pandas.DataFrame): DataFrame containing the lightning event data.
      strike_dir (str): Directory to save the generated heatmap images. Defaults to "strikes".
      num_cores (int): Number of worker processes to use for parallel processing. Defaults to 1.
      as_gif (bool): Set to true to export as a gif instead
      sigma (float): Standard deviation for the Gaussian kernel applied for smoothing. 1.0.
      transparency_threshold (float): A number such that if the power_db is below this threshold, it becomes transparent. Set to -1 to disable
            
    Returns:
      None
    """
    # Prepare the argument tuples for each strike
    args_list = [
        (strike_indices, events, strike_dir, as_gif, sigma, transparency_threshold)
        for strike_indices in bucketed_strike_indices
    ]

    # Use a pool of worker processes to parallelize
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Use imap so that we can attach tqdm for a progress bar
        for _ in tqdm(pool.imap(_plot_strike, args_list), total=len(args_list)):
            pass


def plot_lightning_stitch(
    lightning_correlations: list[Tuple[int, int]], 
    events: pd.DataFrame,
    output_filename: str = "strike_stitched_map.png",
    _export_fig: bool = True,
    _range=None
) -> go.Figure:
    """
    Plot the stitched lightning correlations on a 2D scatter plot using latitude and longitude.
    Optionally, add invisible points to enforce axis ranges.

    Parameters:
      lightning_correlations (list[Tuple[int, int]]): List of tuples (parent_index, child_index)
      events (pd.DataFrame): DataFrame containing event data with "lat", "lon", and "time_unix" columns.
      output_filename (str): Filename to export the plot image. Defaults to "strike_stitched_map.png".
      _export_fig (bool): If True, export the figure as an image.
      _range (list or None): Optional axis ranges in the format [[lat_min, lat_max], [lon_min, lon_max]].
                             If None, the range will be computed from the data.

    Returns:
      go.Figure: The Plotly figure containing the lightning stitch plot.
    """
    # Get the start time from the last correlation's child event.
    start_time_unix = events.loc[lightning_correlations[-1][1]]["time_unix"]
    start_time_dt = datetime.datetime.fromtimestamp(start_time_unix, tz=datetime.timezone.utc)
    frac = int(start_time_dt.microsecond / 10000)  # Convert microseconds to hundredths (0-99)
    start_time_str = start_time_dt.strftime(f"%Y-%m-%d %H:%M:%S.{frac:02d} UTC")

    # Use a variable name 'plot_range' to avoid clashing with built-in 'range'
    plot_range = _range
    # Prepare lists for line segments; also compute range if not provided.
    lines_x = []
    lines_y = []
    computed_lat_min, computed_lat_max, computed_lon_min, computed_lon_max = None, None, None, None

    for parent_idx, child_idx in lightning_correlations:
        parent_row = events.loc[parent_idx]
        child_row = events.loc[child_idx]
        # Use longitude for x and latitude for y (conventional mapping)
        x1, y1 = parent_row["lon"], parent_row["lat"]
        x2, y2 = child_row["lon"], child_row["lat"]
        lines_x.extend([x1, x2, None])
        lines_y.extend([y1, y2, None])
        if plot_range is None:
            for x_val in [x1, x2]:
                if computed_lon_min is None or x_val < computed_lon_min:
                    computed_lon_min = x_val
                if computed_lon_max is None or x_val > computed_lon_max:
                    computed_lon_max = x_val
            for y_val in [y1, y2]:
                if computed_lat_min is None or y_val < computed_lat_min:
                    computed_lat_min = y_val
                if computed_lat_max is None or y_val > computed_lat_max:
                    computed_lat_max = y_val

    if plot_range is None:
        # Note: plot_range is defined as [[lat_min, lat_max], [lon_min, lon_max]]
        plot_range = [[computed_lat_min, computed_lat_max], [computed_lon_min, computed_lon_max]]
                
    # Create a scatter trace for the correlation lines.
    line_trace = go.Scatter(
        x=lines_x,
        y=lines_y,
        mode="lines",
        line=dict(color="blue", width=2),
        name="Lightning Stitch"
    )

    # Gather unique indices for lightning strikes.
    unique_indices = set()
    for parent_idx, child_idx in lightning_correlations:
        unique_indices.add(parent_idx)
        unique_indices.add(child_idx)
    
    # Extract coordinates for the strike points.
    points_x = []
    points_y = []
    for idx in unique_indices:
        row = events.loc[idx]
        points_x.append(row["lon"])
        points_y.append(row["lat"])
    
    # Create a scatter trace for the strike points.
    points_trace = go.Scatter(
        x=points_x,
        y=points_y,
        mode="markers",
        marker=dict(color="red", size=3),
        name="Lightning Strikes"
    )
    
    # Add invisible points to enforce the specified range.
    # Since x-axis corresponds to longitude and y-axis to latitude,
    # we add two points: one at the lower bound and one at the upper bound.
    invisible_trace = go.Scatter(
        x=[plot_range[1][0], plot_range[1][1]],
        y=[plot_range[0][0], plot_range[0][1]],
        mode="markers",
        marker=dict(opacity=0),
        showlegend=False,
        hoverinfo="none"
    )
    
    # Build the figure.
    fig = go.Figure(data=[line_trace, points_trace, invisible_trace])
    fig.update_layout(
        title=f"Lightning Strike Stitching ({start_time_str})",
        xaxis=dict(title="Longitude", range=plot_range[1], showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="Latitude", range=plot_range[0], showgrid=True, gridcolor="lightgray"),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Export the figure to a file if requested.
    if _export_fig:
        fig.write_image(output_filename, scale=3)
    
    return fig, plot_range



def plot_lightning_stitch_gif(
    lightning_correlations: list[Tuple[int, int]], 
    events: pd.DataFrame,
    num_frames: int = 30,
    output_filename: str = "strike_stitched_map_animation.gif",
    duration: float = 3000,
    looped: bool = True,
) -> str:
    """
    Generate a GIF animation of the lightning stitching process.
    
    The function progressively adds lightning correlations and creates an animated plot
    showing the incremental construction of the stitched lightning map.
    
    Parameters:
      lightning_correlations (list[Tuple[int, int]]): List of tuples (parent_index, child_index).
      events (pd.DataFrame): DataFrame containing event data with "lat", "lon", and "time_unix" columns.
      num_frames (int): Number of frames in the GIF animation. Defaults to 30.
      output_filename (str): Filename for the output GIF. Defaults to "strike_stitched_map_animation.gif".
      duration (float): Total duration (in milliseconds) of the GIF. Defaults to 3000.
      looped (bool): If True, the GIF will loop indefinitely. Defaults to True.
    
    Returns:
      str: The filename where the GIF animation is saved.
    """

    # Sort the correlations by the child's event time to ensure proper progression.
    sorted_correlations = sorted(lightning_correlations, key=lambda corr: events.loc[corr[1], "time_unix"])
    
    # Compute the full plot range from all correlations if not provided.
    computed_lat_min, computed_lat_max, computed_lon_min, computed_lon_max = None, None, None, None
    for parent_idx, child_idx in lightning_correlations:
        parent_row = events.loc[parent_idx]
        child_row = events.loc[child_idx]
        for lat_val in [parent_row["lat"], child_row["lat"]]:
            if computed_lat_min is None or lat_val < computed_lat_min:
                computed_lat_min = lat_val
            if computed_lat_max is None or lat_val > computed_lat_max:
                computed_lat_max = lat_val
        for lon_val in [parent_row["lon"], child_row["lon"]]:
            if computed_lon_min is None or lon_val < computed_lon_min:
                computed_lon_min = lon_val
            if computed_lon_max is None or lon_val > computed_lon_max:
                computed_lon_max = lon_val
    full_range = [[computed_lat_min, computed_lat_max], [computed_lon_min, computed_lon_max]]
    
    frames = []
    total_corr = len(sorted_correlations)
    
    # Generate frames by progressively adding more correlations.
    for frame in range(1, num_frames + 1):
        # Determine the cutoff index for the current frame (ensure at least one correlation is shown).
        cutoff = max(1, int(round((frame / num_frames) * total_corr)))
        subset = sorted_correlations[:cutoff]
        
        # Generate the plot for the current subset.
        # Note: _export_fig is False so the figure is not saved to disk.
        fig, _ = plot_lightning_stitch(
            subset, 
            events, 
            output_filename="temp.png",  # Dummy filename; image export is disabled.
            _export_fig=False,
            _range=full_range
        )
        
        # Convert the Plotly figure to an image.
        img_bytes = fig.to_image(format="png", scale=3)
        img = Image.open(BytesIO(img_bytes))
        frames.append(np.array(img))
    
    # Calculate frame duration (in milliseconds) and set loop parameter (0 for infinite looping).
    frame_duration = duration / num_frames
    loop_val = 0 if looped else 1
    
    # Save all frames as a GIF animation.
    imageio.mimsave(output_filename, frames, duration=frame_duration, loop=loop_val)
    return output_filename

            
def _plot_strike_stitchings(args):
    """
    Helper function to generate and save a stitched lightning map for a single group of correlations.

    This function unpacks the input arguments, determines a safe filename based on the last
    correlation's child event time, and then generates either a static image or a GIF animation
    using the corresponding plotting function.

    Parameters:
      args (tuple): A tuple containing:
          - lightning_correlations (list[Tuple[int, int]]): List of correlation tuples for a group.
          - events (pandas.DataFrame): DataFrame containing lightning event data.
          - output_dir (str): Directory where the stitched image/GIF should be saved.
          - as_gif (bool): If True, generate a GIF animation; otherwise, create a static image.

    Returns:
      None.
    """
    lightning_correlations, events, output_dir, as_gif = args

    # Use the last correlation's child event time for filename generation.
    start_time_unix = events.loc[lightning_correlations[-1][1]]["time_unix"]
    start_time_dt = datetime.datetime.fromtimestamp(start_time_unix, tz=datetime.timezone.utc)
    start_time_str = start_time_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    safe_start_time = re.sub(r'[<>:"/\\|?*]', '_', start_time_str)

    if not as_gif:
        output_filename = os.path.join(output_dir, f"{safe_start_time}.png")
        # Generate a static stitched map.
        plot_lightning_stitch(
            lightning_correlations,
            events,
            output_filename=output_filename,
            _export_fig=True
        )
    else:
        output_filename = os.path.join(output_dir, f"{safe_start_time}.gif")
        # Generate a GIF animation of the stitching process.
        plot_lightning_stitch_gif(
            lightning_correlations,
            events,
            output_filename=output_filename
        )


def plot_all_strike_stitchings(
    bucketed_lightning_correlations: list[list[Tuple[int, int]]],
    events: pd.DataFrame,
    output_dir: str = "strike_stitchings",
    num_cores: int = 1,
    as_gif: bool = False
):
    """
    Generate and save stitched lightning maps for all groups of correlations using parallel processing.

    This function prepares argument tuples for each group of lightning correlations and utilizes a
    multiprocessing pool to generate stitched maps concurrently. A progress bar is displayed to indicate
    processing status.

    Parameters:
      bucketed_lightning_correlations (list[list[Tuple[int, int]]]): List where each sublist contains
                                                                      correlation tuples for a group.
      events (pandas.DataFrame): DataFrame containing the lightning event data.
      output_dir (str): Directory to save the generated stitched images/GIFs. Defaults to "strike_stitchings".
      num_cores (int): Number of worker processes to use for parallel processing. Defaults to 1.
      as_gif (bool): If True, export as a GIF animation; otherwise, export as static images.

    Returns:
      None.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args_list = [
        (lightning_correlations, events, output_dir, as_gif)
        for lightning_correlations in bucketed_lightning_correlations
    ]

    with multiprocessing.Pool(processes=num_cores) as pool:
        for _ in tqdm(pool.imap(_plot_strike_stitchings, args_list), total=len(args_list)):
            pass
