import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import binned_statistic_2d
from typing import List
from scipy.ndimage import gaussian_filter
import os
import multiprocessing
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import imageio
import math

def plot_strikes_over_time(
    bucketed_strikes_indeces_sorted: list[list[int]],
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
      bucketed_strikes_indeces_sorted (list of list of int): Sorted list of lightning strike indices,
                                                             where each sublist corresponds to a strike.
      events (pandas.DataFrame): DataFrame containing lightning event data, including a 'time_unix' column.
      output_filename (str): The filename to save the resulting plot. Defaults to "strike_points_over_time.png".

    Returns:
      str: The output filename where the image is saved.
    """
    # Prepare data: For each bucket, extract the start time (as a timezone-aware datetime) and the number of strike points.
    plot_data = []
    for strike in bucketed_strikes_indeces_sorted:
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
    strike_indeces: list[int],
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
      strike_indeces (list of int): List of indices corresponding to rows in the 'events' DataFrame for a specific strike.
      events (pandas.DataFrame): DataFrame containing at least 'lat', 'lon', and 'power_db' columns.
      lat_bins (int): Number of bins for latitude. Defaults to 500.
      lon_bins (int): Number of bins for longitude. Defaults to 500.
      sigma (float): Standard deviation for the Gaussian kernel applied for smoothing. 1.0.
      transparency_threshold (float): A number such that if the power_db is below this threshold, it becomes transparent. Set to -1 to disable
      output_filename (str): Filename for the output image. Defaults to "strike_avg_power_map.png".

    Returns:
      str: The output filename where the heatmap image is saved.
    """

    strike_events = events.iloc[strike_indeces]

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
          - strike_indeces (list of int): List of indices representing a lightning strike.
          - events (pandas.DataFrame): DataFrame containing the lightning event data.
          - strike_dir (str): Directory where the heatmap image should be saved.

    Returns:
      None
    """
    strike_indeces, events, strike_dir, as_gif, sigma, transparency_threshold = args

    # Get the start time
    start_time_unix = events.iloc[strike_indeces[0]]["time_unix"]
    start_time_dt = datetime.datetime.fromtimestamp(
        start_time_unix, tz=datetime.timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    if not as_gif:
        output_filename = os.path.join(strike_dir, start_time_dt) + ".png"
        plot_avg_power_map(strike_indeces, events, output_filename=output_filename, sigma=sigma, transparency_threshold=transparency_threshold)
    else:
        output_filename = os.path.join(strike_dir, start_time_dt) + ".gif"
        generate_strike_gif(strike_indeces, events, output_filename=output_filename, sigma=sigma, transparency_threshold=transparency_threshold)


def plot_all_strikes(
    bucketed_strike_indeces, events, strike_dir="strikes", num_cores=1, as_gif=False, sigma=1.0, transparency_threshold=0.01
):
    """
    Generate and save heatmaps for all detected lightning strikes using parallel processing.

    This function prepares argument tuples for each lightning strike and utilizes a multiprocessing pool
    to generate average power heatmaps concurrently. A progress bar is displayed to indicate processing status.

    Parameters:
      bucketed_strike_indeces (list of list of int): List where each sublist contains indices corresponding to a lightning strike.
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
        (strike_indeces, events, strike_dir, as_gif, sigma, transparency_threshold)
        for strike_indeces in bucketed_strike_indeces
    ]

    # Use a pool of worker processes to parallelize
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Use imap so that we can attach tqdm for a progress bar
        for _ in tqdm(pool.imap(_plot_strike, args_list), total=len(args_list)):
            pass
