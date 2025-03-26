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
import base64
from PIL import Image, ImageSequence
import io
import re
from typing import Tuple


def create_interactive_html(gif_filename: str, html_filename: str, display_width: int = 400):
    """
    Reads a GIF file, extracts its frames, encodes them in Base64, and writes a self-contained
    HTML file with an interactive slider to scroll through the frames.

    Parameters:
      gif_filename (str): Path to the input GIF file.
      html_filename (str): Path to the output HTML file.
      display_width (int): Width (in pixels) for the displayed image. Defaults to 400.

    Returns:
      None. Writes the HTML file to disk.
    """
    # Open the GIF file.
    with Image.open(gif_filename) as img:
        frames = []
        # Iterate over all frames in the GIF.
        for frame in ImageSequence.Iterator(img):
            # Create an in-memory binary stream.
            buffer = io.BytesIO()
            # Save the current frame as a PNG to preserve quality.
            frame.save(buffer, format="PNG")
            # Encode the binary data to Base64.
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            frames.append(f"data:image/png;base64,{base64_str}")

    # Create HTML content with the pre-baked frames.
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Interactive GIF Viewer</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 20px;
      text-align: center;
    }}
    #frame-display {{
      width: {display_width}px;
      height: auto;
      border: 1px solid #ccc;
      margin-bottom: 10px;
    }}
    #slider {{
      width: {display_width}px;
    }}
  </style>
</head>
<body>
  <img id="frame-display" src="{frames[0]}" alt="GIF Frame">
  <br>
  <input type="range" id="slider" min="0" max="{len(frames)-1}" value="0">
  <script>
    // Pre-baked frames array (Base64 encoded images).
    const frames = {frames};
    const slider = document.getElementById("slider");
    const frameDisplay = document.getElementById("frame-display");

    // Function to update the displayed frame.
    function updateFrame(index) {{
      frameDisplay.src = frames[index];
    }}

    // Initialize with the first frame.
    updateFrame(0);

    // Update frame when slider value changes.
    slider.addEventListener("input", function(event) {{
      updateFrame(event.target.value);
    }});
  </script>
</body>
</html>
"""

    # Write the HTML content to the specified file.
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Interactive HTML file saved as {html_filename}")

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
) -> go.Figure:
    """
    Plot the stitched lightning correlations on a 2D scatter plot using latitude and longitude.

    Parameters:
      lightning_correlations (list[Tuple[int, int]]): List of tuples (parent_index, child_index)
      events (pd.DataFrame): DataFrame containing event data with "lat" and "lon" columns.
      output_filename (str): Filename to export the plot image. Defaults to "strike_stitched_map.png".
      _export_fig (bool): If True, export the figure as an image.

    Returns:
      go.Figure: The Plotly figure containing the lightning stitch plot.
    """

    # Get the start time
    start_time_unix = events.iloc[lightning_correlations[-1][-1]]["time_unix"]
    start_time_dt = datetime.datetime.fromtimestamp(start_time_unix, tz=datetime.timezone.utc)
    frac = int(start_time_dt.microsecond / 10000)  # Convert microseconds to hundredths (0-99)
    start_time_dt = start_time_dt.strftime(f"%Y-%m-%d %H:%M:%S.{frac:02d} UTC")

    # Prepare lists for line segments; insert None to separate individual segments.
    lines_x = []
    lines_y = []
    for parent_idx, child_idx in lightning_correlations:
        parent_row = events.loc[parent_idx]
        child_row = events.loc[child_idx]
        # Use latitude for x and longitude for y
        x1, y1 = parent_row["lon"], parent_row["lat"]
        x2, y2 = child_row["lon"], child_row["lat"]
        lines_x.extend([x1, x2, None])
        lines_y.extend([y1, y2, None])
    
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
    
    # Build the figure.
    fig = go.Figure(data=[line_trace, points_trace])
    fig.update_layout(
        title=f"Lightning Strike Stitching ({start_time_dt})",
        xaxis=dict(title="Longitude", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="Latitude", showgrid=True, gridcolor="lightgray"),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Export the figure to a file if requested.
    if _export_fig:
        fig.write_image(output_filename, scale=3)
    
    return fig
