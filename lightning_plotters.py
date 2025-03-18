# sudo apt-get install libmagickwand-dev

import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import griddata
import numpy as np
from scipy.stats import binned_statistic_2d
from typing import List
from scipy.ndimage import gaussian_filter
import os
import multiprocessing
from tqdm import tqdm

def plot_strikes_over_time(
    bucketed_strikes_indeces_sorted: list[list[int]],
    events: pd.DataFrame,
    output_filename="strike_points_over_time.png",
) -> str:
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

    # Save as svg
    fig.write_image(output_filename, scale=3)

    return output_filename

def plot_avg_power_map(
    strike_indeces: list[int],
    events: pd.DataFrame,
    lat_bins: int = 500,
    lon_bins: int = 500,
    sigma: float = 2.0,
    output_filename: str = "strike_avg_power_map.png",
) -> str:
    """
    Generates a heatmap of average power (dBW) over latitude/longitude for the
    specified strike event indices. Applies a Gaussian blur to the binned data.

    :param strike_indeces: A list of integer indices referencing rows in 'events'.
    :param events: A Pandas DataFrame containing at least 'lat', 'lon', and 'power_db' columns.
    :param lat_bins: Number of bins for latitude.
    :param lon_bins: Number of bins for longitude.
    :param sigma: Standard deviation for the Gaussian kernel used to blur the data.
    :param output_filename: Filename for the output image (PNG or SVG).
    :return: The output filename.
    """

    strike_events = events.iloc[strike_indeces]

    # Get the strike's start time from the first event.
    start_time_unix = strike_events.iloc[0]["time_unix"]
    start_time_dt = datetime.datetime.fromtimestamp(
        start_time_unix, tz=datetime.timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

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
        range=[[lat_min, lat_max], [lon_min, lon_max]],
    )

    # Replace NaNs with 0 (or any default) so they appear in the heatmap.
    stat_filled = np.nan_to_num(stat, nan=0.0)

    # Apply Gaussian blur to smooth the binned data.
    # Increase or decrease `sigma` depending on how much smoothing you want.
    blurred_stat = gaussian_filter(stat_filled, sigma=sigma)

    # Compute bin centers for plotting.
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    # Create heatmap trace; x-axis = longitude, y-axis = latitude.
    heatmap = go.Heatmap(
        x=lon_centers,
        y=lat_centers,
        z=blurred_stat,
        colorscale="Viridis",
        colorbar=dict(title="Average Power (dBW)"),
        zauto=True,
    )

    # Build the figure with layout settings.
    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=f"Smoothed (Gaussian) Average Power Heatmap (dBW)\n ({start_time_dt})",
        xaxis=dict(title="Longitude", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="Latitude", showgrid=True, gridcolor="lightgray"),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    # Export the figure to file (SVG/PNG).
    fig.write_image(output_filename, scale=3)

    return output_filename


def _plot_strike(args):
    strike_indeces, events, strike_dir = args

    # Get the start time
    start_time_unix = events.iloc[strike_indeces[0]]["time_unix"]
    start_time_dt = datetime.datetime.fromtimestamp(
        start_time_unix, tz=datetime.timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    output_filename = os.path.join(strike_dir, start_time_dt) + ".png"
    plot_avg_power_map(strike_indeces, events, output_filename=output_filename)

def plot_all_strikes(bucketed_strike_indeces, events, strike_dir="strikes", num_cores=1):
    # Prepare the argument tuples for each strike
    args_list = [
        (strike_indeces, events, strike_dir) 
        for strike_indeces in bucketed_strike_indeces
    ]

    # Use a pool of worker processes to parallelize
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Use imap so that we can attach tqdm for a progress bar
        for _ in tqdm(pool.imap(_plot_strike, args_list), total=len(args_list)):
            pass
