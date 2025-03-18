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



def plot_strikes_over_time(bucketed_strikes_indeces_sorted: list[list[int]], events: pd.DataFrame, output_filename="strike_points_over_time.png") -> str:
    # Prepare data: For each bucket, extract the start time (as a timezone-aware datetime) and the number of strike points.
    plot_data = []
    for strike in bucketed_strikes_indeces_sorted:
        start_time_unix = events.iloc[strike[0]]['time_unix']
        dt = datetime.datetime.fromtimestamp(start_time_unix, tz=datetime.timezone.utc)
        plot_data.append({"Time": dt, "StrikePoints": len(strike)})
    
    df_plot = pd.DataFrame(plot_data)
    # Sort the DataFrame by time.
    df_plot.sort_values(by="Time", inplace=True)
    
    # Compute global start time (earliest strike bucket) for display.
    global_start_time = df_plot['Time'].min().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Create a scatter plot with lines connecting the points.
    fig = px.scatter(
        df_plot,
        x="Time",
        y="StrikePoints",
        title=f"Number of Strike Points Over Time ({global_start_time})",
        template="plotly_white",
        labels={"Time": "Time (UTC)", "Strike Points": "Number of Strike Points"}
    )
    fig.update_traces(
        mode="lines+markers",
        marker=dict(size=8, color="blue"),
        line=dict(color="darkblue", width=2)
    )
    fig.update_layout(
        title_font_size=18,
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Save as svg
    fig.write_image(output_filename, scale=3)

    return output_filename

def plot_strike_instance(strike_indeces: list[int], events: pd.DataFrame, output_filename="strike_graph.png"):
    # Extract the relevant events for this strike instance.
    strike_events = events.iloc[strike_indeces].copy()
    
    # Create a new column 'marker_size': use power_db when > 1, otherwise set a minimum size.
    strike_events['marker_size'] = strike_events['power_db'].apply(lambda x: x/2 if x/2 > 1 else 1)
    
    # Get the strike's start time from the first event.
    start_time_unix = strike_events.iloc[0]['time_unix']
    start_time_dt = datetime.datetime.fromtimestamp(start_time_unix, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Create the base scatter plot using longitude and latitude.
    fig = px.scatter(
        strike_events,
        x="lon",
        y="lat",
        size="marker_size",
        color="power_db",
        title=f"Lightning Strike Instance ({start_time_dt})",
        hover_data=["id", "time_unix", "power_db"],
        template="plotly_white",
        labels={"lon": "Longitude", "lat": "Latitude", "power_db": "Power (dBW)"}
    )
    # Remove any default marker outlines.
    fig.update_traces(marker=dict(line=dict(width=0)))
    
    # Simulate a glow effect by overlaying an additional scatter trace with larger markers.
    glow_factor = 2.5  # Factor by which to increase the marker size for the glow.
    glow_sizes = strike_events['marker_size'] * glow_factor
    
    glow_trace = go.Scatter(
        x=strike_events['lon'],
        y=strike_events['lat'],
        mode='markers',
        marker=dict(
            size=glow_sizes,
            color=strike_events['power_db'],  # Uses the same color mapping.
            colorscale='Viridis',
            opacity=0.3,  # Lower opacity for a soft glow.
            symbol='circle'
        ),
        hoverinfo='skip',  # Do not show hover info for the glow.
        showlegend=False
    )
    fig.add_trace(glow_trace)
    
    fig.update_layout(
        title_font_size=18,
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Save as svg
    fig.write_image(output_filename, scale=3)

    return output_filename



def plot_avg_power_map(strike_indeces: list[int],
                       events: pd.DataFrame,
                       lat_bins: int = 300,
                       lon_bins: int = 300,
                       sigma: float = 1.0,  # The "spread" for the Gaussian blur
                       output_filename: str = "avg_power_map.png") -> str:
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
    start_time_unix = strike_events.iloc[0]['time_unix']
    start_time_dt = datetime.datetime.fromtimestamp(start_time_unix, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Extract lat, lon, and power for binning.
    lat = strike_events['lat'].values
    lon = strike_events['lon'].values
    power = strike_events['power_db'].values
    
    # Determine the min/max for lat/lon.
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()
    
    # Use binned_statistic_2d to compute mean power in each lat/lon bin.
    stat, lat_edges, lon_edges, _ = binned_statistic_2d(
        lat, lon, power,
        statistic='mean',
        bins=[lat_bins, lon_bins],
        range=[[lat_min, lat_max], [lon_min, lon_max]]
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
        colorscale='Viridis',
        colorbar=dict(title='Average Power (dBW)'),
        zauto=True
    )
    
    # Build the figure with layout settings.
    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=f"Smoothed (Gaussian) Average Power Heatmap (dBW)\n ({start_time_dt})",
        xaxis=dict(title='Longitude', showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title='Latitude', showgrid=True, gridcolor="lightgray"),
        template='plotly_white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # 10. Export the figure to file (SVG/PNG).
    fig.write_image(output_filename, scale=3)
    
    return output_filename