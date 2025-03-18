# sudo apt-get install libmagickwand-dev

import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go


def plot_strikes_over_time(bucketed_strikes_indeces_sorted: list[list[int]], events: pd.DataFrame, output_filename="strike_points_over_time.svg") -> str:
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
        title=f"Number of Strike Points Over Time (Start: {global_start_time})",
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
    fig.write_image(output_filename)

    return output_filename

def plot_strike_instance(strike_indeces: list[int], events: pd.DataFrame, output_filename="strike_graph.svg"):
    # Extract the relevant events for this strike instance.
    strike_events = events.iloc[strike_indeces].copy()
    
    # Create a new column 'marker_size': use power_db when > 1, otherwise set a minimum size.
    strike_events['marker_size'] = strike_events['power_db'].apply(lambda x: x if x > 1 else 1)
    
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
        title=f"Lightning Strike Instance (Start: {start_time_dt})",
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
    fig.write_image(output_filename)

    return output_filename

    return png_filename
