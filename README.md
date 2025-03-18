## Connor's Lightning Parser Program

![avg_power_map](strike_avg_power_map.png)

### Note that the library uses CUDA

>1. Install your designated CUDA: https://developer.nvidia.com/cuda-downloads
>
>2. Determine CUDA version by typing into terminal: `nvidia-smi`
>
>![smi](.img/smi-screenshot.png)
>
>3. Install your designated cupy version: https://pypi.org/project/cupy/
>
>![cupy](.img/cupy_versions.png)
>
> As My `nvidia-smi` stated that my CUDA version was 12.8, I would do `pip install cupy-cuda12x`

### To Run the project

1. Drag and drop your LYLOUT text files into "lylout_files" directory.

![lylout](.img/lylout_files.png)

2. Modify the filters in "main.py":
```py
start_time = datetime.datetime(
    2022, 7, 12, 0, 0, tzinfo=datetime.timezone.utc
).timestamp()
end_time = datetime.datetime(
    2022, 7, 12, 23, 0, tzinfo=datetime.timezone.utc
).timestamp()

# Build filter list for time_unix boundaries.
filters = [
    ("time_unix", ">=", start_time),
    ("time_unix", "<=", end_time),
    ("reduced_chi2", "<", 2.0),
    ("num_stations", ">=", 7),
    ("alt", "<=", 17000),  # 20 km = 20000m
    ("alt", ">", 0),  # Above ground
    ("power_db", ">", -4),  # dBW
    ("power_db", "<", 50),  # dBW
]
```

3. Modify parameters
```py
params = {
    "max_lightning_dist": 50000,  # meters
    "max_lightning_speed": 299792.458,  # m/s
    "min_lightning_speed": 0,  # m/s
    "min_lightning_points": 300,  # The minimum number of points to pass the minimum amount
    "max_lightning_time_threshold": 0.15,  # seconds between points
}

lightning_bucketer.USE_CACHE = (
    True  # Generate cache of result to save time for future identical requests
)
```

4. Run with `python run main.py` and observe the new ".png" outputs and a "strikes" directory

> [!NOTE]
> Some individuals may upload a compressed LYLOUT file without adding a suggestive extension filename. Make sure that all LYLOUT files are able to be readable as a text file. If they are not, they are likely compressed, with or without the extension name. It is suggested to try to add the ".gz" extension at the end manually by renaming the file, and attempt to unzip it. If that is not successful, try adding ".zip" and attempt to unzip.
>
>![gz_example](.img/gz_example.png)