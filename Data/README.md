# Data Directory

This directory intentionally does not contain weather data.

For the unified training script, place NPY files in the following layout:

```text
Data/
  weatherbench/
    mean_std.npy
    max_min.npy
    2006/
      2006-0000.npy
      2006-0001.npy
      ...
```

Each frame file should have shape `(C, H, W)`.
The channel count can be changed with `--channels`; WeatherGFT uses `--weathergft_channels` and defaults to 69.

