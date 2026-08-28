# 📊 Data Documentation for OpenGWF

This document provides comprehensive instructions for preparing, formatting, and organizing weather data for use with the OpenGWF project. All models in this repository expect input data in a specific `.npy` format to ensure efficient I/O and standardized preprocessing.

---

## 📦 Data Sources

The OpenGWF framework supports various grid-based weather datasets. We primarily recommend the following publicly available sources:

| Dataset | Description | Resolution | Access |
| :--- | :--- | :--- | :--- |
| **WeatherBench** | A benchmark dataset for global weather forecasting based on ERA5 reanalysis. | 1.40625° (128 × 256)<br>2.8125° (64 × 128)<br>5.625° (32 × 64) | [WeatherBench GitHub](https://github.com/pangeo-data/WeatherBench) |
| **WeatherBench 2** | An updated version with improved metrics and additional variables. || [WeatherBench 2 Paper](https://arxiv.org/abs/2308.15560) |
| **ERA5** | High-resolution global atmospheric reanalysis from ECMWF. || [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) |

> **Note:** For the unified training pipeline, we recommend starting with WeatherBench or WeatherBench 2, as they provide pre-processed data in a consistent format.

---

## 🛠️ Data Processing Pipeline

We provide scripts to convert raw weather data (e.g., from WeatherBench) into `.npy` format for efficient I/O. The general pipeline consists of the following steps:

1.  **Download raw data**: Obtain data from sources listed above.
2.  **Select variables**: Choose relevant atmospheric variables (e.g., geopotential, temperature, specific humidity).
3.  **Interpolate (if needed)**: Ensure consistent spatial resolution across different sources.
4.  **Normalize**: Compute global statistics (mean, std, max, min) over the training period.
5.  **Save as `.npy`**: Convert each time step into a separate NumPy array file with the standardized naming convention.

---

## 📁 Data Storage Format

### Naming Convention

Each file is saved as a single NumPy array with the following naming convention:

`data_path/year/year-hour.npy`


- `year`: Four-digit year (e.g., `2006`).
- `hour`: Zero-padded hour index starting from `0000` for `01-01 00:00:00` of that year. The index is computed as `(day_of_year - 1) * 24 + hour_of_day`.

**Examples:**

| Timestamp | File Path |
| :--- | :--- |
| `2000-01-01 00:00:00` | `your_data_path/2000/2000-0000.npy` |
| `2000-01-01 06:00:00` | `your_data_path/2000/2000-0006.npy` |
| `2010-01-02 01:00:00` | `your_data_path/2010/2010-0025.npy` |
| `2010-12-31 23:00:00` | `your_data_path/2010/2010-8759.npy` |

### Data Specifications

- **Shape**: Each `.npy` frame file has the shape `(C, H, W)`.
  - `C`: Number of channels (e.g., geopotential, temperature, humidity at different pressure levels).
  - `H`: Grid height (latitude dimension).
  - `W`: Grid width (longitude dimension).
- **Data Type**: All arrays are stored as `np.float32` to balance precision and memory usage.
- **Channel Order**: Channels are organized consistently across all files. See the `--channels` argument for customization.

---

## 📂 Directory Structure for Training

For the unified training script, place the `.npy` files and normalization statistics in the following layout:

```text
Data/
  weatherbench/               # Dataset name (e.g., WeatherBench)
    mean_std.npy              # Pre-computed mean & std for normalization
    max_min.npy               # Pre-computed max & min values
    2006/
      2006-0000.npy
      2006-0001.npy
      ...
    2007/
    ...
    2018/
  weatherbench2/              # Another dataset (e.g., WeatherBench 2)
    mean_std.npy
    max_min.npy
    2006/
    2007/
    ...
  era5/                       # Custom ERA5 dataset
    mean_std.npy
    max_min.npy
    2010/
    2011/
    ...
```

## 🗂️ File Descriptions
| File / Directory | Description |
| :--- | :--- |
| `Data/{dataset}/` | Root folder for each dataset. Name is user-defined (e.g., `weatherbench`). |
| `mean_std.npy` | NumPy array of shape `(2, C)` containing the mean and standard deviation for each channel, computed over the training set. Used for z-score normalization. |
| `max_min.npy` | NumPy array of shape `(2, C)` containing the max and min values for each channel, computed over the training set. Used for min-max scaling if needed. |
| `{year}/` | Subfolder for each year containing all hourly `.npy` files. |
| `{year}-{hour}.npy` | Individual frame file for a specific hour. |


