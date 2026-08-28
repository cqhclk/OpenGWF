# 🌐 OpenGWF: A Comprehensive Collection of Grid-based Weather Forecasting Models

## 📊Introduction
OpenGWF is a comprehensive collection for grid-based weather forecasting, encompassing a broad spectrum of numerical and deep learning-based models. It spans diverse forecasting tasks, ranging from global reanalysis datasets like Weatherbench to regional high-resolution nowcasting scenarios. OpenGWF offers a modular and extensible codebase, excelling in user-friendliness, organization, and reproducibility.
The project is structured to bridge the gap between cutting-edge research and practical implementation. The repository is organized into several key components, including a unified environment setup, standardized data loaders for common weather datasets, and modular implementations of representative models. To ensure ease of use, we provide:
- Unified Environment: A Conda environment file (environment.yml) that installs all necessary dependencies.
- Standardized Data Support: Common interfaces for popular datasets such as ERA5, Weatherbench, and CRA.
- Model Zoo: A growing collection of state-of-the-art grid-based forecasting models with their paper links and reference implementations.
- Reproducibility: Clear documentation and scripts to reproduce experimental results.

We support both an organized library of PyTorch implementations for quick exploration and visualization, making OpenGWF suitable for both researchers and practitioners.
<img width="1171" height="989" alt="image" src="https://github.com/user-attachments/assets/72720985-2392-4596-910b-6787b6d32506" />

<details>
<summary><b>Code Structures</b> </summary>
  
- ✅[PhyDNet](https://arxiv.org/abs/2003.01460) (CVPR'2020)
- ✅[Simvp](https://arxiv.org/abs/2206.05099) (CVPR'2022)
- ✅[FourCastNet](https://arxiv.org/abs/2202.11214) (PASC'23)
- ✅[TAU](https://arxiv.org/abs/2206.12126) (CVPR'2023)
- ✅[WeatherGFT](https://arxiv.org/abs/2405.13796) (NeurIPS'2024)
- ✅[ClimODE](https://arxiv.org/abs/2404.10024) (ICLR'2024)
- ✅[AlphaPre](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_AlphaPre_Amplitude-Phase_Disentanglement_Model_for_Precipitation_Nowcasting_CVPR_2025_paper.html) (CVPR'2025)
- [EarthFormer](https://arxiv.org/abs/2207.05833) (NeurIPS'2022)
- [Prediff](https://arxiv.org/abs/2307.10422) (NeurIPS'2023)
- [ClimaX](https://arxiv.org/abs/2301.10343) (ICML'2023)
- [DiffCast](https://arxiv.org/abs/2312.06734) (CVPR'2024)
- [CasCast](https://arxiv.org/abs/2402.04290) (ICML'2024)
- [OneForecast](https://arxiv.org/abs/2502.00338) (ICML'2025)

</details>

## 📰 News and Updates
[2026-08-29] [OpenGWF](https://github.com/cqhclk/OpenGWF) (OpenGWF V1.0) is realeased.


## ⬇️ Installation
This project provides a Conda environment configuration file (environment.yml). Users can easily reproduce the complete runtime environment for all grid-based weather forecasting models with the following commands:
```bitex
git clone https://github.com/cqhclk/OpenGWF/
cd OpenGWF
conda env create -f environment.yml
conda activate opengwf
```

## 📚 Model Collection

### Grid-based Weather Forecasting Methods

<details>
<summary><b>Currently supported methods</b> </summary>
  
- ✅[PhyDNet](https://arxiv.org/abs/2003.01460) (CVPR'2020)
- ✅[Simvp](https://arxiv.org/abs/2206.05099) (CVPR'2022)
- ✅[FourCastNet](https://arxiv.org/abs/2202.11214) (PASC'23)
- ✅[TAU](https://arxiv.org/abs/2206.12126) (CVPR'2023)
- ✅[WeatherGFT](https://arxiv.org/abs/2405.13796) (NeurIPS'2024)
- ✅[ClimODE](https://arxiv.org/abs/2404.10024) (ICLR'2024)
- ✅[AlphaPre](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_AlphaPre_Amplitude-Phase_Disentanglement_Model_for_Precipitation_Nowcasting_CVPR_2025_paper.html) (CVPR'2025)
- [EarthFormer](https://arxiv.org/abs/2207.05833) (NeurIPS'2022)
- [Prediff](https://arxiv.org/abs/2307.10422) (NeurIPS'2023)
- [ClimaX](https://arxiv.org/abs/2301.10343) (ICML'2023)
- [DiffCast](https://arxiv.org/abs/2312.06734) (CVPR'2024)
- [CasCast](https://arxiv.org/abs/2402.04290) (ICML'2024)
- [OneForecast](https://arxiv.org/abs/2502.00338) (ICML'2025)

</details>

## 📁 Datasets
### Weather Forecasting Benchmarks
<details>
<summary><b>Currently supported datasets</b> </summary>
  
- [weatherbench](https://arxiv.org/abs/2002.00469) (Arxiv'2020) [download](https://mediatum.ub.tum.de/1524895) [config]
- [weatherbenc 2](https://arxiv.org/abs/2308.15560) (Arxiv'2023) [download](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) [config] 
- [CRA](https://link.springer.com/article/10.1007/s13351-023-2086-x) ( Journal of Meteorological Research 2023) [download]()[config] 

</details>

### Data Processing

We provide scripts to convert raw weather data (e.g., from WeatherBench) into `.npy` format for efficient I/O. 

**Data Storage Format**
Each file is saved as a single NumPy array with the naming convention:

`data_path/year/year-hour.npy`

where `year` is the four-digit year, and `hour` is a zero-padded hour index (starting from `0000` for `01-01 00:00`). For example:

- `2000-01-01 00:00:00` → `your_data_path/2000/2000-0000.npy`
- `2010-01-02 01:00:00` → `your_data_path/2010/2010-0025.npy`

**Directory Structure for Training**
For the unified training script, place the `.npy` files in the following layout:

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
  weatherbench2/              # Another dataset (e.g., WeatherBench 2)
    mean_std.npy
    max_min.npy
    2006/
    2007/
    ...
```

**Data Specifications**
- Each `.npy` frame file has the shape `(C, H, W)`.
  - `C`: Number of channels (e.g., geopotential, temperature, humidity).
  - `H`: Grid height (latitude).
  - `W`: Grid width (longitude).
- The number of channels can be flexibly adjusted using the `--channels` argument in the training script.
- The `mean_std.npy` and `max_min.npy` files are used for data normalization; we recommend pre-computing them over the training set.
For more details, please refer to [Data/README.md](Data/README.md).


## ☀️ Visualization
We present visualization examples of some baseline below. For more detailed information, please refer to the visualization.
<img width="1131" height="1066" alt="c34a6555-3d5e-45cd-a1fe-de42e505bf57" src="https://github.com/user-attachments/assets/3aee80df-7ea8-4594-9d65-9f9145bba1ee" />

For more visualization results, please refer to [visualization](visualization)

## 🙏 Acknowledgement
OpenGWF is an open-source project for grid-based weather forecasting algorithms created by researchers in Hunan University. We encourage researchers interested in grid-based weather forecasting to contribute to OpenGWF! We borrow the official implementations of [PhyDNet](https://github.com/vincent-leguen/PhyDNet), [Simvp](https://github.com/vincent-leguen/PhyDNet), [FourCastNet](https://github.com/NVlabs/FourCastNet), [TAU](https://github.com/chengtan9907/OpenSTL/tree/OpenSTL-Lightning/openstl/methods), [WeatherGFT](https://github.com/black-yt/WeatherGFT), [ClimODE](https://github.com/Aalto-QuML/ClimODE), [AlphaPre](https://github.com/Aalto-QuML/ClimODE).

## 📖 Citation

If you find this collection useful for your research, please consider citing the relevant models as follows:
