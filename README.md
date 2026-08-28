# 🌐 OpenGWF: A Comprehensive Collection of Grid-based Weather Forecasting Models

## 📡Introduction
OpenGWF is a comprehensive collection for grid-based weather forecasting, encompassing a broad spectrum of numerical and deep learning-based models. It spans diverse forecasting tasks, ranging from global reanalysis datasets like Weatherbench to regional high-resolution nowcasting scenarios. OpenGWF offers a modular and extensible codebase, excelling in user-friendliness, organization, and reproducibility.
The project is structured to bridge the gap between cutting-edge research and practical implementation. The repository is organized into several key components, including a unified environment setup, standardized data loaders for common weather datasets, and modular implementations of representative models. To ensure ease of use, we provide:
- Unified Environment: A Conda environment file (environment.yml) that installs all necessary dependencies.
- Standardized Data Support: Common interfaces for popular datasets such as ERA5, Weatherbench, and CRA.
- Model Zoo: A growing collection of state-of-the-art grid-based forecasting models with their paper links and reference implementations.
- Reproducibility: Clear documentation and scripts to reproduce experimental results.

We support both an organized library of PyTorch implementations for quick exploration and visualization, making OpenGWF suitable for both researchers and practitioners.
<img width="1171" height="989" alt="image" src="https://github.com/user-attachments/assets/72720985-2392-4596-910b-6787b6d32506" />


## ⬇️ Installation
This project provides a Conda environment configuration file (environment.yml). Users can easily reproduce the complete runtime environment for all grid-based weather forecasting models with the following commands:
```bitex
git clone https://github.com/kunli1992/OpenGWF/
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

</details>

## 📊 Datasets
### Weather Forecasting Benchmarks
<details>
<summary><b>Currently supported datasets</b> </summary>
  
- [weatherbench](https://arxiv.org/abs/2002.00469) (Arxiv'2020) [download](https://mediatum.ub.tum.de/1524895) [config]
- [weatherbenc 2](https://arxiv.org/abs/2308.15560) (Arxiv'2023) [download](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) [config] 
- ... more coming soon

</details>

### Data Processing
Saving the data as npy file. The folder orgFor the unified training script, place NPY files in the following layout:
```text
Data/
  weatherbench/
    mean_std.npy
    max_min.npy
    2006/
      2006-0000.npy
      2006-0001.npy
      ...
    2007/
    ...
  weatherbench2/
    mean_std.npy
    max_min.npy
    
```

Each frame file should have shape `(C, H, W)`.
The channel count can be changed with `--channels`; WeatherGFT uses `--weathergft_channels` and defaults to 69.



## ☀️ Visualization
We present visualization examples of some baseline below. For more detailed information, please refer to the visualization.
<img width="1131" height="1066" alt="c34a6555-3d5e-45cd-a1fe-de42e505bf57" src="https://github.com/user-attachments/assets/3aee80df-7ea8-4594-9d65-9f9145bba1ee" />


## 📖 Citation

If you find this collection useful for your research, please consider citing the relevant models as follows:
