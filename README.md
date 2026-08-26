
# 🌐 OpenGWF: A Comprehensive Collection of Grid-based Weather Forecasting Models

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
  
- [PhyDNet](https://arxiv.org/abs/2003.01460) (CVPR'2020)
- [Simvp](https://arxiv.org/abs/2206.05099) (CVPR'2022)
- [FourCastNet](https://arxiv.org/abs/2202.11214) (PASC'23)
- [TAU](https://arxiv.org/abs/2206.12126) (CVPR'2023)
- [WeatherGFT](https://arxiv.org/abs/2405.13796) (NeurIPS'2024)
- [ClimODE](https://arxiv.org/abs/2404.10024) (ICLR'2024)
- [AlphaPre](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_AlphaPre_Amplitude-Phase_Disentanglement_Model_for_Precipitation_Nowcasting_CVPR_2025_paper.html) (CVPR'2025)
- ... more coming soon

</details>

## 📊 Datasets
### Weather Forecasting Benchmarks
<details>
<summary><b>Currently supported datasets</b> </summary>
  
- [weatherbench](https://arxiv.org/abs/2002.00469) (Arxiv'2020) [download](https://mediatum.ub.tum.de/1524895) [config]
- [weatherbenc 2]() () [download]() [config] 
- ... more coming soon

</details>


## 📖 Citation

If you find this collection useful for your research, please consider citing the relevant models as follows:
```bibtex
@inproceedings{guen2020disentangling,
  title={Disentangling physical dynamics from unknown factors for unsupervised video prediction},
  author={Guen, Vincent Le and Thome, Nicolas},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={11474--11484},
  year={2020}
}

@inproceedings{gao2022simvp,
  title={Simvp: Simpler yet better video prediction},
  author={Gao, Zhangyang and Tan, Cheng and Wu, Lirong and Li, Stan Z},
  booktitle={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={3160--3170},
  year={2022},
  organization={IEEE}
}

@inproceedings{kurth2023fourcastnet,
  title={Fourcastnet: Accelerating global high-resolution weather forecasting using adaptive fourier neural operators},
  author={Kurth, Thorsten and Subramanian, Shashank and Harrington, Peter and Pathak, Jaideep and Mardani, Morteza and Hall, David and Miele, Andrea and Kashinath, Karthik and Anandkumar, Anima},
  booktitle={Proceedings of the platform for advanced scientific computing conference},
  pages={1--11},
  year={2023}
}

@inproceedings{tan2023temporal,
  title={Temporal attention unit: Towards efficient spatiotemporal predictive learning},
  author={Tan, Cheng and Gao, Zhangyang and Wu, Lirong and Xu, Yongjie and Xia, Jun and Li, Siyuan and Li, Stan Z},
  booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={18770--18782},
  year={2023},
  organization={IEEE}
}

@article{xu2024generalizing,
  title={Generalizing weather forecast to fine-grained temporal scales via physics-ai hybrid modeling},
  author={Xu, Wanghan and Ling, Fenghua and Zhang, Wenlong and Han, Tao and Chen, Hao and Ouyang, Wanli and Bai, Lei},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={23325--23351},
  year={2024}
}

@inproceedings{verma2024climode,
  title={Climode: Climate and weather forecasting with physics-informed neural odes},
  author={Verma, Yogesh and Heinonen, Markus and Garg, Vikas},
  booktitle={International Conference on Learning Representations},
  volume={2024},
  pages={8408--8430},
  year={2024}
}

@inproceedings{lin2025alphapre,
  title={AlphaPre: Amplitude-phase disentanglement model for precipitation nowcasting},
  author={Lin, Kenghong and Zhang, Baoquan and Yu, Demin and Feng, Wenzhi and Chen, Shidong and Gao, Feifan and Li, Xutao and Ye, Yunming},
  booktitle={2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={17841--17850},
  year={2025},
  organization={IEEE}
}

```
