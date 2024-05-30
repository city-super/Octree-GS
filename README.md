# *Octree*-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians

### [Project Page](https://city-super.github.io/octree-gs/) | [Paper](https://arxiv.org/abs/2403.17898) | [Viewers for Windows](https://drive.google.com/file/d/12jg6Upko_pAfX3f4jgnb1Q2BhSQDPScE/view?usp=sharing)

[Kerui Ren*](https://github.com/tongji-rkr), [Lihan Jiang*](https://jianglh-whu.github.io/), [Tao Lu](https://github.com/inspirelt), [Mulin Yu](https://scholar.google.com/citations?user=w0Od3hQAAAAJ), [Linning Xu](https://eveneveno.github.io/lnxu), [Zhangkai Ni](https://eezkni.github.io/), [Bo Dai](https://daibo.info/) ✉️ <br />

## News
**[2024.05.30]** 👀We update new mode (`depth`, `normal`, `Gaussian distribution` and `LOD Bias`) in the [viewer](https://github.com/city-super/Octree-GS/tree/main/SIBR_viewers) for Octree-GS.

**[2024.05.30]** 🎈We release the checkpoints for the Mip-NeRF 360, Tanks&Temples, Deep Blending and MatrixCity Dataset.

**[2024.04.08]** 🎈We update the latest quantitative results on three datasets.

**[2024.04.01]** 🎈👀 The [viewer](https://github.com/city-super/Octree-GS/tree/main/SIBR_viewers) for Octree-GS is available now. 

**[2024.04.01]** We release the code.


## Overview

<p align="center">
<img src="assets/pipeline.png" width=100% height=100% 
class="center">
</p>

  Inspired by the Level-of-Detail (LOD) techniques,
  we introduce \modelname, featuring an LOD-structured 3D Gaussian approach supporting level-of-detail decomposition for scene representation that contributes to the final rendering results.
  Our model dynamically selects the appropriate level from the set of multi-resolution anchor points, ensuring consistent rendering performance with adaptive LOD adjustments while maintaining high-fidelity rendering results.

<p align="center">
<img src="assets/teaser_big.png" width=100% height=100% 
class="center">
</p>


## Installation

We tested on a server configured with Ubuntu 18.04, cuda 11.6 and gcc 9.4.0. Other similar configurations should also work, but we have not verified each one individually.

1. Clone this repo:

```
git clone https://github.com/city-super/Octree-GS --recursive
cd Octree-GS
```

2. Install dependencies

```
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate octree_gs
```

## Data

First, create a ```data/``` folder inside the project path by 

```
mkdir data
```

The data structure will be organised as follows:

```
data/
├── dataset_name
│   ├── scene1/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── scene2/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```

### Public Data

- The MipNeRF360 scenes are provided by the paper author [here](https://jonbarron.info/mipnerf360/). 
- The SfM data sets for Tanks&Temples and Deep Blending are hosted by 3D-Gaussian-Splatting [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).
- The BungeeNeRF dataset is available in [Google Drive](https://drive.google.com/file/d/1nBLcf9Jrr6sdxKa1Hbd47IArQQ_X8lww/view?usp=sharing)/[百度网盘[提取码:4whv]](https://pan.baidu.com/s/1AUYUJojhhICSKO2JrmOnCA). 
- The MatrixCity dataset can be downloaded from [Hugging Face](https://huggingface.co/datasets/BoDai/MatrixCity/tree/main)/[Openxlab](https://openxlab.org.cn/datasets/bdaibdai/MatrixCity)/[百度网盘[提取码:hqnn]](https://pan.baidu.com/share/init?surl=87P0e5p1hz9t5mgdJXjL1g). Point clouds used for training in our paper: [pcd](https://drive.google.com/file/d/1J5sGnKhtOdXpGY0SVt-2D_VmL5qdrIc5/view?usp=sharing)
Download and uncompress them into the ```data/``` folder.

### Custom Data

For custom data, you should process the image sequences with [Colmap](https://colmap.github.io/) to obtain the SfM points and camera poses. Then, place the results into ```data/``` folder.


## Training

### Training multiple scenes

To train multiple scenes in parallel, we provide batch training scripts: 

 - MipNeRF360: ```train_mipnerf360.sh```
 - Tanks&Temples: ```train_tandt.sh```
 - Deep Blending: ```train_db.sh```
 - BungeeNeRF: ```train_bungeenerf.sh```
 - MatrixCity: ```train_matrix_city.sh```

 run them with 

 ```
bash train_xxx.sh
 ```

 > Notice 1: Make sure you have enough GPU cards and memories to run these scenes at the same time.

 > Notice 2: Each process occupies many cpu cores, which may slow down the training process. Set ```torch.set_num_threads(32)``` accordingly in the ```train.py``` to alleviate it.

### Training a single scene

For training a single scene, modify the path and configurations in ```single_train.sh``` accordingly and run it:

```
bash single_train.sh
```

- scene: scene name with a format of ```dataset_name/scene_name/``` or ```scene_name/```;
- exp_name: user-defined experiment name;
- gpu: specify the GPU id to run the code. '-1' denotes using the most idle GPU. 
- ratio: sampling interval of the SfM point cloud at initialization
- appearance_dim: dimensions of appearance embedding
- fork: proportion of subdivisions between LOD levels
- base_layer: the coarsest layer of the octree, corresponding to LOD 0, '<0' means scene-based setting
- visible_threshold: the threshold ratio of anchor points with low training frequency
- dist2level: the way floating-point values map to integers when estimating the LOD level
- update_ratio: the threshold ratio of anchor growing
- progressive: whether to use progressive learning
- levels: The number of LOD levels, '<0' means scene-based setting
- init_level: initial level of progressive learning
- extra_ratio: the threshold ratio of LOD bias
- extra_up: Increment of LOD bias per time

> For these public datasets, the configurations of 'voxel_size' and 'fork' can refer to the above batch training script. 


This script will store the log (with running-time code) into ```outputs/dataset_name/scene_name/exp_name/cur_time``` automatically.

## Evaluation

We've integrated the rendering and metrics calculation process into the training code. So, when completing training, the ```rendering results```, ```fps``` and ```quality metrics``` will be printed automatically. And the rendering results will be save in the log dir. Mind that the ```fps``` is roughly estimated by 

```
torch.cuda.synchronize();t_start=time.time()
rendering...
torch.cuda.synchronize();t_end=time.time()
```

which may differ somewhat from the original 3D-GS, but it does not affect the analysis.

Meanwhile, we keep the manual rendering function with a similar usage of the counterpart in [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting), one can run it by 

```
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

## Results

#### Mip-NeRF 360 Dataset

|  scene   | PSNR | SSIM | LPIPS |  GS(k)  |   Mem(MB)   |
| :------: | :------------: | :------------: | :---------------: | :-----: | :---------: |
| [bicycle](https://drive.google.com/file/d/1Cgy4ZhoT8xi8W6cqkl7HCS-xGKim1Rda/view?usp=sharing)  |     25.14      |     0.753      |       0.238       |   701   |   252.07    |
|  [garden](https://drive.google.com/file/d/1DL2YSzb0lRgqBqeot7Be_hlRTvZKIZKE/view?usp=sharing)  |     27.69      |      0.86      |       0.119       |  1344   |   272.67    |
|  [stump](https://drive.google.com/file/d/1Cz93Kx42zhljLw20kMkGpR-jqYiPuv54/view?usp=sharing)   |     26.61      |     0.763      |       0.265       |   467   |   145.50    |
|  [room](https://drive.google.com/file/d/1CxZJaLkEVYosCOFHuqNcPXlFIeQ08Og9/view?usp=sharing)   |     32.53      |     0.937      |       0.171       |   377   |   118.00    |
| [counter](https://drive.google.com/file/d/1CoNYUv-cuMM0XwV6IC-R4K2xRNpQo6t1/view?usp=sharing)  |     30.30      |     0.926      |       0.166       |   457   |   106.98    |
| [kitchen](https://drive.google.com/file/d/1CxH_QQdGccawvCXEDywpOH6cefFfcmpM/view?usp=sharing)  |     31.76      |     0.933      |       0.115       |   793   |   105.16    |
| [bonsai](https://drive.google.com/file/d/1CkDWj6S7eaD_rOYrMWVFGMDoGOduYGs3/view?usp=sharing)  |     33.41      |     0.953      |       0.169       |   474   |    97.16    |
| [flowers](https://drive.google.com/file/d/1ComRj8et--FFAuiIkyUyAGqCuD1XNGpC/view?usp=sharing)  |     21.47      |     0.598      |       0.342       |   726   |   238.57    |
| [treehill](https://drive.google.com/file/d/1D40-oLQI_UIH2m4vhFjmXTEsumvjCzE7/view?usp=sharing) |     23.19      |     0.645      |       0.347       |   545   |   211.90    |
|   avg    |   **28.01**    |   **0.819**    |     **0.215**     | **654** | **172.00**  |
|  paper   |     27.73      |     0.815      |       0.217       |   686   |   489.59    |
|          |     +0.28      |     +0.004     |      -0.002       | -4.66%  | **-64.87%** |

#### Tanks and Temples Dataset

| scene | PSNR | SSIM | LPIPS |  GS(k)  |   Mem(MB)   |
| :---: | :------------: | :------------: | :---------------: | :-----: | :---------: |
| [truck](https://drive.google.com/file/d/1Di60jON2SF-Q-Gs-VRMH0lHNeirg6wHo/view?usp=sharing) |     26.17      |     0.892      |       0.127       |   401   |    84.42    |
| [train](https://drive.google.com/file/d/1DOeWKCgLsRIcVHjz31dXQl9ONGvGBbV6/view?usp=sharing) |     23.04      |     0.837      |       0.184       |   446   |    84.45    |
|  avg  |   **24.61**    |     0.865      |       0.156       | **424** |  **84.44**  |
| paper |     24.52      |   **0.866**    |     **0.153**     |   481   |   410.48    |
|       |     +0.09      |     -0.001     |      +0.003       | -11.85% | **-79.43%** |

#### Deep Blending Dataset

|   scene   | PSNR | SSIM | LPIPS |  GS(k)  |   Mem(MB)   |
| :-------: | :------------: | :------------: | :---------------: | :-----: | :---------: |
| [drjohnson](https://drive.google.com/file/d/1DpHo1yeJqWODQZS3nELOIp2OSxvHz7Jj/view?usp=sharing) |     29.89      |     0.911      |       0.234       |   132   |   132.43    |
| [playroom](https://drive.google.com/file/d/1DpM0hp8nIs4BAhbdN4309W33UAJkYdMb/view?usp=sharing)  |     31.08      |     0.914      |       0.246       |   93    |    53.94    |
|    avg    |   **30.49**    |   **0.913**    |       0.240       | **113** |  **93.19**  |
|   paper   |     30.41      |   **0.913**    |     **0.238**     |   144   |   254.87    |
|           |     +0.08      |       -        |      +0.002       | -21.52% | **-63.44%** |

#### MatrixCity Dataset

|   scene   | PSNR | SSIM | LPIPS |  GS(k)  |   Mem(GB)   |
| :-------: | :------------: | :------------: | :---------------: | :-----: | :---------: |
| [Block_All](https://drive.google.com/file/d/1E0R4dnzdTxgWSd14rvaM89haEfKLYxPD/view?usp=sharing) |     26.99      |   0.833    |     0.257     |   453   |   2.36    |
|   paper   |     26.41      |   0.814    |     0.282     |   665   |   3.70    |
|           |     +0.59      |    +0.019       |   -0.025  | -31.87% | -36.21% |

## Viewer

The [viewers](https://github.com/city-super/Octree-GS/tree/main/SIBR_viewers) for Octree-GS is available now. 
Please follow the following format

```
<location>
|---point_cloud
|   |---point_cloud.ply
|   |---color_mlp.pt
|   |---cov_mlp.pt
|   |---opacity_mlp.pt
|   (|---embedding_appearance.pt)
|---cameras.json
|---cfg_args
```

or 

```
<location>
|---point_cloud
|   |---iteration_{ITERATIONS}
|   |   |---point_cloud.ply
|   |   |---color_mlp.pt
|   |   |---cov_mlp.pt
|   |   |---opacity_mlp.pt
|   |   (|---embedding_appearance.pt)
|---cameras.json
|---cfg_args
```


## Contact

- Kerui Ren: renkerui@pjlab.org.cn
- Lihan Jiang: mr.lhjiang@gmail.com

## Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{octreegs,
      title={Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians}, 
      author={Kerui Ren and Lihan Jiang and Tao Lu and Mulin Yu and Linning Xu and Zhangkai Ni and Bo Dai},
      year={2024},
      eprint={2403.17898},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## Acknowledgement

We thank all authors from [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) and [Scaffold-GS](https://github.com/city-super/Scaffold-GS) for presenting such an excellent work.
