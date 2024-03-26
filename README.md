# *Octree*-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians

### [Project Page](https://city-super.github.io/octree-gs/) | [Paper](https://city-super.github.io/octree-gs/)

[Kerui Ren*](https://github.com/tongji-rkr), [Lihan Jiang*](https://jianglh-whu.github.io/), [Tao Lu](https://github.com/inspirelt), [Mulin Yu](https://scholar.google.com/citations?user=w0Od3hQAAAAJ), [Linning Xu](https://eveneveno.github.io/lnxu), [Yuanbo Xiangli](https://kam1107.github.io/), [Zhangkai Ni](https://kam1107.github.io/), [Bo Dai](https://daibo.info/) ✉️ <br />

## Overview
<p align="center">
<img src="assets/pipeline.png" width=100% height=100% 
class="center">
</p>
Starting from a sparse point cloud, we construct an octree for the bounded 3D space. Each octree level provides a set of anchor Gaussians assigned to the corresponding LOD level. Unlike conventional 3D-GS methods treating all Gaussians equally, our approach involves anchor Gaussians with different LODs. During novel view rendering, we determine the required LOD level for each occupied anchor voxel within the octree from the observation center and invoke all anchor Gaussians up to that level for final rendering. This process, shown in the middle, results in an increased level of detail by gradually fetching anchors from higher LODs in an accumulation manner. Our model is trained with standard image reconstruction loss and an additional regularization loss.


### The code is coming soon~ 😘