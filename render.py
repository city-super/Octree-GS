#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
from os import makedirs
import torch
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, show_level, ape_code):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(gts_path, exist_ok=True)
    if show_level:
        render_level_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_level")
        makedirs(render_level_path, exist_ok=True)

    t_list = []
    per_view_dict = {}
    per_view_level_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t0 = time.time()

        gaussians.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, ape_code=ape_code)
        
        torch.cuda.synchronize(); t1 = time.time()
        t_list.append(t1-t0)

        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()  
        per_view_dict['{0:05d}'.format(idx)+".png"] = visible_count.item()

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        if show_level:
            for cur_level in range(gaussians.levels):
                gaussians.set_anchor_mask_perlevel(view.camera_center, view.resolution_scale, cur_level)
                voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
                render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, ape_code=ape_code)
                
                rendering = render_pkg["render"]
                visible_count = render_pkg["visibility_filter"].sum()
                
                torchvision.utils.save_image(rendering, os.path.join(render_level_path, '{0:05d}_LOD{1:d}'.format(idx, cur_level) + ".png"))
                per_view_level_dict['{0:05d}_LOD{1:d}'.format(idx, cur_level) + ".png"] = visible_count.item()

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True) 
    if show_level:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count_level.json"), 'w') as fp:
            json.dump(per_view_level_dict, fp, indent=True)     
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, show_level : bool, ape_code : int):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
            dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales)
        gaussians.eval()
        gaussians.plot_levels()
        if dataset.random_background:
            bg_color = [np.random.random(),np.random.random(),np.random.random()] 
        elif dataset.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, show_level, ape_code)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, show_level, ape_code)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=10, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show_level", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.show_level, args.ape)
    
