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

from PIL import Image
import torch
import time
import shutil
import sys
import os
from os.path import join as pjoin
from datetime import datetime
import numpy as np
import random
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution=None):
    
    if resolution is not None:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    
def torchToPIL(tensor):
    np_array = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(np_array)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

class Iter_Schedule():
    def __init__(self, opt, num_viewpoints):
        self.iterations = opt.iterations
        self.init_iter = opt.init_iter
        self.num_viewpoints = num_viewpoints

    def eq_linear(self, x):
        return sum(self.init_iter + i * x for i in range(self.num_viewpoints)) - self.iterations
    
    def eq_quadratic(self, x):
        return sum(self.init_iter + x * (i**2) for i in range(self.num_viewpoints)) - self.iterations

    def constant(self):
        iteration_count_per_view = {d: int(self.iterations / self.num_viewpoints) for d in range(1, self.num_viewpoints + 1)}
        if sum(iteration_count_per_view.values()) != self.iterations:
            iteration_count_per_view[self.num_viewpoints] += self.iterations - sum(iteration_count_per_view.values())

        return iteration_count_per_view
    
    def linear(self):

        inc_iter = -1
        while inc_iter <= 0:
            inc_iter = fsolve(self.eq_linear, x0=1.0)[0]
            self.init_iter -= 1
        
        assert self.eq_linear(inc_iter) < 1e-3 and inc_iter > 0, "Linear schedule not possible with the given parameters"
        iteration_count_per_view = {d: self.init_iter + int(inc_iter * (d - 1)) for d in range(1, self.num_viewpoints + 1)}

        if sum(iteration_count_per_view.values()) != self.iterations:
            iteration_count_per_view[self.num_viewpoints] += self.iterations - sum(iteration_count_per_view.values())

        return iteration_count_per_view

    def quadratic(self):

        inc_iter = -1
        while inc_iter <= 0:
            inc_iter = fsolve(self.eq_quadratic, x0=1.0)[0]
            self.init_iter -= 1
        
        assert self.eq_quadratic(inc_iter) < 1e-3 and inc_iter > 0, "Quadratic schedule not possible with the given parameters"
        iteration_count_per_view = {d: self.init_iter + int(inc_iter * (d - 1)**2) for d in range(1, self.num_viewpoints + 1)}

        if sum(iteration_count_per_view.values()) != self.iterations:
            iteration_count_per_view[self.num_viewpoints] += self.iterations - sum(iteration_count_per_view.values())

        return iteration_count_per_view

    def cosine(self, max_iter=300):
        
        iteration_count_per_view = {d: int(self.init_iter + 0.5 * (max_iter - self.init_iter) * (1 + np.cos(np.pi * (2 * (d - 1) / self.num_viewpoints - 1)))) 
                                    for d in range(1, self.num_viewpoints + 1)}
        # Ensure the total iterations do not exceed the available budget
        total_iterations_used = sum(iteration_count_per_view.values())
        scaling_factor = self.iterations / total_iterations_used
        for d in range(1, self.num_viewpoints + 1):
            iteration_count_per_view[d] = int(iteration_count_per_view[d] * scaling_factor)

        if sum(iteration_count_per_view.values()) != self.iterations:
            # Add the remaining iterations to the view with the highest iteration count
            iteration_count_per_view[max(iteration_count_per_view, key=iteration_count_per_view.get)] += self.iterations - sum(iteration_count_per_view.values())

        return iteration_count_per_view

    def plot_schedule(self, iter_schedule):

        fig, ax1 = plt.subplots(figsize=(15, 6))

        # Original plot
        ax1.plot(iter_schedule.keys(), iter_schedule.values(), color='r')
        ax1.set_xlabel("num_novel_viewpoints")
        ax1.set_ylabel("iterations", color='r')
        ax1.tick_params('y', colors='r')
        ax1.set_title("Iterations Schedule")

        # Cumulative plot
        ax2 = ax1.twinx()
        cumulative_values = [sum(list(iter_schedule.values())[:i+1]) for i in range(len(iter_schedule))]
        ax2.plot(iter_schedule.keys(), cumulative_values, color='b')
        ax2.set_ylabel("cumulative iterations", color='b')
        ax2.tick_params('y', colors='b')

        return fig

def safe_state(opt):
    silent = opt.quiet
    seed = opt.seed
    
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.device = device
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

def get_wandb_run_id(scene_dir):

    local_run_dir = pjoin(scene_dir, "wandb")
    run_folder = [item for item in os.listdir(local_run_dir) if item.startswith('run-') and os.path.isdir(pjoin(local_run_dir, item))][0]
    run_id = run_folder.split('-')[-1]

    return run_id

def force_delete_directory(dir_path):
    """ Forcefully deletes a directory and all its contents, with retries. """
    max_retries = 5
    retry_delay = 1  # seconds

    for _ in range(max_retries):
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            return
        except OSError as e:
            print(f"Deletion failed with error: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    raise Exception(f"Failed to delete directory {dir_path} after {max_retries} retries.")
