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

from errno import EEXIST
from os import makedirs, path
from os.path import join as pjoin
import os
import shutil

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def clean_saved_logs(folder_path, keyword, subdir_names=[]):
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            if keyword in dir:
                subdirs = [d for d in os.listdir(pjoin(root, dir)) if path.isdir(pjoin(root, dir, d))]
                for subdir in subdirs:
                    if subdir in subdir_names:
                        shutil.rmtree(pjoin(root, dir, subdir))
