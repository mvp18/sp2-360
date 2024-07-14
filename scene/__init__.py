import os
from os.path import join as pjoin
import random
import json

from arguments import Options
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:

    def __init__(self, opt: Options, shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = opt.model_path
        self.train_cameras = {}
        self.holdout_cameras = {}
        self.test_cameras = {}

        if os.path.exists(pjoin(opt.source_path, "sparse")):
            # scene_info = sceneLoadTypeCallbacks["Colmap"](opt.source_path, opt.images, opt.eval)
            self.scene_info = sceneLoadTypeCallbacks["Colmap"](opt.source_path, opt.images, opt.holdout_cams, opt.test_cams)
        elif os.path.exists(pjoin(opt.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            self.scene_info = sceneLoadTypeCallbacks["Blender"](opt.source_path, opt.white_background, opt.eval)
        else:
            assert False, "Could not recognize scene type!"

        if opt.start_checkpoint == 'scratch' and not opt.eval:
            with open(self.scene_info.ply_path, 'rb') as src_file, open(pjoin(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if self.scene_info.test_cameras:
                camlist.extend(self.scene_info.test_cameras)
            if self.scene_info.holdout_cameras:
                camlist.extend(self.scene_info.holdout_cameras)
            if self.scene_info.train_cameras:
                camlist.extend(self.scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(pjoin(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            # Multi-res consistent random shuffling
            random.shuffle(self.scene_info.train_cameras)
            random.shuffle(self.scene_info.holdout_cameras)
            random.shuffle(self.scene_info.test_cameras)

        self.cameras_extent = self.scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale, opt)
            print("Loading Holdout Cameras")
            self.holdout_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.holdout_cameras, resolution_scale, opt)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale, opt)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]
    
    def getHoldoutCameras(self, scale=1.0):
        return self.holdout_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]