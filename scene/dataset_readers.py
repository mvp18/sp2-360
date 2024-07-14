import numpy as np
import json
from pathlib import Path
import os
from os.path import join as pjoin
import sys
from typing import NamedTuple
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    fy: float
    fx: float
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    holdout_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):

        extr = cam_extrinsics[key]
        
        image_path = pjoin(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path)
        if image_name not in os.listdir(images_folder):
            continue

        image_name = image_name.split(".")[0]
        image = Image.open(image_path)
        
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        cam_info = CameraInfo(uid=uid, R=R, T=T, fy=focal_length_y, fx=focal_length_x, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
        
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(f"Finished reading camera {idx+1}/{len(cam_extrinsics)}.")
        sys.stdout.flush()
    
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


# def readColmapSceneInfo(path, images, eval, llffhold=8):
def readColmapSceneInfo(train_set, images, holdout_set=None, test_set=None):

    cam_infos = []
    for p in [train_set, holdout_set, test_set]:
        
        if p is not None:
            
            try:
                cameras_extrinsic_file = pjoin(p, "sparse/0", "images.bin")
                cameras_intrinsic_file = pjoin(p, "sparse/0", "cameras.bin")
                cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            except:
                cameras_extrinsic_file = pjoin(p, "sparse/0", "images.txt")
                cameras_intrinsic_file = pjoin(p, "sparse/0", "cameras.txt")
                cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

            reading_dir = "images" if images == None else images

            cam_info_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=pjoin(p, reading_dir))
            cam_info = sorted(cam_info_unsorted.copy(), key = lambda x : x.image_name)
        else:
            cam_info = []
        
        cam_infos.append(cam_info)

    nerf_normalization = getNerfppNorm(cam_infos[0])

    ply_path = pjoin(train_set, "sparse/0/points3D.ply")
    bin_path = pjoin(train_set, "sparse/0/points3D.bin")
    txt_path = pjoin(train_set, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    pcd = None
    pcd = fetchPly(ply_path)
    
    train_cameras = cam_infos[0]
    holdout_cameras = cam_infos[1]
    test_cameras = cam_infos[2]

    if holdout_set is not None:
        train_cam_names = [cam.image_name for cam in train_cameras]
        # make sure that train and test cameras are disjoint
        holdout_cameras = [cam for cam in holdout_cameras if cam.image_name not in train_cam_names]

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cameras,
                           holdout_cameras=holdout_cameras,
                           test_cameras=test_cameras,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(pjoin(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = pjoin(path, frame["file_path"] + extension)

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = pjoin(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fx = fov2focal(fovx, image.size[0])
            fy = fov2focal(fovy, image.size[1])
            fovy = focal2fov(fx, image.size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, fy=fy, fx=fx, FovY=fovy, FovX=fovx, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = pjoin(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}