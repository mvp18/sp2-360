import os
from os.path import join as pjoin
import shutil
from argparse import ArgumentParser
import subprocess

from scene.dataset_readers import read_points3D_binary, storePly
from utils.system_utils import mkdir_p


def run_colmap(args):

    image_path = pjoin(args.data_path, "images")
    
    # Feature extraction
    subprocess.run(['colmap', 'feature_extractor', '--database_path', pjoin(args.data_path, "database.db"), '--image_path', image_path, '--SiftExtraction.estimate_affine_shape=true', '--SiftExtraction.domain_size_pooling=true', '--ImageReader.camera_model', 'PINHOLE', '--ImageReader.single_camera', args.single_camera])

    # Feature matching
    subprocess.run(['colmap', 'exhaustive_matcher', '--database_path', pjoin(args.data_path, "database.db"), '--SiftMatching.guided_matching=true'])

    # Sparse reconstruction
    sparse_ply_dir = pjoin(args.data_path, "sparse")
    mkdir_p(sparse_ply_dir)
    subprocess.run(['colmap', 'mapper', '--database_path', pjoin(args.data_path, "database.db"), '--image_path', image_path, '--output_path', sparse_ply_dir, '--Mapper.min_model_size', str(args.min_model_size), '--Mapper.max_model_overlap', str(args.max_model_overlap)])
    # '--Mapper.multiple_models=false'

    # Image undistortion
    undistorted_dir = pjoin(args.data_path, "undistorted_images")
    mkdir_p(undistorted_dir)
    subprocess.run(['colmap', 'image_undistorter', '--image_path', image_path, '--input_path', pjoin(sparse_ply_dir, '0'), '--output_path', undistorted_dir, '--output_type', 'COLMAP'])

    if len(os.listdir(pjoin(undistorted_dir, 'images'))) != len(os.listdir(image_path)):
        # dump a flag file to indicate that the undistortion failed
        with open(pjoin(args.data_path, 'undistortion_failed'), 'w') as f:
            f.write("")

    # replace images directory with undistorted images
    shutil.rmtree(image_path)
    shutil.move(pjoin(undistorted_dir, 'images'), image_path)
    shutil.rmtree(undistorted_dir)

    xyz, rgb, _ = read_points3D_binary(pjoin(sparse_ply_dir, "0/points3D.bin"))
    ply_path = pjoin(sparse_ply_dir, "0/points3D.ply")
    storePly(ply_path, xyz, rgb)

if __name__ == "__main__":

    parser = ArgumentParser(description="Convert images to COLMAP format")
    parser.add_argument("--data_path", required=True, type=str, help="Path to the data directory")
    parser.add_argument("--single_camera", type=str, default='1', choices=['0', '1'], help="Whether to use a single camera model")
    # A higher overlap threshold means that models need to share more images before they're considered separate reconstructions. 
    # This can help in merging models that might otherwise be separate due to minimal disconnects.
    parser.add_argument("--max_model_overlap", required=False, type=int, default=20)
    # Setting a higher threshold for what constitutes a valid model encourages the algorithm to build larger, more comprehensive reconstructions rather than settling for smaller, disconnected pieces.
    parser.add_argument("--min_model_size", required=False, type=int, default=10)
    args = parser.parse_args()
    
    run_colmap(args)