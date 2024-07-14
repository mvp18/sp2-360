import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from os.path import join as pjoin

from scene import Scene
from scene.cameras import calc_distance_matrix
from scene.dataset_readers import fetchPly

from utils.run_colmap import run_colmap
from utils.general_utils import safe_state
from utils.system_utils import mkdir_p

from arguments import Options


def filter_n_views(opt: Options, nth_closest_view=0):

    scene = Scene(opt, shuffle=False)
    train_cameras = scene.getTrainCameras()

    new_data_dirname = opt.source_path.split("/")[-1].split("_")[0] + f"_{opt.num_eval_views}views_{nth_closest_view + 1}"
    new_data_path = pjoin(opt.source_path[:opt.source_path.rfind('/')], new_data_dirname)
    new_image_path = pjoin(new_data_path, "images")
    mkdir_p(new_image_path)

    file_ext = os.listdir(pjoin(opt.source_path, "images"))[0].split(".")[-1]

    viewpoint_stack = []
    idx = 0

    while len(train_cameras) > 0 and idx < opt.num_eval_views:

        if idx==0:
            view = train_cameras.pop(0)
            viewpoint_stack.append(view)
        else:
            geodesic_distances = calc_distance_matrix(viewpoint_stack, train_cameras, w_translation=opt.w_translation)
            # min_dist_index = np.argmin(geodesic_distances)
            min_distances = np.min(geodesic_distances, axis=0)
            nth_closest_view_idx = np.argsort(min_distances)[nth_closest_view]
            # _, viewpoint_idx = np.unravel_index(min_dist_index, geodesic_distances.shape)
            # view = train_cameras.pop(viewpoint_idx)
            view = train_cameras.pop(nth_closest_view_idx)
            viewpoint_stack.append(view)

        shutil.copyfile(pjoin(opt.source_path, "images", f"{view.image_name}.{file_ext}"), pjoin(new_image_path, f"{view.image_name}.{file_ext}"))
        idx += 1

    max_geodesic_distance = np.max(calc_distance_matrix(viewpoint_stack, viewpoint_stack, w_translation=opt.w_translation))

    opt.data_path = new_data_path
    opt.min_model_size = 10
    opt.max_model_overlap = 20

    run_colmap(opt)
    sparse_ply_dir = pjoin(new_data_path, "sparse")

    if os.path.exists(pjoin(new_data_path, "undistortion_failed")) or len(os.listdir(sparse_ply_dir))==0:
        return 0, 0

    ply_path = pjoin(sparse_ply_dir, "0/points3D.ply")
    
    camera_path_txt = pjoin(sparse_ply_dir, "0/cameras.txt")
    camera_path_bin = pjoin(sparse_ply_dir, "0/cameras.bin")
    
    if os.path.exists(camera_path_bin):
        os.remove(camera_path_bin)
    else:
        os.remove(camera_path_txt)
        
    image_path_txt = pjoin(sparse_ply_dir, "0/images.txt")
    image_path_bin = pjoin(sparse_ply_dir, "0/images.bin")

    if os.path.exists(image_path_bin):
        os.remove(image_path_bin)
    else:
        os.remove(image_path_txt)

    try:
        # replace intrinsice camera file with full camera list
        shutil.copyfile(pjoin(opt.source_path, "sparse/0/cameras.bin"), camera_path_bin)
        shutil.copyfile(pjoin(opt.source_path, "sparse/0/images.bin"), image_path_bin)
    except:
        # replace extrinsice camera file with full camera list
        shutil.copyfile(pjoin(opt.source_path, "sparse/0/cameras.txt"), camera_path_txt)
        shutil.copyfile(pjoin(opt.source_path, "sparse/0/images.txt"), image_path_txt)

    pcd = fetchPly(ply_path)
    num_points = np.asarray(pcd.points).shape[0]

    return num_points, max_geodesic_distance


if __name__ == "__main__":
    
    opt = Options().parse_args()
    # Initialize system state (RNG)
    safe_state(opt)

    pts = []
    max_dists = []

    for n in range(200):
        try:
            num_points, max_geodesic_distance = filter_n_views(opt, nth_closest_view=n)
            pts.append(num_points)
            max_dists.append(max_geodesic_distance)
        except Exception as e:
            print(f"Reconstruction failed for n={n} with error:", e)
            break
    
    if n == 199:
        n = n + 1

    match_pattern = f"_{opt.num_eval_views}views_"
    candidate_dirs = [pjoin(opt.source_path, d) for d in os.listdir(opt.source_path) if match_pattern in d]
    split_dir = candidate_dirs[np.argmax(max_dists)]
    print(f"Directory with best scene coverage: {split_dir}")
    
    fig, ax1 = plt.subplots(figsize=(15, 6))
    x = np.arange(1, n+1)

    ax1.plot(x, pts, color='r')
    ax1.set_xlabel("Nth Closest Viewpoint to Add")
    ax1.set_ylabel("Point Cloud Size", color='r')
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(x[::2], rotation=90)
    ax1.tick_params('y', colors='r')
    ax1.set_title("Point Cloud Size and Maxm Geodesic Distance vs Nth Closest Viewpoint")

    ax2 = ax1.twinx()
    ax2.plot(x, max_dists, color='b')
    ax2.set_ylabel("Max Geodesic Distance", color='b')
    ax2.tick_params('y', colors='b')

    plt.tight_layout()
    
    # Save the figure
    savedir = f"media/{opt.source_path.split('/')[-1].split('_')[0]}"
    mkdir_p(savedir)
    plt.savefig(f"{savedir}/filter_{opt.num_eval_views}_views_trans{opt.w_translation}.png")