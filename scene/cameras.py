import torch
from torch import nn
import numpy as np
import copy

from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, fx, fy, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, image_height=None, image_width=None, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"):
        super().__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.fy = fy
        self.fx = fx
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        else:
            self.image_width = image_width
            self.image_height = image_height
            self.original_image = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def clone(self):
        return copy.deepcopy(self)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def interpolate_cameras(cameras, num_samples: int):

    assert len(cameras) > 1, "Cannot interpolate a single camera"

    rotations = []
    translations = []
    image_names = []
    # image_heights = []
    # image_widths = []
    # fxs = []
    # fys = []

    for cam in cameras:
        rotations.append(cam.R)
        translations.append(cam.T)
        image_names.append(cam.image_name)
        # image_heights.append(cam.image_height)
        # image_widths.append(cam.image_width)
        # fxs.append(cam.fx)
        # fys.append(cam.fy)

    rotations = np.array(rotations)
    translations = np.array(translations)
    # image_heights = np.array(image_heights)
    # image_widths = np.array(image_widths)
    # fxs = np.array(fxs)
    # fys = np.array(fys)

    in_times = np.linspace(0, 1, len(cameras))
    
    # Adjust out_times to exclude 0 and 1
    if num_samples == 1:
        out_times = np.array([0.5])  # Midpoint for a single sample
    else:
        out_times = np.linspace(0, 1, num_samples + 2)[1:-1]  # Exclude the first and last viewpoints in output

    slerp = Slerp(in_times, R.from_matrix(rotations))
    out_rotations = slerp(out_times).as_matrix()

    if len(cameras) > 2:
        spline_tr = CubicSpline(in_times, translations)
        out_translations = spline_tr(out_times)

        # spline_ht = CubicSpline(in_times, image_heights)
        # out_image_heights = spline_ht(out_times)

        # spline_wd = CubicSpline(in_times, image_widths)
        # out_image_widths = spline_wd(out_times)

        # spline_fx = CubicSpline(in_times, fxs)
        # out_fxs = spline_fx(out_times)

        # spline_fy = CubicSpline(in_times, fys)
        # out_fys = spline_fy(out_times)
    else:
        f_trans = interp1d(in_times, translations, axis=0, kind='linear')
        out_translations = f_trans(out_times)

        # f_image_heights = interp1d(in_times, image_heights, axis=-1, kind='linear')
        # out_image_heights = f_image_heights(out_times)
        
        # f_image_widths = interp1d(in_times, image_widths, axis=-1, kind='linear')
        # out_image_widths = f_image_widths(out_times)

        # f_fxs = interp1d(in_times, fxs, axis=-1, kind='linear')
        # out_fxs = f_fxs(out_times)

        # f_fys = interp1d(in_times, fys, axis=-1, kind='linear')
        # out_fys = f_fys(out_times)

    ref_cam = cameras[0].clone()

    # All cameras have the same intrinsic parameters anyway
    out_fx = ref_cam.fx
    out_fy = ref_cam.fy
    out_wd = ref_cam.image_width
    out_ht = ref_cam.image_height
    
    out_cams = []
    
    # for i, (out_rot, out_trans, out_ht, out_wd, out_fx, out_fy) in enumerate(zip(out_rotations, out_translations, out_image_heights, out_image_widths, out_fxs, out_fys)):
    for i, (out_rot, out_trans) in enumerate(zip(out_rotations, out_translations)):

        image_name = "_".join(image_names + [str(i+1)])

        out_FoVx = focal2fov(out_fx, out_wd)
        out_FoVy = focal2fov(out_fy, out_ht)

        out_cam = Camera(colmap_id=ref_cam.colmap_id, R=out_rot, T=out_trans, fx=out_fx, fy=out_fy,
                         FoVx=out_FoVx, FoVy=out_FoVy, image=None, gt_alpha_mask=None, image_name=image_name, 
                         uid=ref_cam.uid, image_height=int(out_ht), image_width=int(out_wd), data_device=ref_cam.data_device)
        
        out_cams.append(out_cam)

    return out_cams


def distance_SE3(cam1, cam2, w_translation=0.4):
    """
    Compute the distance between two SE3 transformations
    https://math.stackexchange.com/questions/2231466/metric-on-se3
    http://www.boris-belousov.net/2016/12/01/quat-dist/
    """
    R1 = cam1.R
    R2 = cam2.R

    T1 = cam1.T
    T2 = cam2.T

    assert np.linalg.norm(R1, ord=2) - 1 < 1e-5 and np.linalg.det(R1) - 1 < 1e-5, "R1 is not a proper rotation matrix"
    assert np.linalg.norm(R2, ord=2) - 1 < 1e-5 and np.linalg.det(R2) - 1 < 1e-5, "R2 is not a proper rotation matrix"

    R = R1 @ R2.transpose()
    # T12 = T1 - R2.transpose() @ T2
    # T21 = T2 - R1.transpose() @ T1
    
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta_clamped = np.clip(cos_theta, -1, 1)
    rotation_distance = np.arccos(cos_theta_clamped)
    
    translation_distance = np.linalg.norm(T1 - T2)
    # translation_distance = 0.5 * (np.linalg.norm(T12) + np.linalg.norm(T21))
    d_geodesic = rotation_distance + w_translation * translation_distance

    # print(f"rotation distance: {rotation_distance}, translation distance: {translation_distance}, total distance: {d_geodesic}")

    return d_geodesic


def calc_distance_matrix(cameras1, cameras2, w_translation=0.4):

    num_cams1 = len(cameras1)
    num_cams2 = len(cameras2)
    distance_matrix = np.zeros((num_cams1, num_cams2))

    for i in range(num_cams1):
        for j in range(num_cams2):
            distance_matrix[i, j] = distance_SE3(cameras1[i], cameras2[j], w_translation=w_translation)

    return distance_matrix


def random_three_vector():
    """
    https://gist.github.com/andrewbolster/10274979
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos(costheta)
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return np.array([x,y,z])


def perturb_viewpoint(viewpoint_cam, mean=0, std_dev_rotation=0.01, std_dev_translation=0.03, num_samples=1):

    """
    https://math.stackexchange.com/questions/3827667/perturbing-rotation-matrices-as-a-means-to-pertub-points-in-euclidean-space
    """

    out_cams = []

    for i in range(num_samples):

        # might as well use gaussian perturbation
        # trans_perturbation = np.random.uniform(-std_dev_translation, std_dev_translation, size=3)
        # trans_x = np.random.normal(mean, std_dev_translation)
        # trans_y = np.random.normal(mean, std_dev_translation)
        # trans_z = np.random.normal(mean, std_dev_translation)
        # translate = np.array([trans_x, trans_y, trans_z])
        translate = np.random.normal(mean, std_dev_translation, size=3)
        
        # angle_perturbation = np.random.normal() # radians        
        # perturbation_axis = random_three_vector()
        # # A random axis and a random angle would produce rotations about all 3 x, y, z axes
        # # Rodrigues formula
        # perturbation_matrix = np.cos(angle_perturbation)*np.eye(3) + \
        #                       np.sin(angle_perturbation) * np.cross(np.eye(3), perturbation_axis) + \
        #                       (1 - np.cos(angle_perturbation)) * np.outer(perturbation_axis, perturbation_axis)
        
        # # should produce same result as Rodrigues formula above
        # assert np.allclose(perturbation_matrix, 
        #                    R.from_rotvec(angle_perturbation * perturbation_axis).as_matrix(),
        #                    atol=1e-8), "Rodrigues formula and rotation vector formula are not the same"

        angle_x = np.random.normal(mean, std_dev_rotation)
        angle_y = np.random.normal(mean, std_dev_rotation)
        angle_z = np.random.normal(mean, std_dev_rotation)
        # combined rotation matrix
        perturbation_matrix = R.from_euler('xyz', [angle_x, angle_y, angle_z]).as_matrix()

        # switching the order of multplication would give incorrect effective rotation
        perturbed_R = viewpoint_cam.R @ perturbation_matrix
        # perturbed_R = np.matmul(viewpoint_cam.R, perturbation_matrix)
        
        ref_cam = viewpoint_cam.clone()
        image_name = f"{ref_cam.image_name}_perturbed_{i}"
        
        out_cam = Camera(colmap_id=ref_cam.colmap_id, R=perturbed_R, T=ref_cam.T + translate, fx=ref_cam.fx, fy=ref_cam.fy,
                         FoVx=ref_cam.FoVx, FoVy=ref_cam.FoVy, image=None, gt_alpha_mask=None, image_name=image_name, uid=ref_cam.uid, 
                         image_height=ref_cam.image_height, image_width=ref_cam.image_width, data_device=ref_cam.data_device)
        out_cams.append(out_cam)

    return out_cams