import os
import datetime
from typing import Optional
from typing_extensions import Literal

from tap import Tap


class Options(Tap):

    train_module: Literal["TrainerBaseline", "TrainerGuidance", "TrainerIterativeGT"] = "TrainerGuidance"
    
    # Model Params from gsplat
    sh_degree: int = 3
    images: str = "images"
    resolution: int = -1
    single_camera: str = '1'
    white_background: bool = False
    random_background: bool = False
    nvs:bool = False # novel view synthesis
    shuffle: bool = True # shuffle the cameras before training

    # General
    wandb: bool = False
    tensorboard: bool = False
    viewer: bool = False # have viewer running during training, creates port conflicts on slurm
    wandb_tags: list[str] = ['baseline']
    seed: int = 42 # random seed
    fp16: bool = False # use mixed precision
    quiet: bool = False # suppress print statements
    
    source_path: str = "" # path to the dataset
    start_checkpoint: Optional[str] = 'scratch' # (optional) specify a checkpoint to start from
    model_path: Optional[str] = None # (optional) specify an exact output directory
    load_iteration: Optional[int] = -1 # specify iteration for evaluation (used in render.py), -1 for latest

    # Pipeline Params from gsplat
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    detect_anomaly: bool = False
    debug: bool = False
    debug_from: int = -1

    # Optimizer Params from gsplat
    iterations: int = 30000
    position_lr_max_steps: int = 30000
    test_iterations: list[int] = [1000, 7000, 10000, 20000, 30000]
    num_eval_views: int = 6
    save_iterations: list[int] = [1000, 10000, 30000]
    checkpoint_iterations: list[int] = [10000, 30000]
    SH_update_freq: int = 1000

    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    
    min_opacity: float = 0.005
    densify_grad_threshold: float = 0.0002

    # Viewer params from gsplat
    ip: str = "127.0.0.1"
    port: int = 6009

    # Test params for render.py
    eval: bool = False
    holdout_cams: Optional[str] = None
    test_cams: Optional[str] = None
    dir_keyword: Optional[str] = None # specify a keyword to filter the output directory
    skip_train: bool = False
    skip_holdout: bool = False
    skip_test: bool = False
    skip_metrics: bool = False
    render_video: bool = False
    trajectory_type: Literal["spiral", "spherical_sample", "elliptical_xy", "elliptical_xz", "elliptical_yz", "spherify", "closest_next"] = "elliptical_xy"
    num_trajectory_views: int = 600
    fps: int = 30

    # Camera interpolation params
    w_translation: float = 0.1 # weight for translation in geodesic distance
    num_consec_views: int = 3 # number of consecutive views to interpolate between
    num_interpolated_views: int = 1 # number of interpolated views between each set of consecutive views

    # Camera perturbation params
    num_perturbed_views: int = 1 # number of perturbed views per camera
    std_dev_translation: float = 0.03
    std_dev_rotation: float = 0.01

    # For each newly added viewpoint, the iteration count for current set of viewpoints grow at a certain rate
    schedule_type: Literal["linear", "constant", "quadratic", "cosine"] = "quadratic"
    init_iter: int = 150 # iterations for the first set of 1 or 2 viewpoint(s) - can change depending on requirements of the schedule
    select_next_viewpoint: Literal["random", "closest", "farthest"] = "closest"
    num_viewpoints_at_once: int = 2
    viewpoints_to_add: Literal["all", 1, 2] = "all"
    add_trajectory_views: bool = False
    
    scale_adjustment_factor: float = 1.0 # adjust scale of gaussians once new viewpoint is added
    # opreset: bool = False # reset opacity of all gaussians after each viewpoint is added
    prob_factor: float = 1.0 # factor to multiply probability of picking each new viewpoint compared to the old ones
    enable_densify_thres_annealing: bool = False
    
    # Guidance Params
    guidance_pipelines: list[str] = ['SDInpaint', 'SDImg2Img']
    realfill_key: str = "output/360_v2/bicycle/realfill/bicycle9_realfill_lora_alpha-27-lora_rank-16-mask_threshold-0.8-max_train_steps-3000"
    ip2p_key: str = "output/360_v2/bicycle/ip2p/bicycle_ip2p_scratch_conditioning_dropout_prob-0.05-learning_rate-0.0001-max_train_steps-3000-resolution-256-train_batch_size-16"
    
    # guidance_keys: list[str] = ["stabilityai/stable-diffusion-2-inpainting", "timbrooks/instruct-pix2pix"]
    realfill_guidance_scale: float = 1.0
    ip2p_guidance_scale: float = 7.0
    
    tokenizer_path: Optional[str] = None 
    text_encoder_path: Optional[str] = None 
    unet_path: Optional[str] = None # fine-tuned UNET using realfill
    vae_sample_mode: Literal["sample", "argmax"] = "sample"

    # inpaint params - 1 implies start from timestep correponding to pure gaussian noise
    strength_inpaint_min_init: float = 0.98
    strength_inpaint_min_final: float = 0.90
    strength_inpaint_max: float = 0.999
    enable_strength_annealing: bool = False
    binary_mask: bool = True
    mask_threshold: float = 0.8
    prompt_inpaint: str = "A photo of <sks>"
    
    # img2img or ip2p params
    strength_img2img: float = 0.3
    strength_ip2p_min_init: float = 0.98
    strength_ip2p_min_final: float = 0.70
    strength_ip2p_max: float = 0.999 # 1.0
    prompt_ip2p: str = "Denoise the noisy image and remove all floaters and Gaussian artifacts."
    image_guidance_scale: float = 2.5

    # similar to IN2N, after how many internal optimization steps should gt of the current (novel )viewpoint be set to None, 
    # so that guidance module can take over
    edit_rate: int = 1
    save_guidance_image_interval: int = 1000
    enable_sds: bool = False # calculate sds loss using the guidance module, else calculate multistep loss like in ReconFusion, SparseFusion
    enable_multistep: bool = False # calculate multistep loss like in Reconfusion, SparseFusion else optimize using 3DGS/sds loss
    w_decay: float = 0.1 # if < 1, loss calculated at novel viewpoints is decayed linearly from 1 to w_decay

    # ControlNet params - https://huggingface.co/docs/diffusers/api/pipelines/controlnet
    controlnet_key: Literal["lllyasviel/control_v11p_sd15_inpaint", "lllyasviel/control_v11f1p_sd15_depth"] = "lllyasviel/control_v11p_sd15_inpaint"
    use_rendered_depth_as_control: bool = False # only for controlnet depth conditioning, controlnetkey must be None
    controlnet_conditioning_scale: float = 0.5
    guess_mode: bool = False
    controlnet_guidance_start: float = 0.0
    controlnet_guidance_end: float = 1.0
    
    # scheduler params
    scheduler: str = 'DDIMScheduler'
    num_inference_steps: int = 20
    timestep_spacing: Literal["leading", "linspace", "trailing"] = "linspace" # linspace to do ddim sampling like in reconfusion
    eta: float = 0.0 # only needed for DDIMScheduler
    
    tomesd_ratio: float = 0.5 # https://huggingface.co/docs/diffusers/optimization/tome
    vram_O: bool = False
    use_safetensors: bool = False

    # textual inversion params for inpainting/img2img
    custom_token: str = "<sks>"
    negative_prompt: str = "lowres, low resolution, worst quality, low quality, jpeg artifacts, lighting artifacts, unrealistic, monochrome, distorted, overexposed, underexposed, low contrast, noisy, watermark, signature"
    enable_blip2_caption: bool = False # induces too much variance in general
    blip2_caption = "a bicycle parked in front of a bench in a park"

    # Repaint Scheduler
    jump_length: int = 10
    jump_n_sample: int = 10

    # Depth Estimation
    mde_module: Literal["Midas", "ZoeDepth"] = "Midas"
    mde_key: Literal["dpt-hybrid-midas", "dpt-large", "ZoeD_N", "ZoeD_K", "ZoeD_NK"] = "dpt-large"
    pseudo_view_from_iter: int = 2000 # start generating pseudo views using interpolated cameras from this iteration
    lambda_dcorr: float = 0.0 # or 0.05 # weight for depth correlation loss applied to training views
    lambda_pseudo: float = 0.0 # or 0.05 # weight for depth correlation loss applied to pseudo views

    @staticmethod
    def check_guidance_params(self):
        assert len(self.guidance_modules) > 0, "At least one guidance module must be specified"

        available_pipelines = ['ControlNetInpaint', 'ControlNetImg2Img', 'SDInpaint', 'SDImg2Img', 'SDRepaint']
        # available_sd_keys = ['stabilityai/stable-diffusion-2-1-base', 'stabilityai/stable-diffusion-2-base', 'runwayml/stable-diffusion-v1-5', 'stabilityai/stable-diffusion-2-inpainting', 'CompVis/stable-diffusion-v1-4']
        # available_controlnet_keys = ['lllyasviel/control_v11p_sd15_inpaint', 'lllyasviel/sd-controlnet-depth', 'lllyasviel/control_v11f1p_sd15_depth']
        available_schedulers = ['LMSDiscreteScheduler', 'EulerDiscreteScheduler', 'EulerAncestralDiscreteScheduler', 'DDIMScheduler', 'UniPCMultistepScheduler', 'PNDMScheduler', 'DPMSolverMultistepScheduler']
        
        if not set(self.guidance_pipelines).issubset(set(available_pipelines)):
            raise ValueError(f"guidance_pipelines must be a subset of {available_pipelines}")
        
        if not self.scheduler in available_schedulers:
            raise ValueError(f"scheduler must be one of {available_schedulers}")

    def process_args(self):        
        if self.debug:
            self.wandb = False
            self.tensorboard = False
            self.model_path = None
            self.quiet = False
            self.test_iterations = [10, 100]
            self.save_iterations = [50]
            self.checkpoint_iterations = []
            self.pseudo_view_from_iter = 20

        if self.eval:
            self.shuffle = False
            assert self.load_iteration is not None, "load_iteration must be specified for evaluation"

        # Set up output directory
        if self.model_path is None:
            timestr = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
            self.model_path = f'output/{timestr}'

        self.source_path = os.path.abspath(self.source_path)
        self.model_path = os.path.abspath(self.model_path)

        if self.start_checkpoint != 'scratch':
            self.start_checkpoint = os.path.abspath(self.start_checkpoint)

        if self.tokenizer_path is not None:
            self.tokenizer_path = os.path.abspath(self.tokenizer_path)
        
        # Set up output folder
        print(f'Workspace: {self.model_path}')