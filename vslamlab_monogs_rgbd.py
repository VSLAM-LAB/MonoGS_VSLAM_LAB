import os
import sys
import time
import argparse
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians, write_vslamlab_trajectory
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        backend_process.start()
        self.frontend.run()
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        write_vslamlab_trajectory(self.frontend.cameras, self.frontend.kf_indices,
                             config["Results"]['save_dir'], config["Results"]['exp_id'],                    
                             config["Dataset"]["rgb_csv"], config["Dataset"]["cam_mono"])

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            backend_queue.put(["color_refinement"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
            )
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, help="path to image directory")
    parser.add_argument("--calibration_yaml", type=str, help="path to calibration file")
    parser.add_argument("--rgb_csv", type=str, help="path to image list")
    parser.add_argument("--exp_folder", type=str, help="path to save results")
    parser.add_argument("--exp_it", type=str, help="experiment iteration")
    parser.add_argument("--settings_yaml", type=str, help="settings_yaml")
    parser.add_argument("--verbose", type=str, help="verbose")

    args, unknown = parser.parse_known_args()
    sequence_path = args.sequence_path
    exp_id = args.exp_it
    exp_folder = args.exp_folder
    config_yaml = args.settings_yaml
    verbose = bool(int(args.verbose))
    rgb_csv = args.rgb_csv

    mp.set_start_method("spawn")

    config = load_config(config_yaml)
    print(f"Loading config from {config_yaml}")
    config["Dataset"]["type"] = "vslamlab_rgbd"
    config["Dataset"]["sensor_type"] = "depth"
    config["Dataset"]["dataset_path"] = sequence_path
    config["Dataset"]["rgb_csv"] = rgb_csv
    config["Results"]["save_dir"] = exp_folder
    config["Results"]["exp_id"] = exp_id

    ####################################################################
    with open(args.calibration_yaml, 'r') as file:
        data = yaml.safe_load(file)
    cameras = data.get('cameras', [])
    for cam_ in cameras:
        if cam_['cam_name'] == config["Dataset"]["cam_rgbd"]:
            cam = cam_;
            break;
    print(f"\nCamera Name: {cam['cam_name']}")
    print(f"Camera Type: {cam['cam_type']}")
    print(f"Camera Model: {cam['cam_model']}")
    print(f"Focal Length: {cam['focal_length']}")
    print(f"Principal Point: {cam['principal_point']}")
    has_dist = ('distortion_type' in cam) and ('distortion_coefficients' in cam)
    if has_dist:
        print(f"Distortion Type Dimension: {cam['distortion_type']}")
        print(f"Distortion Coefficients: {cam['distortion_coefficients']}")
    print(f"Image Dimension: {cam['image_dimension']}")
    print(f"Fps: {cam['fps']}")

    config["Dataset"]["Calibration"]["fx"] = cam['focal_length'][0]
    config["Dataset"]["Calibration"]["fy"] = cam['focal_length'][1]
    config["Dataset"]["Calibration"]["cx"] = cam['principal_point'][0]
    config["Dataset"]["Calibration"]["cy"] = cam['principal_point'][1]
    has_dist = ('distortion_type' in cam) and ('distortion_coefficients' in cam)
    config["Dataset"]["Calibration"]["distorted"] = has_dist
    if has_dist:
        dist = cam['distortion_coefficients']
        config["Dataset"]["Calibration"]["k1"] = dist[0]
        config["Dataset"]["Calibration"]["k2"] = dist[1]
        config["Dataset"]["Calibration"]["p1"] = dist[2]
        config["Dataset"]["Calibration"]["p2"] = dist[3]
        if len(dist) > 4:
            config["Dataset"]["Calibration"]["k3"] = dist[4]
        else:
            config["Dataset"]["Calibration"]["k3"] = 0.0
    config["Dataset"]["Calibration"]["width"] = cam['image_dimension'][0]
    config["Dataset"]["Calibration"]["height"] = cam['image_dimension'][1]

    config["Dataset"]["Calibration"]["depth_scale"] = cam["depth_factor"]
    config["Dataset"]["depth_name"] = cam['depth_name']

    Log(f"\tuse_gui={verbose}")
    config["Results"]["use_gui"] = verbose

    slam = SLAM(config, save_dir=exp_folder)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
    sys.exit(0)

if __name__ == "__main__":
    main()