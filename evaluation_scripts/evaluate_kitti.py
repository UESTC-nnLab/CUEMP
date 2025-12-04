import sys
sys.path.append('/home/honsen/honsen/VO_SLAM/DROID-SLAM/droid_slam')
from droid import Droid
from pathlib import Path
import numpy as np
import cv2
import evo.main_ape as main_ape
import evo.common_ape_rpe as arp
import numpy as np
import torch
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from tqdm import tqdm
SKIP = 0

def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path(reconstruction_path).mkdir(parents=True, exist_ok=True)
    np.save("{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("{}/images.npy".format(reconstruction_path), images)
    np.save("{}/disps.npy".format(reconstruction_path), disps)
    np.save("{}/poses.npy".format(reconstruction_path), poses)
    np.save("{}/intrinsics.npy".format(reconstruction_path), intrinsics)

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

# From https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

from dpt.models import DPTDepthModel
class DPTModel(torch.nn.Module):
    def __init__(self):
        super(DPTModel, self).__init__()
        dpt = DPTDepthModel(
            path="/home/honsen/honsen/depthEstimation/DPT-main/weights/dpt_hybrid-midas-501f0c75.pt",
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        self.depth_model = dpt.cuda()
        self.depth_model.requires_grad_(False)
        self.depth_model.eval()
    def forward(self, x):
        output_48x64, output = self.depth_model(x)
        s = .7 * torch.quantile(output.float(), .98)
        output = output/s
        return  output_48x64, output
import os
def image_stream(imagedir, sequence, stride):
    """ image generator """
    images_dir = imagedir / "RGB" / sequence
    image_list = sorted((images_dir / "image_2").glob("*.png"))[0::stride]
    # calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy, bl = 718.856,718.856,607.1928, 185.2157, 0.53715
    # fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    for t, imfile in enumerate(image_list):
        # if t>500:
        #     break
        image = cv2.imread(os.path.join(imagedir, imfile))
        
        # image = cv2.undistort(image, K, bl)

        h0, w0, _ = image.shape

        image = cv2.resize(image, (480, 128))
        scalex = 480 / w0
        scaley = 128 / h0
        # image = image[:h1 - h1 % 8, :w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0] = intrinsics[0] * scalex
        intrinsics[2] = intrinsics[2] * scalex
        intrinsics[1] = intrinsics[1] * scaley
        intrinsics[3] = intrinsics[3] * scaley

        # intrinsics[0::2] *= (w1 / w0)
        # intrinsics[1::2] *= (h1 / h0)
        yield t, image[None], intrinsics


@torch.no_grad()
def run(args, sequence):

    dpt = DPTModel()
    droid = Droid(args)
    tstamps = []
   
    for (t, image, intrinsics) in tqdm(image_stream(args.kittidir, sequence, stride=args.stride)):
            droid.track(t, image, dpt=dpt, intrinsics=intrinsics)
            tstamps.append(t)
    tstamps = np.array(tstamps)
    tstamps = tstamps.astype(np.float64)
    traj_est = droid.terminate(image_stream(args.kittidir, sequence, stride=args.stride))
    # save_reconstruction(droid,"/home/honsen/tartan/test/reconstruction/kitti/droid")
    return traj_est,tstamps


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--kittidir', type=Path, default="/home/honsen/tartan/kitti/KITTI")
    parser.add_argument("--image_size", default=[128,480])
    parser.add_argument("--weights", default="demo.pth")
    parser.add_argument("--buffer", type=int, default=2300)
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=1.75)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=2.8)
    parser.add_argument("--frontend_thresh", type=float, default=17.5)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=24.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=2)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    print("\nRunning with config...")
    print(args, "\n")

    torch.manual_seed(1234)

    kitti_scenes = [f"{i:02d}" for i in range(11)]
    ks = [ '00']
    results = {}
    for scene in ks:
        groundtruth = args.kittidir / "dataset" / "poses" / f"{scene}.txt"
        poses_ref = file_interface.read_kitti_poses_file(groundtruth)
        print(f"Evaluating KITTI {scene} with {poses_ref.num_poses // args.stride} poses")

        scene_results = []
        for trial_num in range(args.trials):
            traj_est, timestamps = run(args, scene)

            traj_est = PoseTrajectory3D(
                positions_xyz=traj_est[:,:3],
                orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
                timestamps=timestamps * args.stride)#

            traj_ref = PoseTrajectory3D(
                positions_xyz=poses_ref.positions_xyz,
                orientations_quat_wxyz=poses_ref.orientations_quat_wxyz,
                timestamps=np.arange(poses_ref.num_poses, dtype=np.float64))

            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

            result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
            ate_score = result.stats["rmse"]

            if 0:
                Path("/home/honsen/honsen/VO_SLAM/DROID-SLAM/evaluation_scripts/saved_traj").mkdir(exist_ok=True)
                file_interface.write_tum_trajectory_file(f"/home/honsen/honsen/VO_SLAM/DROID-SLAM/evaluation_scripts/saved_traj/KITTI_{scene}.txt", traj_est)
                # file_interface.write_kitti_poses_file(f"saved_trajectories/{scene}.txt", traj_est) # standard kitti format

            scene_results.append(ate_score)

        results[scene] = np.median(scene_results)
        print(scene, sorted(scene_results))

    xs = []
    for scene in results:
        print(scene, results[scene])
        xs.append(results[scene])

    print("AVG: ", np.mean(xs))


 

    
