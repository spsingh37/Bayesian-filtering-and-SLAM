## First import necessary modules
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from utils.data import load_data
from utils.vis import visualize_trajectory_map_2d

from slam.ekf import EKFSLAM

## Get the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=int, default=10) #10 #0o3
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

## Load the data
SCENE = args.scene
VIDEO_PATH = f"data/{SCENE:02d}_video_every10frames.avi"
DATA_PATH = f"data/{SCENE:02d}.npz"

t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(DATA_PATH)

time_steps = t.shape[1]
t = t[0, :]
device = args.device

features = torch.from_numpy(features[:, ::2, :]).double().to(device)
K = torch.from_numpy(K).double().to(device)
imu_T_cam = torch.from_numpy(imu_T_cam).double().to(device)
b = torch.from_numpy(b).double().to(device)

slam = EKFSLAM(n_landmarks=features.shape[1], imu_T_cam=imu_T_cam, device=device)

poses = []
cov_traces = []

for i in tqdm(range(1, time_steps)):
    
    u = np.concatenate((linear_velocity[:, i], angular_velocity[:, i]))
    u = torch.from_numpy(u).float().to(slam.device)

    slam.predict(u[None, :], t[i] - t[i-1])
    # slam.update(features[:, :, i], K, b, imu_T_cam)
    poses.append(slam.get_pose(numpy=True))
    
    # if i % 100 == 0:
        
    #     visualize_trajectory_map_2d(np.array(poses).squeeze(1).transpose(1, 2, 0), slam.get_landmarks()[:, ::5], show_ori=False)    
    #     plt.plot(cov_traces)
    #     plt.title(f"Covariance Trace {i}")
    #     plt.show()

    cov = slam.cov.trace().item()
    cov_traces.append(cov)

visualize_trajectory_map_2d(np.array(poses).squeeze(1).transpose(1, 2, 0), slam.get_landmarks()[:, ::5], show_ori=False)    
plt.plot(cov_traces)
plt.title(f"Covariance Trace {i}")
plt.show()