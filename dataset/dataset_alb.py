import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import copy
from pipeline.pipeline import Pipeline
from dataset.finding_ground_truth import RegistrationData, TimeSeriesDataset

def get_mean_std(data_list):
    """
    Calculate the mean and standard deviation of a list.
    """
    arr = np.array(data_list)
    return np.mean(arr), np.std(arr)

def quaternion_to_euler(qw, qx, qy, qz):
    """
    Convert a quaternion (qw, qx, qy, qz) to Euler angles (roll, pitch, yaw) in radians.
    """
    rotation = R.from_quat([qx, qy, qz, qw])
    return rotation.as_euler('xyz', degrees=False)

def quaternion_distance(q1, q2):
    """
    Compute the angular distance between two quaternions.
    """
    dot = np.dot(q1, q2)
    dot = np.clip(dot, -1.0, 1.0)
    return 2 * np.arccos(np.abs(dot))

def convert_poses_to_euler(pose_list):
    """
    Convert a list of poses from quaternion representation to Euler angles.
    Each pose: [x, y, z, qw, qx, qy, qz] -> [x, y, z, roll, pitch, yaw]
    """
    euler_list = []
    for pose in pose_list:
        x, y, z, qw, qx, qy, qz = pose
        roll, pitch, yaw = quaternion_to_euler(qw, qx, qy, qz)
        euler_list.append([x, y, z, roll, pitch, yaw])
    return euler_list

def get_outlier_indices(traj1, traj2, pos_threshold=0.5, quat_threshold=0.2):
    """
    Return the indices where the differences between two trajectories exceed the given thresholds.
    """
    traj1 = np.array(traj1)
    traj2 = np.array(traj2)
    outliers = []
    for i in range(len(traj1)):
        pos_diff = np.linalg.norm(traj1[i][:3] - traj2[i][:3])
        quat_diff = quaternion_distance(traj1[i][3:7], traj2[i][3:7])
        if pos_diff > pos_threshold or quat_diff > quat_threshold:
            outliers.append(i)
    return outliers

def calculate_rmse_trajectory(traj1, traj2):
    """
    Calculate the RMSE for positions and angular errors between two trajectories.
    """
    assert len(traj1) == len(traj2), "Trajectories must have the same length."
    pos_errors = []
    rot_errors = []
    for p1, p2 in zip(traj1, traj2):
        pos1, pos2 = np.array(p1[:3]), np.array(p2[:3])
        quat1, quat2 = np.array(p1[3:]), np.array(p2[3:])
        pos_errors.append(np.linalg.norm(pos1 - pos2))
        dot = np.dot(quat1, quat2)
        dot = np.clip(dot, -1.0, 1.0)
        angle_error = 2 * np.arccos(np.abs(dot))
        rot_errors.append(angle_error)
    pos_rmse = np.sqrt(np.mean(np.square(pos_errors)))
    rot_rmse = np.sqrt(np.mean(np.square(rot_errors)))
    return pos_rmse, rot_rmse

def calculate_rmse_trajectory_rpy(traj1, traj2):
    """
    Calculate the RMSE for positions and roll-pitch-yaw differences between two trajectories.
    """
    assert len(traj1) == len(traj2), "Trajectories must have the same length."
    pos_errors = []
    rpy_errors = []
    for p1, p2 in zip(traj1, traj2):
        pos1, pos2 = np.array(p1[:3]), np.array(p2[:3])
        rpy1, rpy2 = np.array(p1[3:]), np.array(p2[3:])
        pos_errors.append(np.linalg.norm(pos1 - pos2))
        rpy_errors.append(np.linalg.norm(rpy1 - rpy2))
    pos_rmse = np.sqrt(np.mean(np.square(pos_errors)))
    rpy_rmse = np.sqrt(np.mean(np.square(rpy_errors)))
    return pos_rmse, rpy_rmse

def main():
    from config.config import get_general_settings
    args = get_general_settings()
    args.sequence_batch = 200
    args.batch_size = 1
    pipeline_instance = Pipeline(args=args)
    # Here you can build the system model, load data, etc.
    print("Dataset module is ready.")

if __name__ == '__main__':
    main()
