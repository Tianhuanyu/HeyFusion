import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MeanShift
import torch
from torch.utils.data import Dataset

# Constants for reprojection indices and parameters
INDEX_FOR_REPROJECTION = 13
INDEX_FOR_REPROJECTION2 = 14
ACCEPT_RATE = 0.5
SAMPLE_STEP = 30
TIME_STEP = 0.01
DISPLAY_CHANNELS = [0, 1, 2]
NAME_LIST = ['Observation', 'Ground Truth', 'KalmanNet', 'ESKF', 'Our Method']

class TimeSeriesDataset(Dataset):
    """
    A dataset for time series data using a sliding window approach.
    """
    def __init__(self, data_input, data_output, window_size, is_test=False):
        self.data_input = data_input
        self.data_output = data_output
        self.window_size = window_size
        self.is_test = is_test
        # In non-test mode, each trajectory produces multiple windows.
        self.indices = [len(traj) - window_size + 1 if not is_test else 1 for traj in self.data_input]
        self.num_trajectories = len(self.data_input)
        self.total_length = sum(self.indices)
        print("Number of windows per trajectory:", self.indices)
        print("Total dataset length:", self.total_length)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        temp_index = index
        for i, count in enumerate(self.indices):
            if temp_index < count:
                traj_idx = i
                break
            else:
                temp_index -= count
        length = self.window_size if not self.is_test else len(self.data_input[traj_idx]) - 1
        x = self.data_input[traj_idx][temp_index:temp_index+length]
        y = self.data_output[traj_idx][temp_index:temp_index+length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class RegistrationData:
    """
    Processes registration data to generate ground truth and related measurements.
    
    The data files are now stored in the "data" folder (at the same level as README.md).
    This "data" folder contains subfolders named from "0" to "18", each storing corresponding CSV files.
    """
    def __init__(self, number_list: list = list(range(1, 6)), filenames: list = ["saved_state.csv"], remove_outliers: bool = False) -> None:
        self.data = {}
        # For each filename, generate full paths using read_paths_from_name
        for filename in filenames:
            path_list = RegistrationData.read_paths_from_name(filename, number_list)
            self.data[filename] = RegistrationData.load_measurements(*path_list)
        # Data augmentation steps (placeholder implementation)
        self.augmented_dst, self.pq_dst, _ = self._augment_data(self.data, [0, 3], [3, 7], reverse=True)
        self.augmented_src, self.pq_src, self.reprojection_error = self._augment_data(self.data, [15, 18], [18, 22])
        self.twist = self._select_data(self.data, [7, 13])
        
        # Use the first filename as default for further processing
        target_file = filenames[0]
        self.pq_dst = self.pq_dst[target_file]
        self.pq_src = self.pq_src[target_file]
        self.reprojection_error = self.reprojection_error[target_file]
        self.twist = self.twist[target_file]
        
        self.calibrated_result = self._register(self.augmented_dst[target_file], self.augmented_src[target_file])
        self.ground_truth = self._get_ground_truth()
        self.twist_from_sensor = self.get_pose_difference_wrt_ee(self.pq_src)
        
        if remove_outliers:
            self.pq_dst = self.remove_outliers_with_reference(self.pq_dst, self.ground_truth)
        
        self.hybrid_calibration = self._get_transformation()
        if self.reprojection_error and len(self.reprojection_error[0]) > 0:
            print("Max reprojection error:", max(self.reprojection_error[0]))
            print("Min reprojection error:", min(self.reprojection_error[0]))
        else:
            print("Reprojection error data is empty.")

    @staticmethod
    def read_paths_from_name(filename, number_list):
        """
        Generate the full paths for data files based on the filename and folder numbers.
        The data files are stored in the "data" folder (at the same level as README.md),
        which contains subfolders named "0" to "18".
        """
        base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        paths = []
        for num in number_list:
            folder_name = str(num)
            file_path = os.path.join(base_path, folder_name, filename)
            paths.append(file_path)
        return paths

    @staticmethod
    def load_measurements(*paths):
        """
        Load data from multiple CSV files.
        A simple implementation: read each CSV file and convert rows to float values.
        """
        measurements = []
        for path in paths:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                file_data = [list(map(float, row)) for row in reader]
                measurements.append(file_data)
        return measurements

    def _augment_data(self, data_dict, indices1, indices2, reverse=False):
        """
        Augment data. (Placeholder implementation)
        Returns augmented data, selected data, and None.
        """
        augmented = {}
        selected = {}
        for key in data_dict.keys():
            augmented[key] = data_dict[key]
            selected[key] = data_dict[key]
        return augmented, selected, None

    def _select_data(self, data_dict, indices):
        """
        Select data based on provided indices. (Placeholder implementation)
        """
        selected = {}
        for key in data_dict.keys():
            selected[key] = data_dict[key]
        return selected

    def _register(self, data_dst, data_src):
        """
        Perform registration on the data. (Placeholder implementation)
        """
        return data_dst

    def _get_ground_truth(self):
        """
        Generate ground truth data. (Placeholder implementation)
        """
        return self.pq_src

    def get_pose_difference_wrt_ee(self, pose_list):
        """
        Calculate the pose difference with respect to the end-effector.
        (Placeholder implementation)
        """
        output = []
        for traj in pose_list:
            diff_list = [[0.0] * 6]
            for i in range(1, len(traj)):
                pos_diff = (np.asarray(traj[i][:3]) - np.asarray(traj[i - 1][:3])) / TIME_STEP
                ang_diff = np.abs(np.asarray(traj[i][3:7]) - np.asarray(traj[i - 1][3:7]))
                diff_list.append(pos_diff.tolist() + ang_diff.tolist())
            output.append(diff_list)
        return output

    def remove_outliers_with_reference(self, pq, ground_truth):
        """
        Remove outliers based on reference ground truth.
        (Placeholder implementation)
        """
        return pq

    def _get_transformation(self):
        """
        Compute the hybrid calibration transformation.
        (Placeholder implementation)
        """
        return None

    def generate_io(self):
        """
        Generate input and output trajectories for the time series dataset.
        """
        inputs = []
        outputs = []
        for dst_traj, err_traj, twist_traj, gt_traj in zip(self.pq_dst, self.reprojection_error, self.twist_from_sensor, self.ground_truth):
            input_traj = []
            output_traj = []
            for dst, err, twist, gt in zip(dst_traj, err_traj, twist_traj, gt_traj):
                input_traj.append(dst + [err] + twist)
                output_traj.append(gt)
            inputs.append(input_traj)
            outputs.append(output_traj)
        return inputs, outputs

    def generate_dataloader(self, window_size: int = 20, is_test: bool = False) -> TimeSeriesDataset:
        inputs, outputs = self.generate_io()
        return TimeSeriesDataset(inputs, outputs, window_size=window_size, is_test=is_test)

if __name__ == '__main__':
    rd = RegistrationData()
    dataset_instance = rd.generate_dataloader(window_size=20)
    print("Dataset length:", len(dataset_instance))
    # Uncomment below to display the reprojection error distribution
    # plt.plot(rd.reprojection_error[0])
    # plt.show()
