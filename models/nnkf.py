import torch
from SystemModel import SystemModel
import torch.nn as nn
import torch.nn.functional as F

class ESKF_Torch(nn.Module):
    """
    Extended State Kalman Filter (ESKF) implemented in PyTorch.
    """
    def __init__(self, system_model: SystemModel, initial_state: torch.Tensor, initial_covariance: torch.Tensor, args):
        # Initialize state, error state, and covariance using system model's setup function.
        self.system_model = system_model
        self.state, self.error_state, self.covariance = self.system_model.initsetup(initial_state, initial_covariance)
        super().__init__()
        self.device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
        self.args = args

    def predict(self, control_vector):
        F_matrix = self.system_model._compute_state_transition_matrix(self._dt)
        B_matrix = self.system_model._compute_control_matrix(self._dt, self.state)
        Q = self.init_Q
        # Propagate the error state using the control input.
        self.error_state_prior = B_matrix @ control_vector
        self.predict_state = self.system_model._state_injection(self._dt, self.state, self.error_state_prior).to(self.device)
        self.covariance = self.system_model._propagate_covariance(F_matrix, Q, self.covariance).to(self.device)

    def update(self, measurement):
        Hx = self.system_model.Hx_fun()
        Xdx = self.system_model.Xdx_fun(self.state)
        H = Hx @ Xdx
        R = self.init_S
        cov = H @ self.covariance @ H.transpose(1, 2) + R
        K = self.covariance @ H.transpose(1, 2) @ torch.inverse(cov)
        innovation = measurement - self.predict_state
        self.error_state = K @ innovation
        I = torch.eye(self.error_state.shape[1]).unsqueeze(0).repeat(self.system_model.n_batch, 1, 1).to(self.device)
        self.covariance = (I - K @ H) @ self.covariance
        self.state = self.system_model._state_injection(self._dt, self.predict_state, self.error_state)
        self.previous_error_state = self.error_state  # fixed spelling
        self.error_state = torch.zeros_like(self.error_state)
        return self.state

    def get_state(self):
        return self.state

    def reset_state(self, init_state, re_error=None):
        self.covariance = torch.diag(torch.tensor([0.001]*3 + [0.2]*3, requires_grad=True)).to(self.device)
        diag_matrix_P = torch.diag(torch.tensor([0.0001]*3 + [0.0002]*3, requires_grad=True)).unsqueeze(0).repeat(self.args.n_batch, 1, 1).to(self.device)
        diag_matrix_R = torch.diag(torch.tensor([0.000001]*3 + [0.0001]*4, requires_grad=True)).unsqueeze(0).repeat(self.args.n_batch, 1, 1).to(self.device)
        T_filter = torch.tensor(0.01)
        self.reset_init_state(init_state, diag_matrix_P, diag_matrix_R, T_filter, self.args)

    def reset_init_state(self, state, system_noise_covariance, measurement_noise_covariance, dt, args):
        self.state = state
        self.NNBuild(system_noise_covariance, measurement_noise_covariance, dt, args)

    def NNBuild(self, system_noise_covariance, measurement_noise_covariance, dt, args):
        self.prior_Sigma = self.covariance
        self.init_Q = system_noise_covariance
        self.init_S = measurement_noise_covariance
        self._dt = dt
        self.batch_size = self.system_model.n_batch
        self.seq_len_input = 1
        self.predict_state = self.state
        self.previous_error_state = self.error_state
        self.m = self.prior_Sigma.shape[1]
        self.n = self.init_S.shape[1]
        self.fnn_output_dim = args.in_mult_KNet

    def forward(self, x):
        # Process input: extract twist (control vector) and measurement.
        twist = x[8:14].unsqueeze(0).permute(2, 1, 0)
        measurement = x[0:7].unsqueeze(0).permute(2, 1, 0)
        self.predict(control_vector=twist)
        state = self.update(measurement)
        return state

# KalmanNetOrigin: variant of KalmanNet using the ESKF approach.
class KalmanNetOrigin(ESKF_Torch):
    def __init__(self, system_model: SystemModel, initial_state: torch.Tensor, initial_covariance: torch.Tensor, args, dt):
        super().__init__(system_model, initial_state, initial_covariance, args)
        diag_matrix_P = torch.diag(torch.tensor([0.001]*3 + [0.002]*3, requires_grad=True)).to(self.device)
        diag_matrix_R = torch.diag(torch.tensor([0.0001]*3 + [0.01, 0.01, 0.01, 0.01], requires_grad=True)).to(self.device)
        self.covariance = torch.diag(torch.tensor([0.001]*3 + [0.002]*3, requires_grad=True)).to(self.device)
        self.NNBuild(diag_matrix_P, diag_matrix_R, dt, args)

    def NNBuild(self, system_noise_covariance, measurement_noise_covariance, dt, args):
        self.device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
        self.prior_Sigma = self.covariance
        self.init_Q = dt * system_noise_covariance
        self.init_S = measurement_noise_covariance
        self._dt = dt
        self.batch_size = args.n_batch
        self.seq_len_input = 1
        self.last_measurement = self.state
        self.previous_error_state = torch.zeros_like(self.error_state)
        self.m = self.prior_Sigma.shape[1]
        self.n = measurement_noise_covariance.shape[0]
        self.fnn_output_dim = args.in_mult_KNet
        self.init_k_gain_network()

    def init_k_gain_network(self):
        # GRU to track Q
        self.d_input_Q = self.m * self.fnn_output_dim
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)
        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * self.fnn_output_dim
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)
        # GRU to track S
        self.d_input_S = self.n ** 2 + 2 * self.n * self.fnn_output_dim
        self.d_hidden_S = self.n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)
        # Fully connected layers
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1),
            nn.Dropout(p=0.5),
            nn.ReLU()
        ).to(self.device)
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * self.fnn_output_dim
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2),
            nn.Dropout(p=0.5)
        ).to(self.device)
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3),
            nn.Dropout(p=0.5),
            nn.ReLU()
        ).to(self.device)
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4),
            nn.ReLU()
        ).to(self.device)
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * self.fnn_output_dim
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5),
            nn.Dropout(p=0.5),
            nn.ReLU()
        ).to(self.device)
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * self.fnn_output_dim
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6),
            nn.Dropout(p=0.5),
            nn.ReLU()
        ).to(self.device)
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * self.fnn_output_dim
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7),
            nn.Dropout(p=0.5),
            nn.ReLU()
        ).to(self.device)

class KalmanNetV2(KalmanNetOrigin):
    def __init__(self, system_model: SystemModel, initial_state: torch.Tensor, initial_covariance: torch.Tensor, args, dt):
        super().__init__(system_model, initial_state, initial_covariance, args, dt)

    def NNBuild(self, system_noise_covariance, measurement_noise_covariance, dt, args):
        self.pos_n = 3
        super().NNBuild(system_noise_covariance, measurement_noise_covariance, dt, args)

    def init_k_gain_network(self):
        # GRU to track Q
        self.d_input_Q = self.m * self.fnn_output_dim
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)
        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * self.fnn_output_dim
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)
        # GRU to track S
        self.d_input_S = self.n ** 2 + 2 * self.n * self.fnn_output_dim
        self.d_hidden_S = self.n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)
        # Fully connected layers
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1),
            nn.Dropout(p=0.5),
            nn.ReLU()
        ).to(self.device)
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.pos_n * self.pos_n
        self.d_hidden_FC2 = self.d_input_FC2 * self.fnn_output_dim
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2)
        ).to(self.device)
        self.d_input_FC21 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC21 = (self.n - self.pos_n) * (self.m - self.pos_n)
        self.d_hidden_FC21 = self.d_input_FC2 * self.fnn_output_dim
        self.FC21 = nn.Sequential(
            nn.Linear(self.d_input_FC21, self.d_hidden_FC21),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC21, self.d_output_FC21)
        ).to(self.device)
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2 + self.d_output_FC21
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3),
            nn.Dropout(p=0.5),
            nn.ReLU()
        ).to(self.device)
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4),
            nn.ReLU()
        ).to(self.device)
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * self.fnn_output_dim
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5),
            nn.Dropout(p=0.5),
            nn.ReLU()
        ).to(self.device)
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * self.fnn_output_dim
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6),
            nn.Dropout(p=0.5),
            nn.ReLU()
        ).to(self.device)
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * self.fnn_output_dim
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7),
            nn.Dropout(p=0.5),
            nn.ReLU()
        ).to(self.device)

class KalmanNet(KalmanNetV2):
    def __init__(self, system_model: SystemModel, initial_state: torch.Tensor, initial_covariance: torch.Tensor, args, dt):
        super().__init__(system_model, initial_state, initial_covariance, args, dt)

    def init_hidden_KNet(self, re_error):
        weight = next(self.parameters()).data
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = self.init_S.reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1).to(self.device)
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1).to(self.device)
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = self.init_Q.reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1).to(self.device)
        if re_error is not None:
            re_error = re_error.permute(2, 0, 1)
            self.h_S = self.h_S * re_error
            self.h_Sigma = self.h_Sigma * re_error

    def reset_state(self, init_state, re_error=None):
        self.state = init_state
        self.init_hidden_KNet(re_error)

    def KNet_step(self, x):
        twist = x[8:14].unsqueeze(0).permute(2, 1, 0)
        F_matrix = self.system_model._compute_state_transition_matrix(self._dt)
        B_matrix = self.system_model._compute_control_matrix(self._dt, self.state)
        measurement = x[0:7].unsqueeze(0).permute(2, 1, 0)
        self.error_state_prior = F_matrix @ self.error_state + B_matrix @ twist
        self.predict_state = self.system_model._state_injection(self._dt, self.state, self.error_state_prior)
        self.step_KGain_est(measurement)
        dy = measurement - self.predict_state
        INOV = torch.bmm(self.KGain, dy)
        if not self.training:
            min_mag = torch.zeros_like(INOV).to(self.device)
            max_mag = torch.tensor([0.005, 0.005, 0.005, 0.1, 0.1, 0.1]).unsqueeze(0).unsqueeze(2).repeat(self.args.n_batch, 1, 1).to(self.device) * 100.0
            sign = INOV.sign()
            INOV = INOV.abs_().clamp_(min_mag, max_mag)
            INOV = INOV * sign
        mask = torch.norm(measurement[:, 3, 0]) > 1.0
        _state = self.state.clone()
        _state[mask, :, :] = measurement[mask, :, :]
        self.state = self.system_model._state_injection(self._dt, _state, INOV)
        self.last_measurement = measurement
        self.previous_error_state = self.error_state
        self.error_state = torch.zeros_like(self.error_state)
        return self.state

    def step_KGain_est(self, state):
        obs_diff = torch.squeeze(state, 2) - torch.squeeze(self.state, 2)
        obs_innov_diff = torch.squeeze(state, 2) - torch.squeeze(self.predict_state, 2)
        fw_evol_diff = torch.squeeze(self.error_state, 2) - torch.squeeze(self.error_state_prior, 2)
        fw_update_diff = torch.squeeze(self.error_state, 2) - torch.squeeze(self.previous_error_state, 2)
        obs_diff = F.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        obs_innov_diff = F.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12)
        fw_evol_diff = F.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        fw_update_diff = F.normalize(fw_update_diff, p=2, dim=1, eps=1e-12)
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))

# Optionally, add an alias so that training scripts referring to KalmanNet use this implementation.
# For example:
# KalmanNet = KalmanNet
