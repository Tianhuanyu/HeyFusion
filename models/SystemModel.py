import torch
from abc import ABC, abstractmethod

class SystemModel(ABC):
    """
    Abstract system model defining the interface for state propagation, measurement functions, etc.
    """
    def __init__(self, state_size: int, error_state_size: int, args):
        self.state_size = state_size
        self.error_state_size = error_state_size
        self.n_batch = args.batch_size

    def initsetup(self, state=None, cov=None):
        error_state = torch.zeros(self.n_batch, self.error_state_size, 1)
        state_tensor = torch.zeros(self.n_batch, self.state_size, 1)
        covariance = torch.eye(self.error_state_size).unsqueeze(0).repeat(self.n_batch, 1, 1)
        if state is not None and cov is not None:
            if state.size() == state_tensor.size() and cov.size() == covariance.size():
                state_tensor = state
                covariance = cov
            else:
                raise ValueError("Incorrect tensor sizes for state or covariance")
        return state_tensor, error_state, covariance

    @abstractmethod
    def compute_state_transition_matrix(self, dt):
        pass

    @abstractmethod
    def propagate_covariance(self, F, Q, covariance):
        pass

    @abstractmethod
    def Hx_function(self):
        pass

    @abstractmethod
    def Xdx_function(self, state):
        pass

    @abstractmethod
    def state_injection(self, dt, state, error_state):
        pass

    @abstractmethod
    def compute_control_matrix(self, dt, state):
        pass

class RobotSensorFusion(SystemModel):
    """
    Concrete system model for robot sensor fusion.
    """
    def __init__(self, state_size: int, error_state_size: int, args):
        super(RobotSensorFusion, self).__init__(state_size, error_state_size, args)
        self.device = torch.device('cuda') if args.use_cuda else torch.device('cpu')

    def quaternion_normalize(self, quaternions):
        normalized = torch.zeros_like(quaternions)
        for i, q in enumerate(quaternions):
            w, x, y, z = q[0][0], q[1][0], q[2][0], q[3][0]
            norm = w*w + x*x + y*y + z*z
            normalized[i, :, :] = torch.tensor([w/norm, x/norm, y/norm, z/norm],
                                                 requires_grad=True).unsqueeze(1)
        return normalized.to(self.device)

    def quaternion_to_left_matrix(self, quaternions):
        Q_left = torch.zeros([quaternions.shape[0], quaternions.shape[1], 3])
        for i, q in enumerate(quaternions):
            w, x, y, z = q[0], q[1], q[2], q[3]
            Q_left[i, :, :] = torch.tensor(0.5) * torch.tensor([
                [-x, -y, -z],
                [w, -z, y],
                [z, w, -x],
                [-y, x, w]
            ], requires_grad=True)
        return Q_left.to(self.device)

    def initsetup(self, initial_state: torch.Tensor, initial_covariance: torch.Tensor):
        initial_state = initial_state.to(self.device)
        initial_covariance = initial_covariance.to(self.device)
        state, error_state, covariance = super().initsetup(initial_state, initial_covariance)
        state[:, self.state_size - 4:self.state_size, :] = self.quaternion_normalize(
            state[:, self.state_size - 4:self.state_size, :]
        )
        return state.to(self.device), error_state.to(self.device), covariance.to(self.device)

    def compute_state_transition_matrix(self, dt):
        l = self.error_state_size // 2
        eye = torch.eye(l).to(self.device)
        zeros = torch.zeros(l, l).to(self.device)
        F_top = torch.cat([eye, zeros], dim=1)
        F_bottom = torch.cat([zeros, eye], dim=1)
        F = torch.cat([F_top, F_bottom], dim=0).unsqueeze(0).repeat(self.n_batch, 1, 1)
        return F.to(self.device)

    def propagate_covariance(self, F, Q, covariance):
        covariance = F @ covariance @ F.transpose(1, 2) + Q
        return covariance.to(self.device)

    def compute_control_matrix(self, dt, state):
        R_matrix = self.quaternion_to_rotation_matrix(state[:, 3:, :]).to(self.device)
        S = torch.eye(6).to(self.device).repeat(self.n_batch, 1, 1)
        S[:, :3, :3] = R_matrix
        S[:, 3:7, 3:7] = R_matrix
        return S.to(self.device) * dt

    def Hx_function(self):
        Hx = torch.eye(self.state_size).unsqueeze(0).repeat(self.n_batch, 1, 1)
        return Hx.to(self.device)

    def Xdx_function(self, state):
        q_rt = state[:, 3:7, :]
        Q_left = self.quaternion_to_left_matrix(q_rt)
        Xdx = torch.zeros(self.n_batch, 7, 6)
        for i in range(self.n_batch):
            Xdx_1 = torch.cat([torch.eye(3), torch.zeros((4, 3))], dim=0).to(self.device)
            Xdx_2 = torch.cat([torch.zeros((3, 3)).to(self.device), Q_left[i, :, :]], dim=0)
            Xdx[i, :, :] = torch.cat([Xdx_1, Xdx_2], dim=1)
        return Xdx.to(self.device)

    def state_injection(self, dt, state, error_state):
        # A placeholder implementation of state injection: simple addition.
        return state + error_state

    def quaternion_to_rotation_matrix(self, quaternions):
        # Placeholder
        batch_size = quaternions.shape[0]
        return torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)

    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1[:, 0, :], q1[:, 1, :], q1[:, 2, :], q1[:, 3, :]
        w2, x2, y2, z2 = q2[:, 0, :], q2[:, 1, :], q2[:, 2, :], q2[:, 3, :]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=1)
