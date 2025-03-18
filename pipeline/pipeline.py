import torch
from models.nnkf import ESKFTorch, KalmanNet
from models.SystemModel import RobotSensorFusion
from dataset.finding_ground_truth import RegistrationData, TimeSeriesDataset
from config.config import get_general_settings
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import copy

class Pipeline:
    """
    Pipeline class for setting up data loaders, models, and managing training.
    """
    def __init__(self, train_indices=list(range(0, 11)), valid_indices=list(range(11, 14)),
                 test_indices=list(range(14, 15)), model_filename="saved_state.csv", args=None):
        self.system_model = None
        self.nn_model = None
        self.args = args
        train_data = RegistrationData(number_list=train_indices, filenames=[model_filename])
        self.train_loader = train_data.generate_dataloader(window_size=args.sequence_batch)
        valid_data = RegistrationData(number_list=valid_indices, filenames=[model_filename])
        self.valid_loader = valid_data.generate_dataloader(window_size=args.sequence_batch)
        test_data = RegistrationData(number_list=test_indices, filenames=[model_filename])
        self.test_loader = test_data.generate_dataloader(window_size=args.sequence_batch, is_test=True)

    def set_system_model(self, model):
        self.system_model = model

    def set_nn_model(self, model):
        self.nn_model = model

    def build_setup(self, system_model, args):
        self.set_system_model(system_model)
        self.args = args
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def train_network(self, checkpoint=None):
        if checkpoint:
            self.nn_model.load_state_dict(torch.load(checkpoint), strict=False)
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        train_data = DataLoader(self.train_loader, batch_size=self.args.batch_size,
                                shuffle=False, num_workers=self.args.num_workers,
                                pin_memory=True, prefetch_factor=self.args.prefetch_factor)
        valid_data = DataLoader(self.valid_loader, batch_size=self.args.batch_size,
                                shuffle=True, num_workers=self.args.num_workers,
                                pin_memory=True, prefetch_factor=self.args.prefetch_factor)
        n_params = sum(p.numel() for p in self.nn_model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {n_params:,}")
        writer = SummaryWriter(f"runs/experiment_{self.args.learning_rate}_{self.args.batch_size}")
        writer.add_hparams({'lr': self.args.learning_rate, 'batch_size': self.args.batch_size,
                            'weight_decay': self.args.weight_decay}, {'model_params': n_params})
        best_loss = float('inf')
        max_norm = 0.1
        for epoch in range(100):
            self.nn_model.train()
            train_loss = 0.0
            for batch_idx, (x_traj, y_traj) in enumerate(train_data):
                self.optimizer.zero_grad()
                x_traj = x_traj.permute(1, 2, 0).to(self.nn_model.device)
                y_traj = y_traj.permute(1, 2, 0).to(self.nn_model.device)
                # Initial state and reprojection error are placeholders in this example
                init_state = x_traj[0, 0:7, :].unsqueeze(0).permute(2, 1, 0)
                repro_error = x_traj[0, 7, :].unsqueeze(0).unsqueeze(0).permute(2, 1, 0)
                # Compute loss over the trajectory (placeholder implementation)
                loss = self.criterion(self.nn_model(x_traj), y_traj)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), max_norm)
                self.optimizer.step()
                train_loss += loss.item()
                writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_data) + batch_idx)
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            avg_train_loss = train_loss / len(train_data)
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            print(f"Epoch {epoch+1} Summary: Training Loss: {avg_train_loss:.6f}")
            # Save model if training loss improves (validation routine omitted for brevity)
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                best_model_wts = copy.deepcopy(self.nn_model.state_dict())
                torch.save(best_model_wts,
                           f"FS_best_model_{self.args.learning_rate}_{self.args.batch_size}_{self.args.weight_decay}_seq{self.args.sequence_batch}_KFNET.pth")
                print(f"New best model saved with loss: {best_loss:.6f}")
        writer.close()
        return best_model_wts
