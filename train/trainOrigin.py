import torch
from models.nnkf import KalmanNetOrigin
from models.SystemModel import RobotSensorFusion
from config.config import get_general_settings
from pipeline.pipeline import Pipeline

def main_kalman_net_origin():
    args = get_general_settings()
    pipeline_instance = Pipeline(args=args)
    
    task_model = RobotSensorFusion(state_size=7, error_state_size=6, args=args)
    pipeline_instance.build_setup(task_model, args)
    
    init_state = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])\
                .unsqueeze(0).unsqueeze(2).repeat(args.batch_size, 1, 1).to(task_model.device)
    init_covariance = torch.eye(6).unsqueeze(0).repeat(args.batch_size, 1, 1)
    dt_value = torch.tensor(0.01).to(task_model.device)
    
    KF_model = KalmanNetOrigin(system_model=task_model, initial_state=init_state,
                               initial_covariance=init_covariance, args=args, dt=dt_value)\
                               .to(task_model.device)
    
    pipeline_instance.set_nn_model(KF_model)
    
    pipeline_instance.train_network("best_model.pth")

if __name__ == '__main__':
    main_kalman_net_origin()
