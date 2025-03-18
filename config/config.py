import argparse

def get_general_settings():
    """
    Parse and return the general settings for dataset, training, and KalmanNet.
    """
    parser = argparse.ArgumentParser(
        prog='KalmanNet',
        description='Dataset, training, and network parameters'
    )
    # Dataset settings
    parser.add_argument('--train_size', type=int, default=1000, metavar='TRAIN_SIZE',
                        help='Number of sequences for training')
    parser.add_argument('--cv_size', type=int, default=100, metavar='CV_SIZE',
                        help='Number of sequences for cross-validation')
    parser.add_argument('--test_size', type=int, default=200, metavar='TEST_SIZE',
                        help='Number of sequences for testing')
    parser.add_argument('--sequence_length', type=int, default=100, metavar='SEQ_LENGTH',
                        help='Sequence length')
    parser.add_argument('--test_sequence_length', type=int, default=100, metavar='TEST_SEQ_LENGTH',
                        help='Test sequence length')
    
    # Random length settings
    parser.add_argument('--random_length', type=bool, default=False, metavar='RL',
                        help='If True, use random sequence length')
    parser.add_argument('--max_length', type=int, default=10000, metavar='MAX_LENGTH',
                        help='Maximum sequence length when using random lengths')
    parser.add_argument('--min_length', type=int, default=100, metavar='MIN_LENGTH',
                        help='Minimum sequence length when using random lengths')
    
    # Random initial state settings
    parser.add_argument('--random_init_train', type=bool, default=False, metavar='RI_TRAIN',
                        help='If True, use random initial state for training set')
    parser.add_argument('--random_init_cv', type=bool, default=False, metavar='RI_CV',
                        help='If True, use random initial state for validation set')
    parser.add_argument('--random_init_test', type=bool, default=False, metavar='RI_TEST',
                        help='If True, use random initial state for test set')
    parser.add_argument('--variance', type=float, default=100, metavar='VARIANCE',
                        help='Variance for the random initial state (uniform distribution)')
    parser.add_argument('--distribution', type=str, default='normal', metavar='DISTRIBUTION',
                        help='Distribution for the random initial state (uniform/normal)')
    
    # Training settings
    parser.add_argument('--use_cuda', type=bool, default=True, metavar='CUDA',
                        help='If True, use CUDA')
    parser.add_argument('--steps', type=int, default=1000, metavar='STEPS',
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=150, metavar='BATCH_SIZE',
                        help='Training batch size')
    parser.add_argument('--sequence_batch', type=int, default=200, metavar='SEQ_BATCH',
                        help='Sequence batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='LR',
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='WD',
                        help='Weight decay')
    parser.add_argument('--composition_loss', type=bool, default=False, metavar='LOSS',
                        help='If True, use composition loss')
    parser.add_argument('--alpha', type=float, default=0.3, metavar='ALPHA',
                        help='Alpha value [0,1] for composition loss')
    parser.add_argument('--num_workers', type=int, default=16, metavar='NUM_WORKERS',
                        help='Number of workers for data loading')
    parser.add_argument('--prefetch_factor', type=int, default=16, metavar='PREFETCH_FACTOR',
                        help='Prefetch factor for data loading')
    
    # KalmanNet settings
    parser.add_argument('--input_multiplier', type=int, default=5, metavar='IN_MULT',
                        help='Input dimension multiplier for KalmanNet')
    parser.add_argument('--output_multiplier', type=int, default=40, metavar='OUT_MULT',
                        help='Output dimension multiplier for KalmanNet')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_general_settings()
    print(args)
