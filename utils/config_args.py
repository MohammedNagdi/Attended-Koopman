import argparse    

def get_args(): 
    parser = argparse.ArgumentParser(description='Koopman Neural Networks')
    
    # Data configuration
    parser.add_argument('--dataset', type=str, default='discrete_spectrum', help='dataset name (simple, discrete_spectrum, isolated_repressilator, duffing_oscillator, host_aware_repressilator)')
    parser.add_argument('--data_dir', type=str, default='./data', help='directory to load/save datasets')
    parser.add_argument('--train_size', type=float, default=0.8, help='size of the training set')
    parser.add_argument('--val_size', type=float, default=0.1, help='size of the validation set')
    parser.add_argument('--normalize', type=str, default='scale', help='normalization method (standard, scale, minmax, or None)')
    parser.add_argument('--prediction_length', type=int, default=50, help='prediction horizon length for chunking trajectories')
    parser.add_argument('--stride', type=int, default=None, help='stride for sliding window in trajectory chunking')
    parser.add_argument('--noise', type=float, default=0.0, help='noise level to add to data')
    parser.add_argument('--orthogonal_projection', type=bool, default=False, help='orthogonal projection of data to higher dimensions')
    parser.add_argument('--theta', type=float, default=2.4, help='angular displacement parameter')
    parser.add_argument('--num_combinations', type=int, default=1, help='number of combinations for system parameters')
    parser.add_argument('--num_samples', type=int, default=6000, help='number of samples per combination or initial conditions')
    parser.add_argument('--time_steps', type=int, default=51, help='number of time steps in trajectories')
    parser.add_argument('--max_time', type=int, default=1, help='maximum time value for trajectory generation')
    parser.add_argument('--data_parameters', type=float,nargs='+', default=None, help='list of parameters for generating data')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle data before training')

    # Training configuration
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr_update', type=int, nargs='+', default=[30, 90, 120, 180], help='decrease learning rate at these epochs')
    parser.add_argument('--lr_decay', type=float, default=0.8, help='learning rate decay factor')
    parser.add_argument('--gradclip', type=float, default=0.05, help='gradient clipping threshold')
    parser.add_argument('--decoder_loss_weight', type=float, default=1e-2, help='weight for decoder reconstruction loss')
    parser.add_argument('--unitary_loss_weight', type=float, default=1e-1, help='weight for unitary loss')
    parser.add_argument('--early_stopping', type=bool, default=False, help='use early stopping')
    parser.add_argument('--loss_function', type=str, default='mse', help='loss function to use (mse, or relative_mse)')

    # EncoderDecoder configuration
    parser.add_argument('--encoder', type=str, default='transformer', help='encoder type for EncoderDecoder (gru, transformer)')
    parser.add_argument('--decoder', type=str, default='mlp', help='decoder type for EncoderDecoder (mlp)')
    parser.add_argument('--encoder_depth', type=int, default=1, help='number of layers in EncoderDecoder encoder')
    parser.add_argument('--decoder_depth', type=int, default=1, help='number of layers in EncoderDecoder decoder')
    parser.add_argument('--hidden_size', type=int, default=20, help='hidden size for EncoderDecoder model')
    parser.add_argument('--dim_feedforward', type=int, default=40, help='dimension of feedforward network in transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate for EncoderDecoder')
    parser.add_argument('--n_heads', type=int, default=4, help='number of attention heads in transformer')
    
    # Autoencoder configuration
    parser.add_argument('--bottleneck', type=int, default=50, help='size of bottleneck layer in autoencoder')
    parser.add_argument('--encoder_hidden_layers', type=int,nargs="+", default=[100, 100], help='Number of units in each hidden layer of encoder')
    parser.add_argument('--decoder_hidden_layers', type=int,nargs="+", default=None, help='number of hidden layers in decoder')
    parser.add_argument('--network_type', type=str, default='MLP', help='type of network to use: "MLP", "KAN", or "PolyKan"')
    # additional parameter for the AFT 
    parser.add_argument('--context_length', type=int, default=10, help='length of context for AFT')
    # KAN configuration (if using KAN models)
    parser.add_argument('--basis_function', type=str, default='chebyshev', help='basis function type for KANs')
    parser.add_argument('--degree', type=int, default=4, help='degree for polynomial basis')
    parser.add_argument('--spline_knots', type=int, default=5, help='number of spline knots for B-spline basis')
    
    # Output, visualization, and misc configuration
    parser.add_argument('--folder', type=str, default='test', help='directory to save results')
    parser.add_argument('--save', type=bool, default=True, help='save model and results')
    parser.add_argument('--load_from_checkpoint', type=bool, default=False, help='load model from checkpoint')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducibility')
    parser.add_argument('--opt_params', type=str, default='lr,batch', help='parameters to optimize')
    parser.add_argument('--num_trials', type=int, default=10, help='number of trials for optimization')
    
    args = parser.parse_args()
    
        
    return args