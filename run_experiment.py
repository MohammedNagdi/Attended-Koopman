"""
Advanced Koopman Experiment Runner

This script provides a flexible way to run different types of Koopman operator learning experiments
with various model architectures and training strategies.

USAGE EXAMPLES:

1. Standard AutoEncoder + Koopman experiment:
   python run_experiment.py --experiment_type standard --dataset isolated_repressilator --folder standard_exp --epochs 10 --bottleneck 100 --time_steps 201 --prediction_length 200 --max_time 250 --num_samples 15000 --decoder_loss_weight 1 --lr 1e-3 --unitary_loss_weight 10

2. AFT (Attention-Free Transformer) experiment:
   python run_experiment.py --experiment_type AFT --dataset isolated_repressilator --folder aft_exp --epochs 10 --bottleneck 100 --time_steps 201 --prediction_length 200 --max_time 250 --num_samples 15000 --decoder_loss_weight 1 --lr 1e-3 --unitary_loss_weight 10

3. AFT with Dynamic Re-encoding experiment:
   python run_experiment.py --experiment_type AFT_DR --dataset IRMA --folder AFT_DR_exp --time_steps 401 --prediction_length 40 --max_time 800 --num_samples 3000 --bottleneck 100 --decoder_loss_weight 0.1 --lr 1e-3 --epochs 50 --loss_function mse --unitary_loss_weight 10 --context_length 10 --load_from_checkpoint True

4. GRU:
   python run_experiment.py --experiment_type EncoderDecoder --encoder gru --decoder mlp --encoder_depth 1 --decoder_depth 1 --epochs 2

5. Transformer:
   python run_experiment.py --experiment_type EncoderDecoder --encoder transformer --decoder mlp --encoder_depth 1 --decoder_depth 1 --epochs 2
  

EXPERIMENT TYPES:
- standard: Uses AutoEncoder + Koopman operator with standard Trainer
- AFT: Uses AutoEncoder + Koopman operator + AFT layer with Trainer_AFT
- AFT_DR: Uses AutoEncoder + Koopman operator + AFT layer with Trainer_AFT_DR (Dynamic Re-encoding)

AVAILABLE DATASETS:
- discrete_spectrum: Basic discrete spectrum dataset
- pendulum: Pendulum dynamics
- duffing_oscillator: Duffing oscillator system
- goodwin_oscillator: Goodwin oscillator
- isolated_repressilator: Isolated repressilator
- LotkaVolterra: Lotka-Volterra predator-prey model
- Rossler: Rossler attractor
- FluidFlow: Fluid flow dynamics
- IRMA: IRMA network

KEY PARAMETERS:
- --experiment_type: Type of experiment (standard, AFT, AFT_DR, EncoderDecoder)
- --dataset: Dataset to use
- --folder: Folder name to save results
- --bottleneck: Size of the bottleneck layer (latent dimension)
- --context_length: Context length for AFT models
- --epochs: Number of training epochs
- --lr: Learning rate
- --batch_size: Batch size for training
- --time_steps: Number of time steps in trajectories
- --prediction_length: Prediction horizon length for chunking trajectories
- --max_time: Maximum time value for trajectory generation
- --num_samples: Number of samples per combination or initial conditions
- --decoder_loss_weight: Weight for decoder reconstruction loss
- --unitary_loss_weight: Weight for unitary loss
- --loss_function: Loss function to use (mse, relative_mse)
- --load_from_checkpoint: Load model from checkpoint (True/False)

Results will be saved in the 'experiments/{folder}' directory.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init
import torch.multiprocessing as mp

# Import local modules
from utils.tools import set_seed, get_device
from utils.config_args import get_args
from utils.Viz import *
from data_handler import KoopmanDataHandler
from models.autoencoder import AutoEncoder
from models.koopman_operator import Knet
from models.EncoderDecoder import EncoderDecoder
from models.AFT import AttentionFreeTransformer
from KoopmanTrainer import Trainer
from KoopmanTrainer_AFT import Trainer as Trainer_AFT
from KoopmanTrainer_AFT_DR import Trainer as  Trainer_AFT_DR
from EncoderDecoderTrainer import Trainer as Trainer_EncoderDecoder


try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass  # method already set

# Fix deterministic behavior
torch.use_deterministic_algorithms(False)

#==============================================================================
# Training settings
#==============================================================================

# Add experiment_type argument without modifying config_args.py
import sys
experiment_type = 'standard'  # default value

# Extract experiment_type before calling get_args()
if '--experiment_type' in sys.argv:
    idx = sys.argv.index('--experiment_type')
    if idx + 1 < len(sys.argv):
        experiment_type = sys.argv[idx + 1]
        if experiment_type not in ['standard', 'AFT', 'AFT_DR', 'EncoderDecoder']:
            experiment_type = 'standard'
        # Remove the experiment_type arguments from sys.argv to avoid conflicts
        sys.argv.pop(idx)  # remove --experiment_type
        sys.argv.pop(idx)  # remove the value

args = get_args()


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
set_seed(args.seed)

# Device is cuda else cpu
device = get_device()

#******************************************************************************
# Create folder to save results
#******************************************************************************
experiment_folder = 'experiments'
if not os.path.isdir(experiment_folder):
    os.mkdir(experiment_folder)

folder_path = os.path.join(experiment_folder, args.folder)
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)

#******************************************************************************
# save arguments in json file
#******************************************************************************
args_dict = vars(args)
with open(os.path.join(folder_path, 'args.json'), 'w') as f:
    json.dump(args_dict, f,indent=2)

#==============================================================================
# Load and prepare dataset
#==============================================================================
config = {
    "dataset": args.dataset,
    "data_dir": args.data_dir,
    "train_size": args.train_size,
    "val_size": args.val_size,
    "batch_size": args.batch_size,
    "noise": args.noise,
    "orthogonal_projection": args.orthogonal_projection,
    "num_combinations": args.num_combinations,
    "num_samples": args.num_samples,
    "time_steps": args.time_steps,
    "max_time": args.max_time,
    "normalize": args.normalize,
    "prediction_length": args.prediction_length,
    "stride": args.stride,
    "device": device,
    "theta": args.theta,
    "data_parameters": args.data_parameters,
    "shuffle": args.shuffle
}

data_handler = KoopmanDataHandler(config)
data_handler.load_data()
preprocessed_data = data_handler.preprocess()
dataloaders = data_handler.create_dataloaders()

train_loader = dataloaders["train"]
val_loader = dataloaders["val"]
test_loader = dataloaders["test"]

input_size = preprocessed_data['Xtrain'].shape[-1]
input_length = preprocessed_data['Xtrain'].shape[-2]

#==============================================================================
# Model
#==============================================================================
# Model definition
if experiment_type in ['standard', 'AFT', 'AFT_DR']:
    ae = AutoEncoder(
        input_size=input_size,
        encoded_size=args.bottleneck,
        encoder_hidden_layers=args.encoder_hidden_layers,
        decoder_hidden_layers=args.decoder_hidden_layers,
        network_type=args.network_type,
        batch_norm=False
    )
    knet = Knet(size=args.bottleneck)

    if experiment_type in ['AFT', 'AFT_DR']:
        aft_layer = AttentionFreeTransformer(d_model=args.bottleneck, max_seq_len=args.context_length, device=device)
        model = torch.nn.ModuleDict({'ae': ae, 'knet': knet, 'aft_layer': aft_layer})
    else:
        model = torch.nn.ModuleDict({'ae': ae, 'knet': knet})

elif experiment_type == 'EncoderDecoder':
    model = EncoderDecoder(
        name=f"{args.encoder}_{args.decoder}",
        input_size=input_size,
        input_length=input_length-1, # -1 for output
        hidden_size=args.hidden_size,
        output_size=input_size,
        encoder=args.encoder,
        decoder=args.decoder,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        dropout=args.dropout,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        device=device
    )

model = model.to(device)

#==============================================================================
# Model summary
#==============================================================================
print(f'**** Experiment Type: {experiment_type} ****')
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('Total params: %.2fk' % (sum(p.numel() for p in model.parameters())/1000.0))
print('************')

#==============================================================================
# Model Architecture
#==============================================================================
print('**** Model Architecture ****')
print(model)
print('***************************')

#==============================================================================
# Load from checkpoint
#==============================================================================
if args.load_from_checkpoint:
    checkpoint_path = os.path.join(folder_path, 'model.pkl')
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f'Model loaded from {checkpoint_path}')
    else:
        print(f'No checkpoint found at {checkpoint_path}')

#==============================================================================
# Training 
#==============================================================================
if experiment_type == 'standard':
    trainer = Trainer(model, args, device, do_eval=True, train_loader=train_loader, val_loader=val_loader)
elif experiment_type == 'AFT':
    trainer = Trainer_AFT(model, args, device, do_eval=True, train_loader=train_loader, val_loader=val_loader)
elif experiment_type == 'AFT_DR':
    trainer = Trainer_AFT_DR(model, args, device, do_eval=True, train_loader=train_loader, val_loader=val_loader)
elif experiment_type == 'EncoderDecoder':
    trainer = Trainer_EncoderDecoder(model, args, device, do_eval=True, train_loader=train_loader, val_loader=val_loader)

trainer.train()


#==============================================================================
# Visualization
#==============================================================================
model.eval()
traj = preprocessed_data["Xtest"][8].to(device)
with torch.no_grad():
    reconstructed_traj, predicted_traj = None, None

    # Models supporting reconstruction
    if hasattr(trainer, 'reconstruct_traj'):
        reconstructed_traj = trainer.reconstruct_traj(traj)
        reconstructed_traj = reconstructed_traj.cpu().detach().numpy()
    
    # Models supporting prediction
    if hasattr(trainer, 'predict_traj'):
        predicted_traj = trainer.predict_traj(X0=traj[0], steps=len(traj)-1)
        predicted_traj = predicted_traj.cpu().detach().numpy()

    traj = traj.cpu().detach().numpy()

if traj.shape[1] > 3:
    reconstructed_traj = reconstructed_traj[:, :3] if reconstructed_traj is not None else None
    predicted_traj = predicted_traj[:, :3] if predicted_traj is not None else None
    traj = traj[:, :3] if traj is not None else None
plot_trajectories(ref_trajectory=traj, pred_trajectory=predicted_traj, recon_trajectory=reconstructed_traj, folder_path=folder_path)
plot_trajectory_comparison(ref_trajectory=traj, pred_trajectory=predicted_traj, recon_trajectory=reconstructed_traj, folder_path=folder_path)

# Models supporting reconstruction
if hasattr(trainer, 'reconstruct_traj'):
    violin_plot(model, preprocessed_data["Xtest"], device, folder_path)

#==============================================================================
# print eigenvalues and visualize them
#==============================================================================
# Models supporting eigenvalues
if experiment_type in ['standard', 'AFT', 'AFT_DR']:
    eigenvalues,eigenvectors = torch.linalg.eig(model.knet.net.weight)
    eigenvalues = eigenvalues.cpu().detach().numpy()
    idx = np.argsort(eigenvalues.real)[::-1]
    eigenvalues = eigenvalues[idx]

    print('Eigenvalues:')
    print(eigenvalues)

    plot_eigenvalues_on_unit_circle(eigenvalues, folder_path)

# Use the appropriate test function
trainer.test(test_loader)

print(f"Training and evaluation completed. Results saved in {folder_path}")
