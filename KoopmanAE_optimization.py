import json
import os
import optuna
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import joblib
import sys
import pandas as pd
from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.tools import set_seed, get_device
from utils.config_args import get_args
from data_handler import KoopmanDataHandler
from models.autoencoder import AutoEncoder
from models.koopman_operator import Knet
from models.AFT import AttentionFreeTransformer
from models.EncoderDecoder import EncoderDecoder
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
args.opt_params = args.opt_params.split(',')
args.save = False

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

# Create analysis subdirectory for all visualizations
analysis_folder = os.path.join(folder_path, 'analysis')
if not os.path.isdir(analysis_folder):
    os.mkdir(analysis_folder)

# Save configuration in json file
with open(os.path.join(folder_path, 'config.json'), 'w') as f:
    json.dump(vars(args), f, indent=2)

#================================================================================
# Parameter ranges for optimization
#================================================================================
batch_range = [32, 64, 128, 256, 512]
bottleneck_range = [10, 20, 30, 40, 50, 60]

#==============================================================================
# Load and prepare dataset outside the objective function
#==============================================================================
# Initialize data handler with initial config
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
    "data_parameters": args.data_parameters
}

# Initialize data handler
data_handler = KoopmanDataHandler(config)
data_handler.load_data()
preprocessed_data = data_handler.preprocess()
dataloaders = data_handler.create_dataloaders()

# Store preprocessed data for reuse
train_data = preprocessed_data['Xtrain']
val_data = preprocessed_data['Xval']
test_data = preprocessed_data['Xtest']

input_size = preprocessed_data['Xtrain'].shape[-1]
input_length = preprocessed_data['Xtrain'].shape[-2]

#================================================================================
# Objective function
#================================================================================
def objective(trial):
    # Update parameters based on trial suggestions
    for opt_param in args.opt_params:
        if opt_param == 'lr':
            args.lr = trial.suggest_categorical('lr', [1e-3,5e-4, 1e-4])
        elif opt_param == 'batch':
            args.batch_size = trial.suggest_categorical('batch_size', batch_range)
        elif opt_param == 'degree':
            args.degree = trial.suggest_int('degree', 2, 10)
        elif opt_param == 'spline_knots':
            args.spline_knots = trial.suggest_int('spline_knots', 2, 12)
        elif opt_param == 'bottleneck':
            args.bottleneck = trial.suggest_int('bottleneck', 10, 150, step=10)
        elif opt_param == 'decoder_loss_weight':
            # suggest integer values from 0.1 to 10
            args.decoder_loss_weight = trial.suggest_float('decoder_loss_weight', 0.01, 10.0, step=0.1)
        elif opt_param == 'unitarity_loss_weight':
            args.unitary_loss_weight = trial.suggest_float('unitarity_loss_weight', 0.01, 10.0, step=0.1)
        elif opt_param == 'encoder_layers':
            # Number of units in each hidden layer
            size = trial.suggest_int('encoder_size', 50, 200, step=50)
            depth = trial.suggest_int('encoder_depth', 1, 4)
            args.encoder_hidden_layers = [size] * depth
        elif opt_param == 'network_type':
            args.network_type = trial.suggest_categorical('network_type', ['MLP', 'KAN', 'PolyKAN'])
        elif opt_param == 'context_length':
            args.context_length = trial.suggest_categorical('context_length', [5,10,15,20])
        else:
            raise ValueError(f'Unknown optimization parameter: {opt_param}')
    
    # Update batch size in config and create dataloaders with the new batch size
    data_handler.config['batch_size'] = args.batch_size
    dataloaders = data_handler.create_dataloaders()
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

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

    if experiment_type == 'standard':
        trainer = Trainer(model, args, device, do_eval=True, train_loader=train_loader, val_loader=val_loader)
    elif experiment_type == 'AFT':
        trainer = Trainer_AFT(model, args, device, do_eval=True, train_loader=train_loader, val_loader=val_loader)
    elif experiment_type == 'AFT_DR':
        trainer = Trainer_AFT_DR(model, args, device, do_eval=True, train_loader=train_loader, val_loader=val_loader)
    elif experiment_type == 'EncoderDecoder':
        trainer = Trainer_EncoderDecoder(model, args, device, do_eval=True, train_loader=train_loader, val_loader=val_loader)

    
    # Only train for a smaller number of epochs for optimization trials
    original_epochs = args.epochs
    args.epochs = max(10, int(args.epochs * 0.2))
    trainer.num_epochs = args.epochs
    
    # Train the model
    stats = trainer.train()[2]
    
    # Restore original epochs
    args.epochs = original_epochs
    
    # Evaluate the model - use validation loss as optimization criterion
    reconstruction_losses = stats['recon_loss_va']
    prediction_losses = stats['pred_loss_va']
    linear_losses = stats['lin_loss_va']
    total_losses = stats['total_loss_va']

    # save the losses in study 
    trial.set_user_attr("reconstruction_losses", reconstruction_losses)
    trial.set_user_attr("prediction_losses", prediction_losses)
    trial.set_user_attr("linear_losses", linear_losses)
    trial.set_user_attr("total_losses", total_losses)

    reconstruction_loss = reconstruction_losses[-1]  # Last value of reconstruction loss
    prediction_loss = prediction_losses[-1]  # Last value of prediction loss
    
    return reconstruction_loss, prediction_loss

#================================================================================
# Enhanced visualization functions
#================================================================================
def extract_trial_metrics(study):
    """Extract all metrics from each trial"""
    metrics_by_trial = {}
    
    for trial in study.trials:
        if trial.state.is_finished():
            trial_id = trial.number
            metrics_by_trial[trial_id] = {
                'reconstruction_losses': trial.user_attrs.get('reconstruction_losses', []),
                'prediction_losses': trial.user_attrs.get('prediction_losses', []),
                'linear_losses': trial.user_attrs.get('linear_losses', []),
                'total_losses': trial.user_attrs.get('total_losses', []),
                'params': trial.params,
                'values': trial.values,
            }
    
    return metrics_by_trial

def plot_learning_curves(metrics_by_trial, metric_name, output_dir):
    """Plot learning curves for a specific metric across all trials"""
    plt.figure(figsize=(12, 8))
    
    # Find the best trials
    best_recon_trial = min(metrics_by_trial.keys(), 
                          key=lambda tid: metrics_by_trial[tid]['values'][0])
    best_pred_trial = min(metrics_by_trial.keys(), 
                         key=lambda tid: metrics_by_trial[tid]['values'][1])
    
    # Plot each trial's learning curve
    for trial_id, metrics in metrics_by_trial.items():
        values = metrics[metric_name]
        if not values:  # Skip if no values
            continue
            
        epochs = list(range(1, len(values) + 1))
        
        # Highlight best trials
        if trial_id == best_recon_trial:
            plt.plot(epochs, values, 'b-', linewidth=2, 
                     label=f'Best Reconstruction (Trial {trial_id})')
        elif trial_id == best_pred_trial:
            plt.plot(epochs, values, 'r-', linewidth=2,
                     label=f'Best Prediction (Trial {trial_id})')
        else:
            plt.plot(epochs, values, 'gray', alpha=0.3)
    
    plt.title(f'Learning Curves: {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.yscale('log')  # Log scale often helps visualize loss curves
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'{metric_name}_learning_curves.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {metric_name} learning curves to {output_path}")

def plot_parallel_coordinate(study, output_dir):
    """Create parallel coordinate plots for the study"""
    try:
        # Create two separate plots for reconstruction and prediction loss
        for target_idx, target_name in enumerate(['Reconstruction Loss', 'Prediction Loss']):
            fig = go.Figure(data=
                go.Parcoords(
                    line=dict(
                        color=study.trials_dataframe()['values_' + str(target_idx)],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=target_name)
                    ),
                    dimensions=[
                        # First add all the parameters
                        *[dict(
                            range=[study.trials_dataframe()[col].min(), study.trials_dataframe()[col].max()],
                            label=col,
                            values=study.trials_dataframe()[col]
                        ) for col in study.trials_dataframe().columns if col.startswith('params_')],
                        # Then add the target value
                        dict(
                            range=[study.trials_dataframe()['values_' + str(target_idx)].min(), 
                                  study.trials_dataframe()['values_' + str(target_idx)].max()],
                            label=target_name,
                            values=study.trials_dataframe()['values_' + str(target_idx)]
                        )
                    ]
                )
            )
            
            fig.update_layout(
                title=f"Parallel Coordinates Plot - {target_name}",
                width=1200,
                height=800,
            )
            
            # Save the figure
            output_path = os.path.join(output_dir, f'parallel_coordinates_{target_idx}.html')
            fig.write_html(output_path)
            
            # Also save as PNG for easier viewing
            output_path_png = os.path.join(output_dir, f'parallel_coordinates_{target_idx}.png')
            fig.write_image(output_path_png, scale=2)
            
            print(f"Saved parallel coordinates plot for {target_name} to {output_path}")
        
        # Combined plot with both objectives
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    # Create a color based on the sum of normalized values
                    color=study.trials_dataframe()['values_0'] + study.trials_dataframe()['values_1'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Combined Loss")
                ),
                dimensions=[
                    # First add all the parameters
                    *[dict(
                        range=[study.trials_dataframe()[col].min(), study.trials_dataframe()[col].max()],
                        label=col.replace('params_', ''),
                        values=study.trials_dataframe()[col]
                    ) for col in study.trials_dataframe().columns if col.startswith('params_')],
                    # Then add both target values
                    dict(
                        range=[study.trials_dataframe()['values_0'].min(), 
                              study.trials_dataframe()['values_0'].max()],
                        label='Reconstruction Loss',
                        values=study.trials_dataframe()['values_0']
                    ),
                    dict(
                        range=[study.trials_dataframe()['values_1'].min(), 
                              study.trials_dataframe()['values_1'].max()],
                        label='Prediction Loss',
                        values=study.trials_dataframe()['values_1']
                    )
                ]
            )
        )
        
        fig.update_layout(
            title="Parallel Coordinates Plot - Combined",
            width=1200,
            height=800,
        )
        
        # Save the combined figure
        output_path = os.path.join(output_dir, 'parallel_coordinates_combined.html')
        fig.write_html(output_path)
        
        # Also save as PNG for easier viewing
        output_path_png = os.path.join(output_dir, 'parallel_coordinates_combined.png')
        fig.write_image(output_path_png, scale=2)
        
        print(f"Saved combined parallel coordinates plot to {output_path}")
        
    except Exception as e:
        print(f"Error creating parallel coordinate plot: {e}")
        print("Make sure you have plotly and kaleido installed:")
        print("pip install plotly kaleido")

def plot_final_values_comparison(metrics_by_trial, output_dir):
    """Plot comparison of final values for each metric across trials"""
    # Extract final values for each trial
    trial_ids = []
    recon_finals = []
    pred_finals = []
    linear_finals = []
    total_finals = []
    
    for trial_id, metrics in metrics_by_trial.items():
        recon = metrics['reconstruction_losses']
        pred = metrics['prediction_losses']
        linear = metrics['linear_losses']
        total = metrics['total_losses']
        
        if all([recon, pred, linear, total]):  # Make sure all metrics exist
            trial_ids.append(trial_id)
            recon_finals.append(recon[-1])
            pred_finals.append(pred[-1])
            linear_finals.append(linear[-1])
            total_finals.append(total[-1])
    
    # Create a dataframe for easier plotting
    df = pd.DataFrame({
        'Trial': trial_ids,
        'Reconstruction Loss': recon_finals,
        'Prediction Loss': pred_finals,
        'Linear Loss': linear_finals,
        'Total Loss': total_finals
    }).sort_values('Trial')
    
    # Plot the comparison
    plt.figure(figsize=(14, 10))
    df.plot(x='Trial', y=['Reconstruction Loss', 'Prediction Loss', 
                          'Linear Loss', 'Total Loss'], 
            kind='bar', width=0.8, figsize=(14, 10))
    plt.title('Final Loss Values Across Trials')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'final_values_comparison.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved final values comparison to {output_path}")
    
    # Return top 3 trials for each metric
    top_indices = {}
    for metric, values in [('reconstruction', recon_finals), 
                          ('prediction', pred_finals),
                          ('linear', linear_finals),
                          ('total', total_finals)]:
        top_indices[metric] = np.argsort(values)[:3]
    
    top_trials = {}
    for metric, indices in top_indices.items():
        top_trials[metric] = [trial_ids[i] for i in indices]
    
    return top_trials

def create_optuna_visualizations(study, output_dir):
    """Create and save standard Optuna visualizations"""
    try:
        # Parameter importance for reconstruction error
        fig1_1 = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[0], target_name="Reconstruction Error"
        )
        fig1_1.write_image(os.path.join(output_dir, 'param_importances_reconstruction.png'))
        
        # Parameter importance for prediction error
        fig1_2 = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[1], target_name="Prediction Error"
        )
        fig1_2.write_image(os.path.join(output_dir, 'param_importances_prediction.png'))
        
        # Slice plot for prediction error
        fig2 = optuna.visualization.plot_slice(
            study, target=lambda t: t.values[1], target_name="Prediction Error"
        )
        fig2.write_image(os.path.join(output_dir, 'slice_plot.png'))
        
        # Pareto front visualization
        fig3 = optuna.visualization.plot_pareto_front(
            study, target_names=["Reconstruction Error", "Prediction Error"]
        )
        fig3.write_image(os.path.join(output_dir, 'pareto_front.png'))
        
        # Save HTML versions for interactive viewing
        fig1_1.write_html(os.path.join(output_dir, 'param_importances_reconstruction.html'))
        fig1_2.write_html(os.path.join(output_dir, 'param_importances_prediction.html'))
        fig2.write_html(os.path.join(output_dir, 'slice_plot.html'))
        fig3.write_html(os.path.join(output_dir, 'pareto_front.html'))
        
        # Contour plot if applicable
        if len(study.trials[0].params) >= 2:
            try:
                fig4 = optuna.visualization.plot_contour(
                    study, target=lambda t: t.values[1], target_name="Prediction Error"
                )
                fig4.write_image(os.path.join(output_dir, 'contour_plot.png'))
                fig4.write_html(os.path.join(output_dir, 'contour_plot.html'))
            except Exception as e:
                print(f"Could not create contour plot: {e}")
                
    except ImportError:
        print("Could not create visualizations - please install plotly and kaleido for visualization support.")
    except Exception as e:
        print(f"Error in creating Optuna visualizations: {e}")

def run_enhanced_visualization(study, folder_path):
    """Run all enhanced visualization functions"""
    analysis_dir = os.path.join(folder_path, 'analysis')
    
    # Create Optuna's standard visualizations in the analysis folder
    print("Creating standard Optuna visualizations...")
    create_optuna_visualizations(study, analysis_dir)
    
    # Extract all metrics
    metrics_by_trial = extract_trial_metrics(study)
    print(f"Extracted metrics from {len(metrics_by_trial)} completed trials for enhanced visualization")
    
    # Plot learning curves for each metric
    for metric in ['reconstruction_losses', 'prediction_losses', 
                  'linear_losses', 'total_losses']:
        plot_learning_curves(metrics_by_trial, metric, analysis_dir)
    
    # Plot parallel coordinates
    print("Creating parallel coordinates plot...")
    plot_parallel_coordinate(study, analysis_dir)
    
    # Plot comparison of final values
    top_trials = plot_final_values_comparison(metrics_by_trial, analysis_dir)
    
    # Print best trials for each metric
    print("\nTop 3 trials for each metric:")
    for metric, trials in top_trials.items():
        print(f"  {metric.capitalize()} loss: Trials {trials}")
        
    # Print parameters of best trials
    print("\nParameters of best trial for reconstruction:")
    best_recon_id = top_trials['reconstruction'][0]
    print(f"  Trial {best_recon_id}: {metrics_by_trial[best_recon_id]['params']}")
    
    print("\nParameters of best trial for prediction:")
    best_pred_id = top_trials['prediction'][0]
    print(f"  Trial {best_pred_id}: {metrics_by_trial[best_pred_id]['params']}")
    
    # Save top trials analysis to file
    with open(os.path.join(analysis_dir, 'top_trials_analysis.json'), 'w') as f:
        top_trials_data = {
            'reconstruction': {
                'top_trials': top_trials['reconstruction'],
                'parameters': [metrics_by_trial[tid]['params'] for tid in top_trials['reconstruction']]
            },
            'prediction': {
                'top_trials': top_trials['prediction'],
                'parameters': [metrics_by_trial[tid]['params'] for tid in top_trials['prediction']]
            }
        }
        json.dump(top_trials_data, f, indent=2)
    
    return top_trials

#================================================================================
# Run optimization
#================================================================================
print("Starting hyperparameter optimization...")
study = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=optuna.samplers.TPESampler(seed=args.seed),
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=args.num_trials, n_jobs=1)

#================================================================================
# Save and visualize results
#================================================================================
print("Study statistics: ")
print(f"  Number of finished trials: {len(study.trials)}")

print("Best trials:")
pareto_trials = study.best_trials

best_trial_reconstruction = min(pareto_trials, key=lambda trial: trial.values[0])
best_trial_prediction = min(pareto_trials, key=lambda trial: trial.values[1])

print(f'Best trial for reconstruction error: {best_trial_reconstruction.number}')
print(f'  Value: {best_trial_reconstruction.values[0]}')
print(f'  Params: {best_trial_reconstruction.params}')

print(f'Best trial for prediction error: {best_trial_prediction.number}')
print(f'  Value: {best_trial_prediction.values[1]}')
print(f'  Params: {best_trial_prediction.params}')

# Save the best trial parameters to a JSON file
with open(os.path.join(folder_path, 'best_params_reconstruction.json'), 'w') as f:
    json.dump(best_trial_reconstruction.params, f, indent=2)
    
with open(os.path.join(folder_path, 'best_params_prediction.json'), 'w') as f:
    json.dump(best_trial_prediction.params, f, indent=2)

# Save the study
joblib.dump(study, os.path.join(folder_path, 'study.pkl'))

# Run all visualizations (standard and enhanced)
print("\nGenerating visualizations...")
top_trials = run_enhanced_visualization(study, folder_path)

print(f"Optimization and visualization completed. Results saved in {folder_path}")

# Optional: Train the model with the best parameters for prediction
print("Training model with best parameters for prediction...")

# Update args with best parameters
for param, value in best_trial_prediction.params.items():
    setattr(args, param, value)

# Create dataloaders with the best batch size
dataloaders = data_handler.create_dataloaders()
train_loader = dataloaders["train"]
val_loader = dataloaders["val"]
test_loader = dataloaders["test"]

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

best_model = model.to(device)
if experiment_type == 'standard':
    trainer = Trainer(best_model, args, device, do_eval=True, train_loader=train_loader, val_loader=val_loader)
elif experiment_type == 'AFT':
    trainer = Trainer_AFT(best_model, args, device, do_eval=True, train_loader=train_loader, val_loader=val_loader)
elif experiment_type == 'AFT_DR':
    trainer = Trainer_AFT_DR(best_model, args, device, do_eval=True, train_loader=train_loader, val_loader=val_loader)
elif experiment_type == 'EncoderDecoder':
    trainer = Trainer_EncoderDecoder(best_model, args, device, do_eval=True, train_loader=train_loader, val_loader=val_loader)

# Train the model
best_model_stats = trainer.train()[2]

# Save the best model
if args.save:
    torch.save(best_model.state_dict(), os.path.join(folder_path, 'best_model.pkl'))
    print(f"Best model saved to {os.path.join(folder_path, 'best_model.pkl')}")

# Create final learning curve plot for the best model
plt.figure(figsize=(12, 8))
epochs = range(1, len(best_model_stats['recon_loss_tr']) + 1)

plt.subplot(2, 2, 1)
plt.plot(epochs, best_model_stats['recon_loss_tr'], 'b-', label='Train')
plt.plot(epochs, best_model_stats['recon_loss_va'], 'r-', label='Validation')
plt.title('Reconstruction Loss')
plt.yscale('log')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, best_model_stats['pred_loss_tr'], 'b-', label='Train')
plt.plot(epochs, best_model_stats['pred_loss_va'], 'r-', label='Validation')
plt.title('Prediction Loss')
plt.yscale('log')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs, best_model_stats['lin_loss_tr'], 'b-', label='Train')
plt.plot(epochs, best_model_stats['lin_loss_va'], 'r-', label='Validation')
plt.title('Linear Loss')
plt.yscale('log')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, best_model_stats['total_loss_tr'], 'b-', label='Train')
plt.plot(epochs, best_model_stats['total_loss_va'], 'r-', label='Validation')
plt.title('Total Loss')
plt.yscale('log')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(analysis_folder, 'best_model_training_curves.png'))
plt.close()

print("Done!")