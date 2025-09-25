import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import os

#**************** To do ****************
# 1. Plot multiple trajectories
#*************************************

def plot_eigenvalues_on_unit_circle(eigenvalues, folder_path, save=True):
    """
    Plot eigenvalues on the unit circle.
    
    Args:
        eigenvalues: Array of eigenvalues
        folder_path: Directory path to save the figure
    """
    plt.figure(figsize=(8, 8))
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.7)
    plt.gca().add_patch(circle)

    # Plot eigenvalues
    plt.scatter(eigenvalues.real, eigenvalues.imag, c='blue', alpha=0.8)

    # Set equal aspect ratio
    plt.axis('equal')

    # Set limits slightly larger than the unit circle
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)

    plt.title('Eigenvalues on the Unit Circle')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Save the figure
    if save:
        plt.savefig(os.path.join(folder_path, 'eigenvalues_unit_circle.png'))
    else:
        plt.show()


def violin_plot(model, test_data, device, folder, save=True):
    """
    Create violin plots showing reconstruction error distribution across dimensions
    """
    model.eval()

    with torch.no_grad():
        X = test_data.to(device)
        Y, reconstructed_X = model.ae(X)
    
    M = X.shape[2]  # Number of dimensions
    fig, axes = plt.subplots(2, M, figsize=(5 * M, 8))
    
    for dim in range(M):
        # Original vs Reconstructed Distribution
        orig_vals = X[:, :, dim].cpu().detach().numpy().flatten()  # Flatten to 1D
        recon_vals = reconstructed_X[:, :, dim].cpu().detach().numpy().flatten()  # Flatten to 1D

        df_dist = pd.DataFrame({
            'value': np.concatenate([orig_vals, recon_vals]),
            'type': ['Original'] * len(orig_vals) + ['Reconstructed'] * len(recon_vals)
        })
        sns.violinplot(data=df_dist, x='type', y='value', ax=axes[0, dim])
        axes[0, dim].set_title(f'Dimension {dim} - Distribution')
        
        # Error Distribution
        error = orig_vals - recon_vals
        df_error = pd.DataFrame({'error': error})
        
        sns.violinplot(data=df_error, y='error', ax=axes[1, dim])
        axes[1, dim].set_title(f'Dimension {dim} - Reconstruction Error')

    plt.tight_layout()
    if save:
        plt.savefig(f'{folder}/violin_plot.png', bbox_inches='tight', dpi=300)
    else:
        plt.show()



def plot_trajectories(ref_trajectory, pred_trajectory=None, recon_trajectory=None, folder_path='', save=True):
    """Plots reference, (optional) predicted, and (optional) reconstructed trajectories up to 3 dimensions."""
    ref_trajectory = np.asarray(ref_trajectory)
    if pred_trajectory is not None:
        pred_trajectory = np.asarray(pred_trajectory)
    if recon_trajectory is not None:
        recon_trajectory = np.asarray(recon_trajectory)

    if pred_trajectory is not None and pred_trajectory.shape != ref_trajectory.shape:
        raise ValueError(f"Predicted trajectory {pred_trajectory.shape} must have shape as {ref_trajectory.shape}")
    if recon_trajectory is not None and recon_trajectory.shape != ref_trajectory.shape:
        raise ValueError(f"Reconstructed trajectory {recon_trajectory.shape} must have shape as {ref_trajectory.shape}")

    dimensions = ref_trajectory.shape[1]
    if dimensions > 3:
        raise ValueError("Function supports up to 3 dimensions only")

    plt.figure(figsize=(10, 8))
    
    if dimensions == 1:
        plt.plot(ref_trajectory[:, 0], label='Reference', color='blue', linewidth=2, marker='o')
        if pred_trajectory is not None:
            plt.plot(pred_trajectory[:, 0], label='Predicted', color='red', linewidth=2, linestyle='--', marker='x')
        if recon_trajectory is not None:
            plt.plot(recon_trajectory[:, 0], label='Reconstructed', color='green', linewidth=2, linestyle='-.', marker='s')
        plt.xlabel('Time')
        plt.ylabel('Value')
        
    elif dimensions == 2:
        plt.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], label='Reference', 
                 color='blue', linewidth=2, marker='o')
        if pred_trajectory is not None:
            plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], label='Predicted', 
                     color='red', linewidth=2, linestyle='--', marker='x')
        if recon_trajectory is not None:
            plt.plot(recon_trajectory[:, 0], recon_trajectory[:, 1], label='Reconstructed', 
                     color='green', linewidth=2, linestyle='-.', marker='s')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.axis('equal')
        
    elif dimensions == 3:
        ax = plt.axes(projection='3d')
        ax.plot3D(ref_trajectory[:, 0], ref_trajectory[:, 1], ref_trajectory[:, 2], 
                  label='Reference', color='blue', linewidth=2, marker='o')
        if pred_trajectory is not None:
            ax.plot3D(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
                      label='Predicted', color='red', linewidth=2, linestyle='--', marker='x')
        if recon_trajectory is not None:
            ax.plot3D(recon_trajectory[:, 0], recon_trajectory[:, 1], recon_trajectory[:, 2], 
                  label='Reconstructed', color='green', linewidth=2, linestyle='-.', marker='s')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')

    plt.title('Trajectory Comparison', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(folder_path, 'trajectories.png') if folder_path else 'trajectories.png'
    if save:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_trajectory_comparison(ref_trajectory, pred_trajectory=None, recon_trajectory=None, 
                               time=None, folder_path='', save=True):
    """
    Plots reference, predicted, and reconstructed trajectories over time steps.
    X-axis always shows step indices. When time is provided, it's shown on the top axis.

    Parameters:
        ref_trajectory (np.ndarray): Reference trajectory, shape (length, dimensions)
        pred_trajectory (np.ndarray): Predicted trajectory, shape (length, dimensions) (optional, defaults to None)
        recon_trajectory (np.ndarray): Reconstructed trajectory, shape (length, dimensions) (optional, defaults to None)
        time (np.ndarray, optional): Time array, shape (length,)
        folder_path (str): Path to save the figure (optional)
        save (bool): If True, saves the plot; otherwise, displays it

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    ref_trajectory = np.asarray(ref_trajectory)
    if pred_trajectory is not None:
        pred_trajectory = np.asarray(pred_trajectory)
    if recon_trajectory is not None:
        recon_trajectory = np.asarray(recon_trajectory)

    length, dimensions = ref_trajectory.shape

    if pred_trajectory is not None and pred_trajectory.shape != ref_trajectory.shape:
        raise ValueError(f"Predicted trajectory {pred_trajectory.shape} must have shape as {ref_trajectory.shape}")
    if recon_trajectory is not None and recon_trajectory.shape != ref_trajectory.shape:
        raise ValueError(f"Reconstructed trajectory {recon_trajectory.shape} must have shape as {ref_trajectory.shape}")

    # X-axis is always step indices
    step_indices = np.arange(length)
    
    if time is not None:
        time = np.asarray(time)
        if time.shape[0] != length:
            raise ValueError("Time array length must match trajectory length.")
        use_custom_time = True
    else:
        use_custom_time = False

    fig, axes = plt.subplots(dimensions, 1, figsize=(12, 4 * dimensions), sharex=True, dpi=100)

    if dimensions == 1:
        axes = [axes]

    for dim in range(dimensions):
        ax = axes[dim]
        # Plot against step indices
        ax.plot(step_indices, ref_trajectory[:, dim], label=f'Reference (Dim {dim})', color='blue', linewidth=2.5)
        if pred_trajectory is not None:
            ax.plot(step_indices, pred_trajectory[:, dim], label=f'Predicted (Dim {dim})', color='red', linestyle='--', linewidth=2.5)
        if recon_trajectory is not None:
            ax.plot(step_indices, recon_trajectory[:, dim], label=f'Reconstructed (Dim {dim})', color='green', linestyle='-.', linewidth=2.5)

        ax.set_title(f'Dimension {dim} Over Time Steps', fontsize=16, pad=10)
        ax.set_ylabel('Value', fontsize=14)
        ax.legend(loc='best', fontsize=12, frameon=True, edgecolor='black')
        ax.grid(True, linestyle='--', alpha=0.7, zorder=-1)
        ax.tick_params(axis='both', labelsize=12)

    # Bottom axis always shows step indices
    axes[-1].set_xlabel('Step Index', fontsize=14, labelpad=10)

    # Top axis shows time if provided, otherwise also step indices
    ax_top = axes[0].twiny()
    if use_custom_time:
        # Map step indices to time values for the top axis
        ax_top.set_xlim(0, length - 1)
        
        # Choose 5 evenly spaced step indices for tick marks
        tick_step_indices = np.linspace(0, length - 1, 5, dtype=int)
        corresponding_times = time[tick_step_indices]
        
        ax_top.set_xticks(tick_step_indices)
        ax_top.set_xticklabels([f'{t:.3f}' for t in corresponding_times])
        ax_top.set_xlabel('Time', fontsize=14, labelpad=10)
    else:
        # If no time provided, top axis also shows step indices (redundant but consistent)
        ax_top.set_xlim(0, length - 1)
        ax_top.set_xticks(np.linspace(0, length - 1, 5, dtype=int))
        ax_top.set_xlabel('Step Index', fontsize=14, labelpad=10)
    
    ax_top.tick_params(axis='both', labelsize=12)

    # Add top banner
    if use_custom_time:
        t_min, t_max = time.min(), time.max()
        delta_t = np.mean(np.diff(time))
        banner_text = f"Time Range: {t_min:.3f} to {t_max:.3f} | Δt ≈ {delta_t:.3f} | {length} steps"
        fig.text(0.5, 0.98, banner_text, ha='center', va='top', fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='gray', alpha=0.9))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        banner_text = f"{length} time steps"
        fig.text(0.5, 0.98, banner_text, ha='center', va='top', fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='gray', alpha=0.9))
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save or show
    save_path = os.path.join(folder_path, 'trajectory_comparison.png') if folder_path else 'trajectory_comparison.png'
    if save:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    else:
        plt.show()
    