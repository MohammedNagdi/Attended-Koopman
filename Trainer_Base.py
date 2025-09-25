from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from utils.tools import *


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Args:
            val_loss (float): The current validation loss.
        """
        # Initialize best_loss if it's the first validation
        if self.best_loss is None:
            self.best_loss = val_loss
        # If there is an improvement (greater than min_delta), reset counter
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        # If no improvement, increase the counter
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                self.early_stop = True

# relative mse loss function
class RelativeMSELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(RelativeMSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        numerator = torch.mean((y_true - y_pred) ** 2)
        denominator = torch.mean(y_true ** 2) + self.epsilon
        return numerator / denominator
    
class BaseTrainer(ABC):
    def __init__(self, model, args, device,do_eval, train_loader, val_loader):
        self.args = args
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.device = get_device()
        self.model = model.to(device)
        self.num_epochs = args.epochs
        self.learning_rate_change = args.lr_decay
        self.epoch_update = args.lr_update
        self.gradclip = args.gradclip
        self.folder = args.folder
        self.save = args.save
        self.early_stopping = args.early_stopping
        self.ES_epochs = 0
        self.do_eval = do_eval
        self.prediction_length = args.prediction_length

        # losses 
        self.decoder_loss_weight = args.decoder_loss_weight if hasattr(args, 'decoder_loss_weight') else 1e-2
        self.loss_function = args.loss_function if hasattr(args, 'loss_function') else 'mse'
        self.unitary_loss_weight = args.unitary_loss_weight if hasattr(args, 'unitary_loss_weight') else 1e-2

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if self.loss_function == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss_function == 'relative_mse':
            self.criterion = RelativeMSELoss()

    def lr_scheduler(self, optimizer, epoch, lr_decay_rate=0.8, decayEpoch=[]):
        if epoch in decayEpoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_decay_rate
            return self.optimizer
        else:
            return self.optimizer

    def visualize(self, loss_tr, label, loss_va=None):
        fig = plt.figure(figsize=(15, 12))
        
        # Convert training loss to numpy array
        loss_tr_array = np.array(loss_tr)
        if np.any(loss_tr_array <= 0):
            min_positive = np.min(loss_tr_array[loss_tr_array > 0]) / 10
            loss_tr_array = np.maximum(loss_tr_array, min_positive)

        # Plot training loss
        plt.semilogy(loss_tr_array, lw=2, marker='o', markersize=5, 
                     markerfacecolor='white', markeredgewidth=1.5, 
                     markeredgecolor='#377eb8', linestyle='-', 
                     label=f'Training {label} Loss', color='#377eb8')

        # If validation loss is provided, plot it on the same figure
        if loss_va is not None:
            loss_va_array = np.array(loss_va)
            if np.any(loss_va_array <= 0):
                min_positive = np.min(loss_va_array[loss_va_array > 0]) / 10
                loss_va_array = np.maximum(loss_va_array, min_positive)
            plt.semilogy(loss_va_array, lw=2, marker='s', markersize=5, 
                         markerfacecolor='white', markeredgewidth=1.5, 
                         markeredgecolor='#ff7f00', linestyle='--', 
                         label=f'Validation {label} Loss', color='#ff7f00')

        # Styling
        plt.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
        plt.tick_params(axis='both', which='major', labelsize=18, width=1.5, length=6)
        plt.tick_params(axis='both', which='minor', width=1, length=4)
        plt.ylabel('Loss (log scale)', fontsize=24, fontweight='bold')
        plt.xlabel('Epoch', fontsize=24, fontweight='bold')
        ax = plt.gca()
        ax.set_facecolor('#f8f9fa')
        plt.title(f'{label} Loss Over Training', fontsize=26, fontweight='bold', pad=20)
        plt.legend(fontsize=20, frameon=True, fancybox=True, framealpha=0.9, 
                   shadow=True, loc='upper right')
        ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(1, 10)*0.1))
        
        # Save figure
        fig.tight_layout()
        plt.savefig(os.path.join('experiments', self.folder, f'{label}_log_scale.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    @abstractmethod
    def train(self):
        raise NotImplementedError("Not implemented")
    
    @abstractmethod
    def test(self,test_loader):
        raise NotImplementedError("Not implemented")
