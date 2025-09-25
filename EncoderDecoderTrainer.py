import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import einops
from utils.tools import *
from Trainer_Base import BaseTrainer, EarlyStopping


class Trainer(BaseTrainer):
    def __init__(self, model, args, device,do_eval, train_loader, val_loader):
        super().__init__(model, args, device,do_eval, train_loader, val_loader)
        self.context_length = args.context_length
        self.prediction_length = args.prediction_length

    def _criterion(self, prediction, target_data):
        """
        Custom loss which differentiates between GRU and Transformer encoders.
        """
        if 'transformer' in self.model.name:
            pred_loss = self.criterion(prediction, target_data)
        elif "gru" in self.model.name:
            pred_loss = self.criterion(prediction[:,-1,:], target_data[:, -1, :])
        else:
            raise NotImplementedError(f"Encoder {self.model.name} not implemented")
        
        return pred_loss

    def train(self):
        clip_grad_norm = self.args.gradclip if self.gradclip else None

        self.parameters = list(self.model.parameters())

        stats = {
            'pred_loss_tr': [], 'total_loss_tr': [],
            'pred_loss_va': [], 'total_loss_va': []
        }

        ES = EarlyStopping(patience=10, min_delta=0.00001) if self.early_stopping else None
        do_val = self.do_eval

        for epoch in tqdm(range(self.num_epochs), desc="Training Epochs"):
            self.model.train()

            pred_loss_tr = 0
            total_loss_tr = 0
            num_batches = 0

            for batch_idx, data_list in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                data = data_list[0].to(self.device)
                
                # Start with initial context window
                context_length = self.context_length
                input_window = data[:, 0:context_length, :].clone()
                total_pred_loss = 0
                num_steps = data.shape[1] - context_length
                
                for i in range(num_steps):
                    # Get model prediction
                    prediction_dict = self.model(input_window)
                    prediction = prediction_dict['output']
                    
                    # Get target (next time step)
                    target = data[:, i+1:i+context_length+1, :]
                    
                    # Calculate loss for this step
                    step_loss = self._criterion(prediction, target)
                    total_pred_loss += step_loss
                    
                    # Update input_window using the prediction for the next iteration
                    if i < num_steps - 1:  # Don't need to update for the last step
                        # Use the prediction as the next input window
                        input_window = prediction.clone()
                
                # Average loss over all steps
                pred_loss = total_pred_loss / num_steps

                total_loss = pred_loss

                total_loss.backward()

                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.parameters, clip_grad_norm)
                self.optimizer.step()

                pred_loss_tr += pred_loss.item()
                total_loss_tr += total_loss.item()
                num_batches += 1

            pred_loss_tr /= num_batches
            total_loss_tr /= num_batches

            stats['pred_loss_tr'].append(pred_loss_tr)
            stats['total_loss_tr'].append(total_loss_tr)

            self.lr_scheduler(self.optimizer, epoch, lr_decay_rate=self.learning_rate_change, decayEpoch=self.epoch_update)

            if do_val:
                self.model.eval()
                with torch.no_grad():
                    pred_loss_va = 0
                    total_loss_va = 0
                    num_val_batches = 0

                    for batch_idx, data_list in enumerate(self.val_loader):
                        data = data_list[0].to(self.device)
                        
                        # Start with initial context window
                        context_length = self.context_length
                        input_window = data[:, 0:context_length, :].clone()
                        total_pred_loss = 0
                        num_steps = data.shape[1] - context_length
                        
                        for i in range(num_steps):
                            # Get model prediction
                            prediction_dict = self.model(input_window)
                            prediction = prediction_dict['output']
                            
                            # Get target (next time step)
                            target = data[:, i+1:i+context_length+1, :]
                            
                            # Calculate loss for this step
                            step_loss = self._criterion(prediction, target)
                            total_pred_loss += step_loss
                            
                            # Update input_window using the prediction for the next iteration
                            if i < num_steps - 1:  # Don't need to update for the last step
                                # Use the prediction as the next input window
                                input_window = prediction.clone()
                        
                        # Average loss over all steps
                        pred_loss = total_pred_loss / num_steps

                        total_loss = pred_loss

                        pred_loss_va += pred_loss.item()
                        total_loss_va += total_loss.item()
                        num_val_batches += 1

                    pred_loss_va /= num_val_batches
                    total_loss_va /= num_val_batches

                    stats['pred_loss_va'].append(pred_loss_va)
                    stats['total_loss_va'].append(total_loss_va)

                if self.early_stopping:
                    ES(total_loss_va)
                    if ES.early_stop:
                        print(f"Early stopping triggered at epoch {epoch}")
                        self.ES_epochs = epoch
                        break

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}")
                print(f"Training - Pred Loss: {pred_loss_tr:.6e}, Total Loss: {total_loss_tr:.6e}")
                if do_val:
                    print(f"Validation - Pred Loss: {pred_loss_va:.6e}, Total Loss: {total_loss_va:.6e}")

        if self.save:
            self.visualize(stats['pred_loss_tr'], 'Prediction', stats.get('pred_loss_va'))
            self.visualize(stats['total_loss_tr'], 'Total', stats.get('total_loss_va'))

            torch.save(self.model.state_dict(), os.path.join('experiments', self.folder, 'model.pkl'))
            torch.save(stats, os.path.join('experiments', self.folder, 'stats.pkl'))

        return self.model, self.optimizer, stats


    def test(self,test_loader):
        self.model.eval()

        pred_loss_te = 0
        total_loss_te = 0

        with torch.no_grad():
            for batch_idx, data_list in enumerate(test_loader):
                data = data_list[0].to(self.device)
                
                # Start with initial context window
                context_length = self.context_length
                input_window = data[:, 0:context_length, :].clone()
                total_pred_loss = 0
                num_steps = data.shape[1] - context_length
                
                for i in range(num_steps):
                    # Get model prediction
                    prediction_dict = self.model(input_window)
                    prediction = prediction_dict['output']
                    
                    # Get target (next time step)
                    target = data[:, i+1:i+context_length+1, :]
                    
                    # Calculate loss for this step
                    step_loss = self._criterion(prediction, target)
                    total_pred_loss += step_loss
                    
                    # Update input_window using the prediction for the next iteration
                    if i < num_steps - 1:  # Don't need to update for the last step
                        # Use the prediction as the next input window
                        input_window = prediction.clone()
                
                # Average loss over all steps
                pred_loss = total_pred_loss / num_steps

                total_loss = pred_loss

                pred_loss_te += pred_loss.item()
                total_loss_te += total_loss.item()

            pred_loss_te /= len(test_loader)
            total_loss_te /= len(test_loader)

        print(f"Test - Pred Loss: {pred_loss_te:.6e}, Total Loss: {total_loss_te:.6e}")
    
    def predict_traj(self, X0, steps=50)->torch.Tensor:
        """
        Predict a trajectory given an initial state.
        """
        tmp_in = einops.rearrange(X0, 'dim -> 1 1 dim')

        output = tmp_in
        for _ in range(steps):
            tmp_in = output[:, -self.model.input_length:, :]
            tmp_out = self.model(tmp_in)['output'][:, -1:, :]
            output = torch.cat([output, tmp_out], dim=1)
        outputs = einops.rearrange(output, '1 seq dim -> seq dim')

        return outputs
