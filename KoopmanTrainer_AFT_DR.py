import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from utils.tools import *
from Trainer_Base import BaseTrainer, EarlyStopping
from collections import deque
from scipy.stats import ranksums, mood, mannwhitneyu, ks_2samp, cramervonmises_2samp


class Trainer(BaseTrainer):
    def __init__(self, model, args, device,do_eval, train_loader, val_loader):
        super().__init__(model, args, device,do_eval, train_loader, val_loader)

    def normalize_decoder_columns(self,decoder):
        with torch.no_grad():
            weight = decoder.weight  # [output_dim, latent_dim]
            norms = torch.norm(weight, p=2, dim=0, keepdim=True)  # Column norms
            decoder.weight.data = weight / norms  # Normalize columns

    def _evolve(self, Y0) -> torch.Tensor:
        """
        Autoregressively evolve latent states with fixed context, AFT, and Koopman operator.
        Properly integrates periodic reencoding and avoids inplace operations.
        """

        T = self.prediction_length
        m = self.args.context_length

        # Initialize latent trajectory with the initial state
        Ypred_list = [self.model.knet(Y0)]  # List to collect all latents

        for t in range(1, T):

            # the aft layer expects a context of size m or less 
            if t < m:
                # without padding
                context = torch.stack(Ypred_list, dim=1)
            else:
                # Use last `m` latents from the list
                context = torch.stack(Ypred_list[-m:], dim=1)

            # --- Prediction Step ---
            # Process context with AFT
            aft_output =  Ypred_list[-1] + self.model.aft_layer(context)
            # Evolve using Koopman operator (predict next latent)
            
            new_latent = self.model.knet(aft_output)  # Use last context entry
            # Append to trajectory
            Ypred_list.append(new_latent)

        # Convert list to tensor: [B, T, D]
        return torch.stack(Ypred_list, dim=1)
    

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        clip_grad_norm = self.args.gradclip if self.gradclip else None

        self.parameters = list(self.model.ae.parameters()) + list(self.model.knet.parameters()) + list(self.model.aft_layer.parameters())

        stats = {
            'recon_loss_tr': [], 'lin_loss_tr': [], 'pred_loss_tr': [], 'total_loss_tr': [],
            'recon_loss_va': [], 'lin_loss_va': [], 'pred_loss_va': [], 'total_loss_va': []
        }

        ES = EarlyStopping(patience=10, min_delta=0.00001) if self.early_stopping else None
        do_val = self.do_eval


        for epoch in tqdm(range(self.num_epochs), desc="Training Epochs"):
            self.model.ae.train()
            self.model.knet.train()
            self.model.aft_layer.train()

            recon_loss_tr = 0
            lin_loss_tr = 0
            pred_loss_tr = 0
            total_loss_tr = 0
            num_batches = 0
            unitary_loss_tr = 0

            for batch_idx, data_list in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                data = data_list[0].to(self.device)
                Ytr, Xrtr = self.model.ae(data)

                Ypredtr = self._evolve(Ytr[:, 0, :])
                Xpredtr = self.model.ae.decoder(Ypredtr)

                recon_loss = self.criterion(Xrtr, data)
                pred_loss = self.criterion(Xpredtr, data[:, 1:,:])
                lin_loss = self.criterion(Ypredtr, Ytr[:, 1:, :])

                # add the unitary loss on the koopman operator
                # Unitary loss normalized by matrix size
                K = self.model.knet.net.weight
                K_T = torch.conj(K.T)
                n = K.shape[0]
                unitary_loss = torch.norm(K @ K_T - torch.eye(n, device=self.device), p='fro') / (n * n)
                total_loss = lin_loss + self.decoder_loss_weight * (recon_loss + pred_loss) + self.unitary_loss_weight * unitary_loss 
                
                total_loss.backward()
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.parameters, clip_grad_norm)
                self.optimizer.step()

                #self.normalize_decoder_columns(self.model.ae.decoder)

                recon_loss_tr += recon_loss.item()
                lin_loss_tr += lin_loss.item()
                pred_loss_tr += pred_loss.item()
                total_loss_tr += total_loss.item()
                unitary_loss_tr += unitary_loss
                num_batches += 1

            recon_loss_tr /= num_batches
            lin_loss_tr /= num_batches
            pred_loss_tr /= num_batches
            unitary_loss_tr /= num_batches
            total_loss_tr /= num_batches

            stats['recon_loss_tr'].append(recon_loss_tr)
            stats['lin_loss_tr'].append(lin_loss_tr)
            stats['pred_loss_tr'].append(pred_loss_tr)
            stats['total_loss_tr'].append(total_loss_tr)

            self.lr_scheduler(self.optimizer, epoch, lr_decay_rate=self.learning_rate_change, decayEpoch=self.epoch_update)

            if do_val:
                self.model.ae.eval()
                self.model.knet.eval()
                self.model.aft_layer.eval()

                with torch.no_grad():
                    recon_loss_va = 0
                    lin_loss_va = 0
                    pred_loss_va = 0
                    total_loss_va = 0
                    num_val_batches = 0
                    unitary_loss_va = 0

                    for batch_idx, data_list in enumerate(self.val_loader):
                        data = data_list[0].to(self.device)
                        Yva, Xrva = self.model.ae(data)
                        Ypredva = self._evolve(Yva[:, 0, :])
                        Xpredva = self.model.ae.decoder(Ypredva)

                        recon_loss = self.criterion(Xrva, data) 
                        pred_loss = self.criterion(Xpredva, data[:, 1:,:])
                        lin_loss = self.criterion(Ypredva, Yva[:, 1:, :])
                        total_loss = lin_loss + self.decoder_loss_weight * (recon_loss + pred_loss)

                        recon_loss_va += recon_loss.item()
                        lin_loss_va += lin_loss.item()
                        pred_loss_va += pred_loss.item()
                        unitary_loss_va += unitary_loss.item()
                        total_loss_va += total_loss.item()
                        num_val_batches += 1

                    recon_loss_va /= num_val_batches
                    lin_loss_va /= num_val_batches
                    pred_loss_va /= num_val_batches
                    unitary_loss_va /= num_val_batches
                    total_loss_va /= num_val_batches

                    stats['recon_loss_va'].append(recon_loss_va)
                    stats['lin_loss_va'].append(lin_loss_va)
                    stats['pred_loss_va'].append(pred_loss_va)
                    stats['total_loss_va'].append(total_loss_va)

                if self.early_stopping:
                    ES(total_loss_va)
                    if ES.early_stop:
                        print(f"Early stopping triggered at epoch {epoch}")
                        self.ES_epochs = epoch
                        break

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}")
                print(f"Training - Recon Loss: {recon_loss_tr:.6e}, Linear Loss: {lin_loss_tr:.6e}, Pred Loss: {pred_loss_tr:.6e}, Unitary Loss: {unitary_loss_tr:.6e}, Total Loss: {total_loss_tr:.6e}")
                if do_val:
                    print(f"Validation - Recon Loss: {recon_loss_va:.6e}, Linear Loss: {lin_loss_va:.6e}, Pred Loss: {pred_loss_va:.6e},Unitary Loss: {unitary_loss_va:.6e}, Total Loss: {total_loss_va:.6e}")

        if self.save:
            self.visualize(stats['recon_loss_tr'], 'Reconstruction', stats.get('recon_loss_va'))
            self.visualize(stats['lin_loss_tr'], 'Linear', stats.get('lin_loss_va'))
            self.visualize(stats['pred_loss_tr'], 'Prediction', stats.get('pred_loss_va'))
            self.visualize(stats['total_loss_tr'], 'Total', stats.get('total_loss_va'))

            torch.save(self.model.state_dict(), os.path.join('experiments', self.folder, 'model.pkl'))
            torch.save(stats, os.path.join('experiments', self.folder, 'stats.pkl'))

        return self.model, self.optimizer, stats

    def train_KoopmanAE(self):
        return self.train()
    
    def test(self, test_loader):
        self.model.ae.eval()
        self.model.knet.eval()
        self.model.aft_layer.eval()

        recon_loss_te = 0
        lin_loss_te = 0
        pred_loss_te = 0
        total_loss_te = 0

        decoder_loss_weight = self.args.decoder_loss_weight

        with torch.no_grad():
            for batch_idx, data_list in enumerate(test_loader):
                data = data_list[0].to(self.device)
                Yte, Xrte = self.model.ae(data)
                Ypredte = self._evolve(Yte[:, 0, :])
                Xpredte = self.model.ae.decoder(Ypredte)


                recon_loss = self.criterion(Xrte, data)
                pred_loss = self.criterion(Xpredte, data[:, 1:,:])
                lin_loss = self.criterion(Ypredte, Yte[:, 1:, :])
                total_loss = lin_loss + decoder_loss_weight * (recon_loss + pred_loss)

                recon_loss_te += recon_loss.item()
                lin_loss_te += lin_loss.item()
                pred_loss_te += pred_loss.item()
                total_loss_te += total_loss.item()

            recon_loss_te /= len(test_loader)
            lin_loss_te /= len(test_loader)
            pred_loss_te /= len(test_loader)
            total_loss_te /= len(test_loader)

        print(f"Test - Recon Loss: {recon_loss_te:.6e}, Linear Loss: {lin_loss_te:.6e}, Pred Loss: {pred_loss_te:.6e}, Total Loss: {total_loss_te:.6e}")

    def test_KoopmanAE(self, test_loader):
        self.test(test_loader)

    def trainAE(self):
        self.model.ae.train()

        recon_loss_tr = 0

        stats = {
            'recon_loss_tr': [], 'recon_loss_va': []
        }
        

        for epoch in tqdm(range(self.num_epochs), desc="Training Epochs"):
            for batch_idx, data_list in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                data = data_list[0].to(self.device)
                Ytr, Xrtr = self.model.ae(data)
                recon_loss = self.criterion(Xrtr, data)
                recon_loss.backward()
                self.optimizer.step()
                recon_loss_tr += recon_loss.item()
                

            recon_loss_tr /= len(self.train_loader)
            stats['recon_loss_tr'].append(recon_loss_tr)

            self.lr_scheduler(self.optimizer, epoch, lr_decay_rate=self.learning_rate_change, decayEpoch=self.epoch_update)

            if self.do_eval:
                self.model.ae.eval()
                with torch.no_grad():
                    recon_loss_va = 0
                    
                    for batch_idx, data_list in enumerate(self.val_loader):
                        data = data_list[0].to(self.device)
                        Yva, Xrva = self.model.ae(data)
                        recon_loss = self.criterion(Xrva, data)
                        recon_loss_va += recon_loss.item()
                        
                    recon_loss_va /= len(self.val_loader)
                    stats['recon_loss_va'].append(recon_loss_va)

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}")
                print(f"Training - Recon Loss: {recon_loss_tr:.6e}")
                if self.do_eval:
                    print(f"Validation - Recon Loss: {recon_loss_va:.6e}")

        if self.save:
            self.visualize(stats['recon_loss_tr'], 'Reconstruction', stats.get('recon_loss_va'))
            torch.save(self.model.state_dict(), os.path.join('experiments', self.folder, 'model.pkl'))
            torch.save(stats, os.path.join('experiments', self.folder, 'stats.pkl'))

        return self.model, self.optimizer, stats
    

    def train_Koopman(self):
        self.model.knet.train()
        self.model.ae.eval()

        stats = {
            'pred_loss_tr': [], 'pred_loss_va': []
        }


        for epoch in tqdm(range(self.num_epochs), desc="Training Epochs"):
            pred_loss_tr = 0
            for batch_idx, data_list in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                data = data_list[0].to(self.device)
                Ytr, Xrtr = self.model.ae(data)
                Ypredtr = self._evolve(Ytr[:, 0, :])
                Xpredtr = self.model.ae.decoder(Ypredtr)
                pred_loss = self.criterion(Xpredtr, data[:, 1:,:])
                pred_loss.backward()
                self.optimizer.step()
                pred_loss_tr += pred_loss.item()

            pred_loss_tr /= len(self.train_loader)
            stats['pred_loss_tr'].append(pred_loss_tr)

            self.lr_scheduler(self.optimizer, epoch, lr_decay_rate=self.learning_rate_change, decayEpoch=self.epoch_update)

            if self.do_eval:

                with torch.no_grad():
                    pred_loss_va = 0

                    for batch_idx, data_list in enumerate(self.val_loader):
                        data = data_list[0].to(self.device)
                        Yva, Xrva = self.model.ae(data)
                        Ypredva = self._evolve(Yva[:, 0, :])
                        Xpredva = self.model.ae.decoder(Ypredva)
                        pred_loss = self.criterion(Xpredva, data[:, 1:,:])
                        pred_loss_va += pred_loss.item()

                    pred_loss_va /= len(self.val_loader)
                    stats['pred_loss_va'].append(pred_loss_va)

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}")
                print(f"Training - Pred Loss: {pred_loss_tr:.6e}")
                if self.do_eval:
                    print(f"Validation - Pred Loss: {pred_loss_va:.6e}")    

        if self.save:
            self.visualize(stats['pred_loss_tr'], 'Prediction', stats.get('pred_loss_va'))
            torch.save(self.model.state_dict(), os.path.join('experiments', self.folder, 'model.pkl'))
            torch.save(stats, os.path.join('experiments', self.folder, 'stats.pkl'))

        return self.model, self.optimizer, stats

    def reconstruct_traj(self, traj) -> torch.Tensor:
        Y, reconstructed_traj = self.model.ae(traj)
        return reconstructed_traj

    def predict_new(self, X0, steps=50)->torch.Tensor:
        return self.predict_traj(X0, steps)

    def predict_traj(self, X0, steps=50) -> torch.Tensor:
        """
        Predict future states from a single initial condition using AFT and Koopman operator.
        """
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X0, torch.Tensor):
                X0 = torch.tensor(X0, device=self.device)
            
            X0_batch = X0.unsqueeze(0) if X0.ndim == 1 else X0
            
            # Initialize prediction trajectory
            Xpred = torch.zeros((steps+1, *X0.shape), device=self.device)
            Xpred[0] = X0
            
            m = self.args.context_length  # Context window size
            
            # Initialize latent trajectory with the initial state
            Y0 = self.model.ae.encoder(X0_batch)
            Y_history = [Y0.squeeze(0)]
            
            Y_next = self.model.knet(Y0)
            Xpred[1] = self.model.ae.decoder(Y_next).squeeze(0)
            Y_history.append(Y_next.squeeze(0))
            
            for t in range(2, steps+1):
                # Construct context window 
                if t < m:
                    context_list = Y_history[:t]
                else:
                    context_list = Y_history[-m:]
                
                context = torch.stack(context_list).unsqueeze(0)
                aft_output = Y_history[-1] + self.model.aft_layer(context)
                Y_next = self.model.knet(aft_output)
                Xpred[t] = self.model.ae.decoder(Y_next).squeeze(0)
                Y_history.append(Y_next.squeeze(0))
                
        # Y_history is the latent history
        return Xpred

    def predict_new_Periodic_Reencoding(self, X0,reenc = 10, steps=50) -> torch.Tensor:
        """
        Predict future states from a single initial condition using AFT and Koopman operator.
        the Reencoding policy is periodic reencoding
        """
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X0, torch.Tensor):
                X0 = torch.tensor(X0, device=self.device)
            
            X0_batch = X0.unsqueeze(0) if X0.ndim == 1 else X0
            
            Xpred = torch.zeros((steps+1, *X0.shape), device=self.device)
            Xpred[0] = X0
            
            m = self.args.context_length  # Context window size
            

            Y0 = self.model.ae.encoder(X0_batch)
            Y_history = [Y0.squeeze(0)]
            
            Y_next = self.model.knet(Y0)
            Xpred[1] = self.model.ae.decoder(Y_next).squeeze(0)
            Y_history.append(Y_next.squeeze(0))
            
            for t in range(2, steps+1):
                # Construct context window 
                if t < m:
                    context_list = Y_history[:t]
                else:
                    context_list = Y_history[-m:]
                
                context = torch.stack(context_list).unsqueeze(0)
                aft_output = Y_history[-1] + self.model.aft_layer(context)
                Y_next = self.model.knet(aft_output)
                Xpred[t] = self.model.ae.decoder(Y_next).squeeze(0)
                ### reencoding every n steps###
                if t % reenc == 0:
                    Y_next = self.model.ae.encoder(Xpred[t])
                Y_history.append(Y_next.squeeze(0))
                #################################
                
        
        return Xpred , Y_history  # Return both predictions and latent history
    

    def predict_new_Point_Reencoding(self, X0, reenc = [],steps = 50):
        """
        Predict future states from a single initial condition using AFT and Koopman operator.
        the Reencoding policy is point reencoding
        """
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X0, torch.Tensor):
                X0 = torch.tensor(X0, device=self.device)
            
            X0_batch = X0.unsqueeze(0) if X0.ndim == 1 else X0
            
            Xpred = torch.zeros((steps+1, *X0.shape), device=self.device)
            Xpred[0] = X0
            
            m = self.args.context_length  # Context window size

            Y0 = self.model.ae.encoder(X0_batch)
            Y_history = [Y0.squeeze(0)]

            Y_next = self.model.knet(Y0)
            Xpred[1] = self.model.ae.decoder(Y_next).squeeze(0)
            Y_history.append(Y_next.squeeze(0))

            for t in range(2, steps+1):
                # Construct context window 
                if t < m:
                    context_list = Y_history[:t]
                else:
                    context_list = Y_history[-m:]

                context = torch.stack(context_list).unsqueeze(0)
                aft_output = Y_history[-1] + self.model.aft_layer(context)
                Y_next = self.model.knet(aft_output)
                Xpred[t] = self.model.ae.decoder(Y_next).squeeze(0)

                ### reencoding every step in reencoding list###
                if t in reenc:
                    Y_next = self.model.ae.encoder(Xpred[t])
                Y_history.append(Y_next.squeeze(0))
                #################################

        return Xpred, Y_history  # Return both predictions and latent history
    

    def predict_with_threshold(self, X0, threshold=0.5, steps=50) -> tuple:
        """
        Predict future states from a single initial condition using AFT and Koopman operator
        with automatic reencoding based on threshold.
        
        Args:
            X0: Initial condition
            threshold: Threshold for normalized difference to trigger reencoding (default: 0.5)
            steps: Number of prediction steps
        
        Returns:
            tuple: (Xpred, Ypred, reenc_points) where Xpred is the predicted trajectory in observation space,
                Ypred is the predicted trajectory in latent space, and reenc_points is a list of 
                time steps where reencoding occurred
        """
        self.model.eval()
        with torch.no_grad():

            if not isinstance(X0, torch.Tensor):
                X0 = torch.tensor(X0, device=self.device)
            
            X0_batch = X0.unsqueeze(0) if X0.ndim == 1 else X0
            
            Xpred = torch.zeros((steps+1, *X0.shape), device=self.device)
            Xpred[0] = X0
            
            m = self.args.context_length  # Context window size
            
            # List to track reencoding points
            reenc_points = []
            
            Y0 = self.model.ae.encoder(X0_batch)
            Y_history = [Y0.squeeze(0)]
            
            Y_next = self.model.knet(Y0)
            Xpred[1] = self.model.ae.decoder(Y_next).squeeze(0)
            Y_history.append(Y_next.squeeze(0))
            
            for t in range(2, steps+1):
                # Construct context window 
                if t < m:
                    context_list = Y_history[:t]
                else:
                    context_list = Y_history[-m:]
                
                context = torch.stack(context_list).unsqueeze(0)
                aft_output = Y_history[-1] + self.model.aft_layer(context)
                Y_next = self.model.knet(aft_output)
                Xpred[t] = self.model.ae.decoder(Y_next).squeeze(0)
                Y_next_before = Y_next
                ### Check if reencoding is needed based on threshold ###
                # Calculate reencoding difference for current latent point
                Y_decoded = self.model.ae.decoder(Y_history[-1])
                Y_encoded = self.model.ae.encoder(Y_decoded)
                context[-1] = Y_encoded  
                aft_output = Y_encoded + self.model.aft_layer(context)
                Y_next_after = self.model.knet(aft_output)    

                # Calculate normalized MSE difference
                mse_difference = torch.mean((Y_next_after - Y_next_before) ** 2)
                normalized_difference = mse_difference / (torch.mean(Y_next_before ** 2) + 1e-8)


                ####################################     
                # Perform reencoding if threshold is exceeded
                if normalized_difference > threshold:
                    Y_history[-1] = Y_encoded
                    Y_next = Y_next_after
                    reenc_points.append(t)
                Xpred[t] = self.model.ae.decoder(Y_next).squeeze(0)
                Y_history.append(Y_next.squeeze(0))
                ####################################
                
        return Xpred, Y_history, reenc_points
    

    def predict_window_variance(self, X0, window_size=10, variance_threshold=2.0, steps=50):
        """
        Predict with reencoding based on variance of MSE differences in a sliding window.
        
        Args:
            X0: Initial condition
            window_size: Size of sliding window for variance calculation
            variance_threshold: Threshold multiplier (reencoder if current_diff > mean + threshold*std)
            steps: Number of prediction steps
        
        Returns:
            tuple: (Xpred, Ypred, reenc_points, mse_history)
        """
        self.model.eval()
        if not isinstance(X0, torch.Tensor):
            X0 = torch.tensor(X0, device=self.device)
        
        X0_batch = X0.unsqueeze(0) if X0.ndim == 1 else X0
        Xpred = torch.zeros((steps+1, *X0.shape), device=self.device)
        Xpred[0] = X0
        
        reenc_points = []
        mse_history = []  # Store MSE differences for window analysis
        
        with torch.no_grad():
            Y0 = self.model.ae.encoder(X0_batch)
            Y_history = [Y0.squeeze(0)]
            
            # First step
            Y_next = self.model.knet(Y0)
            Xpred[1] = self.model.ae.decoder(Y_next).squeeze(0)
            Y_history.append(Y_next.squeeze(0))
            
            # Calculate initial MSE difference for window
            Y_decoded = self.model.ae.decoder(Y_next)
            Y_reenc = self.model.ae.encoder(Y_decoded)
            Y_reenc_pred = self.model.knet(Y_reenc)
            mse_diff = torch.mean((Y_reenc_pred - Y_next) ** 2).item()
            mse_history.append(mse_diff)
            
            for t in range(2, steps+1):
                # 1. Apply AFT layer to get intermediate output
                if t < self.args.context_length:
                    context = torch.stack(Y_history[:t]).unsqueeze(0)
                else:
                    context = torch.stack(Y_history[-self.args.context_length:]).unsqueeze(0)
                
                aft_output = Y_history[-1] + self.model.aft_layer(context)
                
                # 2. Apply Koopman to get Y_next (original)
                Y_next = self.model.knet(aft_output)
                
                # 3. Get reencoded version
                Y_decoded = self.model.ae.decoder(aft_output)
                Y_reenc = self.model.ae.encoder(Y_decoded)
                Y_next_reencoded = self.model.knet(Y_reenc)
                
                # 4. Calculate MSE difference for window analysis
                current_mse_diff = torch.mean((Y_next_reencoded - Y_next) ** 2).item()
                mse_history.append(current_mse_diff)
                
                # 5. Window-based variance check to decide which value to use
                use_reencoded = False
                if len(mse_history) >= window_size:
                    window_data = mse_history[-window_size:]
                    window_mean = sum(window_data) / len(window_data)
                    window_var = sum((x - window_mean) ** 2 for x in window_data) / len(window_data)
                    window_std = window_var ** 0.5
                    
                    if current_mse_diff > window_mean + variance_threshold * window_std:
                        use_reencoded = True
                        reenc_points.append(t)
                
                # 6. Replace last value in history with chosen value
                if use_reencoded:
                    Y_history[-1] = Y_reenc.squeeze(0)
                
                # 7. Apply AFT layer again and get final prediction
                if t < self.args.context_length:
                    context = torch.stack(Y_history[:t]).unsqueeze(0)
                else:
                    context = torch.stack(Y_history[-self.args.context_length:]).unsqueeze(0)
                
                final_aft_output = Y_history[-1] + self.model.aft_layer(context)
                final_prediction = self.model.knet(final_aft_output)
                
                # Store results
                Xpred[t] = self.model.ae.decoder(final_prediction).squeeze(0)
                Y_history.append(final_prediction.squeeze(0))
        
        return Xpred, Y_history, reenc_points, mse_history
    
    
    def predict_new_window_variance(self, X0, window_size=10, variance_threshold=2.0, steps=50):
        """
        Predict with reencoding based on variance of MSE differences in a sliding window.
        
        Args:
            X0: Initial condition
            window_size: Size of sliding window for variance calculation
            variance_threshold: Threshold multiplier (reencoder if current_diff > mean + threshold*std)
            steps: Number of prediction steps
        
        Returns:
            tuple: (Xpred, Ypred, reenc_points, mse_history)
        """
        self.model.eval()
        if not isinstance(X0, torch.Tensor):
            X0 = torch.tensor(X0, device=self.device)
        
        X0_batch = X0.unsqueeze(0) if X0.ndim == 1 else X0
        Xpred = torch.zeros((steps+1, *X0.shape), device=self.device)
        Xpred[0] = X0
        
        reenc_points = []
        mse_history = []  # Store MSE differences for window analysis
        
        with torch.no_grad():
            Y0 = self.model.ae.encoder(X0_batch)
            Y_history = [Y0.squeeze(0)]
            
            latent_dim = Y0.shape[-1]
            Ypred = torch.zeros((steps+1, latent_dim), device=self.device)
            Ypred[0] = Y0.squeeze(0)
            
            # First step
            Y_next = self.model.knet(Y0)
            Xpred[1] = self.model.ae.decoder(Y_next).squeeze(0)
            Ypred[1] = Y_next.squeeze(0)
            Y_history.append(Y_next.squeeze(0))
            
            # Calculate initial MSE difference for window
            Y_decoded = self.model.ae.decoder(Y_next)
            Y_reenc = self.model.ae.encoder(Y_decoded)
            Y_reenc_pred = self.model.knet(Y_reenc)
            mse_diff = torch.mean((Y_reenc_pred - Y_next) ** 2).item()
            mse_history.append(mse_diff)
            
            for t in range(2, steps+1):
                # Prediction step
                if t < self.args.context_length:
                    context = torch.stack(Y_history[:t]).unsqueeze(0)
                else:
                    context = torch.stack(Y_history[-self.args.context_length:]).unsqueeze(0)
                
                aft_output = Y_history[-1] + self.model.aft_layer(context)
                Y_next = self.model.knet(aft_output)
                Xpred[t] = self.model.ae.decoder(Y_next).squeeze(0)
                Ypred[t] = Y_next.squeeze(0)
                
                # Calculate MSE difference for current point
                Y_decoded = self.model.ae.decoder(Y_next)
                Y_reenc = self.model.ae.encoder(Y_decoded)
                Y_reenc_pred = self.model.knet(Y_reenc)
                current_mse_diff = torch.mean((Y_reenc_pred - Y_next) ** 2).item()
                mse_history.append(current_mse_diff)
                
                # Window-based variance check
                if len(mse_history) >= window_size:
                    window_data = mse_history[-window_size:]
                    window_mean = sum(window_data) / len(window_data)
                    window_var = sum((x - window_mean) ** 2 for x in window_data) / len(window_data)
                    window_std = window_var ** 0.5
                    
                    # Reencoder if current difference exceeds threshold
                    if current_mse_diff > window_mean + variance_threshold * window_std:
                        Y_next = self.model.ae.encoder(Xpred[t])
                        Ypred[t] = Y_next.squeeze(0)
                        reenc_points.append(t)
                
                Y_history.append(Y_next.squeeze(0))
        
        return Xpred, Ypred, reenc_points, mse_history
    
    def predict_with_twosample(self, X0, threshold=3.1, window_size=20, statistic="Lepage", steps=50) -> tuple:
        """
        Predict future states using AFT and Koopman operator with reencoding triggered by two-sample tests.
        """
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X0, torch.Tensor):
                X0 = torch.tensor(X0, device=self.device)
            
            X0_batch = X0.unsqueeze(0) if X0.ndim == 1 else X0
            Xpred = torch.zeros((steps+1, *X0.shape), device=self.device)
            Xpred[0] = X0

            m = self.args.context_length
            reenc_points = []

            detector = TwoSample(statistic=statistic, threshold=threshold, window_size=window_size)

            Y0 = self.model.ae.encoder(X0_batch)
            Y_history = [Y0.squeeze(0)]

            Y_next = self.model.knet(Y0)
            Xpred[1] = self.model.ae.decoder(Y_next).squeeze(0)
            Y_history.append(Y_next.squeeze(0))

            for t in range(2, steps+1):
                context = torch.stack(Y_history[-m:] if t >= m else Y_history[:t]).unsqueeze(0)
                aft_output = Y_history[-1] + self.model.aft_layer(context)
                Y_next = self.model.knet(aft_output)
                Y_next_before = Y_next

                # Decode and re-encode to check consistency
                Y_decoded = self.model.ae.decoder(Y_history[-1])
                Y_encoded = self.model.ae.encoder(Y_decoded)
                context[-1] = Y_encoded
                aft_output = Y_encoded + self.model.aft_layer(context)
                Y_next_after = self.model.knet(aft_output)

                mse_diff = torch.mean((Y_next_after - Y_next_before) ** 2)
                is_changepoint = detector.process_one(float(mse_diff))

                if is_changepoint:
                    Y_history[-1] = Y_encoded
                    Y_next = Y_next_after
                    reenc_points.append(t)

                Xpred[t] = self.model.ae.decoder(Y_next).squeeze(0)
                Y_history.append(Y_next.squeeze(0))

        return Xpred, Y_history, reenc_points





class TwoSample():
    """Online two-sample test for change detection with rolling windows."""

    def __init__(self, 
                 statistic: str = "Lepage", 
                 threshold: float = 3.1, 
                 window_size: int = 20):
        self.threshold = threshold
        self.stat_name = statistic
        self.window_size = window_size
        self.changepoints = []
        self.t = 0  # Time index

        # Rolling buffers
        self.window_past = deque(maxlen=window_size)
        self.window_recent = deque(maxlen=window_size)

        self._fetch_statistic()

    def _fetch_statistic(self):
        db = {
            "Mann-Whitney": mannwhitneyu,
            "Mood": mood,
            "Lepage": ranksums,
            "Kolmogorov-Smirnov": ks_2samp,
            "Cramer-von-Mises": cramervonmises_2samp
        }
        self.statistic = db[self.stat_name]

    def _compute_statistic(self, x, y):
        try:
            return abs(self.statistic(x, y)[0])
        except:
            try:
                return abs(self.statistic(x, y).statistic)
            except:
                return 0.0

    def process_one(self, data_point: float) -> bool:
        """
        Process a single data point and check for a change point.
        
        Parameters
        ----------
        data_point : float
            New observation.

        Returns
        -------
        is_changepoint : bool
            True if a change point is detected, False otherwise.
        """
        self.t += 1

        # Fill past window first
        if len(self.window_past) < self.window_size:
            self.window_past.append(data_point)
            return False

        # Fill recent window second
        if len(self.window_recent) < self.window_size:
            self.window_recent.append(data_point)
            return False

        # Now we have both windows full, compute test statistic
        x = list(self.window_past)
        y = list(self.window_recent)
        stat = self._compute_statistic(x, y)

        if stat > self.threshold:
            self.changepoints.append(self.t)
            # Reset windows after change
            self.window_past.clear()
            self.window_recent.clear()
            self.window_past.append(data_point)
            return True

        # Slide windows
        self.window_past.append(self.window_recent.popleft())
        self.window_recent.append(data_point)

        return False