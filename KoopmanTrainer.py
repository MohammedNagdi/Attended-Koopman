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

    def _evolve(self, Y0) -> torch.Tensor:
        Ypred = torch.zeros(Y0.shape[0], self.prediction_length, Y0.shape[1], device=self.device)
        Ypred[:, 0, :] = self.model.knet(Y0.clone())
        for index in range(1, Ypred.shape[1]):
            Ypred[:, index] = self.model.knet(Ypred[:, index-1].clone())
        return Ypred

    def train(self):
        
        clip_grad_norm = self.args.gradclip if self.gradclip else None

        self.parameters = list(self.model.ae.parameters()) + list(self.model.knet.parameters())

        stats = {
            'recon_loss_tr': [], 'lin_loss_tr': [], 'pred_loss_tr': [], 'total_loss_tr': [],
            'recon_loss_va': [], 'lin_loss_va': [], 'pred_loss_va': [], 'total_loss_va': []
        }

        ES = EarlyStopping(patience=10, min_delta=0.00001) if self.early_stopping else None
        do_val = self.do_eval

        for epoch in tqdm(range(self.num_epochs), desc="Training Epochs"):
            self.model.ae.train()
            self.model.knet.train()

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

            if (epoch + 1) % 20 == 0:
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
    
    def test(self,test_loader):
        self.model.ae.eval()
        self.model.knet.eval()

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

    def train_KoopmanAE(self):
        return self.train()
    
    def test_KoopmanAE(self,test_loader):
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

    def predict_new(self, X0, steps=50)->torch.Tensor:
        return self.predict_traj(X0, steps)
    
    def predict_new_Periodic_Reencoding(self, X0, steps=50, reenc=10) -> torch.Tensor:
        """
        Predict future states from a single initial condition using Koopman operator.
        Periodic reencoding of the latent state every `reenc` steps.
        """
        self.model.eval()
        
        if not isinstance(X0, torch.Tensor):
            X0 = torch.tensor(X0, device=self.device).unsqueeze(0)
        
        Xpred = torch.zeros((steps+1, *X0.shape), device=self.device)
        Xpred[0, :] = X0

        with torch.no_grad():
            Yencoded = self.model.ae.encoder(X0)
            
            for index in range(1, steps+1):
                Ypred = self.model.knet(Yencoded)
                Xpred[index, :] = self.model.ae.decoder(Ypred)

                # Periodic reencoding
                if index % reenc == 0:
                    Yencoded = self.model.ae.encoder(Xpred[index].unsqueeze(0))
                else:
                    Yencoded = Ypred

        return Xpred
    
    def predict_new_with_threshold(self, X0, threshold=0.5, steps=50) -> tuple:
        """
        Predict future states from a single initial condition using Koopman operator,
        with threshold-based reencoding in latent space (no AFT).
        
        Args:
            X0 (Tensor or array-like): Initial condition.
            threshold (float): Threshold for normalized MSE difference to trigger reencoding.
            steps (int): Number of prediction steps.

        Returns:
            tuple: (Xpred, Y_history, reenc_points)
                - Xpred: Predicted trajectory in observation space.
                - Y_history: List of latent states.
                - reenc_points: Time steps where reencoding occurred.
        """
        self.model.eval()

        if not isinstance(X0, torch.Tensor):
            X0 = torch.tensor(X0, device=self.device).unsqueeze(0)
        
        Xpred = torch.zeros((steps+1, *X0.shape), device=self.device)
        Xpred[0, :] = X0

        reenc_points = []

        with torch.no_grad():
            Yencoded = self.model.ae.encoder(X0)
            Y_history = [Yencoded.squeeze(0)]

            for index in range(1, steps+1):
                Ypred = self.model.knet(Yencoded)
                Xpred[index, :] = self.model.ae.decoder(Ypred)

                # Reencoding check
                Y_decoded = self.model.ae.decoder(Yencoded)
                Y_encoded_new = self.model.ae.encoder(Y_decoded)

                Ypred_after = self.model.knet(Y_encoded_new)

                # Normalized MSE difference
                mse_diff = torch.mean((Ypred_after - Ypred) ** 2)
                norm = torch.mean(Ypred ** 2) + 1e-8
                normalized_diff = mse_diff / norm

                if normalized_diff > threshold:
                    Yencoded = Y_encoded_new
                    reenc_points.append(index)
                    Ypred = Ypred_after

                Y_history.append(Ypred.squeeze(0))
                Yencoded = Ypred  # Continue prediction

        return Xpred, Y_history, reenc_points

    def predict_window_variance(self, X0, window_size=10, variance_threshold=2.0, steps=50):
        """
        Predict future states from a single initial condition using Koopman operator,
        with reencoding triggered by high MSE variance in a sliding window.
        
        Args:
            X0 (Tensor or array-like): Initial condition.
            window_size (int): Sliding window size for variance check.
            variance_threshold (float): Reencode if current MSE > mean + threshold * std.
            steps (int): Number of prediction steps.
        
        Returns:
            tuple: (Xpred, Y_history, reenc_points, mse_history)
        """
        self.model.eval()

        if not isinstance(X0, torch.Tensor):
            X0 = torch.tensor(X0, device=self.device).unsqueeze(0)
        
        Xpred = torch.zeros((steps+1, *X0.shape), device=self.device)
        Xpred[0] = X0

        reenc_points = []
        mse_history = []

        with torch.no_grad():
            Yencoded = self.model.ae.encoder(X0)
            Y_history = [Yencoded.squeeze(0)]

            for index in range(1, steps+1):
                # Predict normally
                Ypred = self.model.knet(Yencoded)
                Xpred[index] = self.model.ae.decoder(Ypred).squeeze(0)

                # Reencode: decode & encode again
                Y_decoded = self.model.ae.decoder(Yencoded)
                Y_reencoded = self.model.ae.encoder(Y_decoded)
                Ypred_reenc = self.model.knet(Y_reencoded)

                # Compute MSE difference
                mse_diff = torch.mean((Ypred_reenc - Ypred) ** 2).item()
                mse_history.append(mse_diff)

                # Variance-based reencoding trigger
                use_reenc = False
                if len(mse_history) >= window_size:
                    window_data = mse_history[-window_size:]
                    mean = sum(window_data) / window_size
                    var = sum((x - mean) ** 2 for x in window_data) / window_size
                    std = var ** 0.5

                    if mse_diff > mean + variance_threshold * std:
                        use_reenc = True
                        reenc_points.append(index)

                # Update latent state
                if use_reenc:
                    Yencoded = Y_reencoded
                else:
                    Yencoded = Ypred

                Y_history.append(Yencoded.squeeze(0))

        return Xpred, Y_history, reenc_points, mse_history

    def predict_new_window_variance(self, X0, window_size=10, variance_threshold=2.0, steps=50):
        """
        Predict future states using Koopman operator with reencoding triggered by high
        MSE variance in a sliding window. No attention or AFT layers are used.
        
        Args:
            X0 (Tensor or array-like): Initial condition.
            window_size (int): Sliding window size for MSE variance check.
            variance_threshold (float): Reencode if current MSE > mean + threshold * std.
            steps (int): Number of prediction steps.
        
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
        mse_history = []

        with torch.no_grad():
            # Initial encoding
            Y0 = self.model.ae.encoder(X0_batch)
            latent_dim = Y0.shape[-1]

            Ypred = torch.zeros((steps+1, latent_dim), device=self.device)
            Ypred[0] = Y0.squeeze(0)
            Y_history = [Y0.squeeze(0)]

            # First prediction step
            Y_next = self.model.knet(Y0)
            Xpred[1] = self.model.ae.decoder(Y_next).squeeze(0)
            Ypred[1] = Y_next.squeeze(0)
            Y_history.append(Y_next.squeeze(0))

            # First MSE diff
            Y_decoded = self.model.ae.decoder(Y_next)
            Y_reenc = self.model.ae.encoder(Y_decoded)
            Y_reenc_pred = self.model.knet(Y_reenc)
            mse_diff = torch.mean((Y_reenc_pred - Y_next) ** 2).item()
            mse_history.append(mse_diff)

            for t in range(2, steps+1):
                # Prediction step
                Y_next = self.model.knet(Y_history[-1].unsqueeze(0))
                Xpred[t] = self.model.ae.decoder(Y_next).squeeze(0)
                Ypred[t] = Y_next.squeeze(0)

                # Reencode candidate
                Y_decoded = self.model.ae.decoder(Y_next)
                Y_reenc = self.model.ae.encoder(Y_decoded)
                Y_reenc_pred = self.model.knet(Y_reenc)

                current_mse_diff = torch.mean((Y_reenc_pred - Y_next) ** 2).item()
                mse_history.append(current_mse_diff)

                # Variance-based reencoding
                if len(mse_history) >= window_size:
                    window_data = mse_history[-window_size:]
                    window_mean = sum(window_data) / len(window_data)
                    window_var = sum((x - window_mean) ** 2 for x in window_data) / len(window_data)
                    window_std = window_var ** 0.5

                    if current_mse_diff > window_mean + variance_threshold * window_std:
                        Y_next = self.model.ae.encoder(Xpred[t].unsqueeze(0))
                        Ypred[t] = Y_next.squeeze(0)
                        reenc_points.append(t)

                Y_history.append(Y_next.squeeze(0))

        return Xpred, Ypred, reenc_points, mse_history
    
    def reconstruct_traj(self, traj) -> torch.Tensor:
        Y, reconstructed_traj = self.model.ae(traj)
        return reconstructed_traj

    def predict_traj(self, X0, steps=50)->torch.Tensor:
        self.model.eval()
        if not isinstance(X0, torch.Tensor):
            X0 = torch.tensor(X0, device=self.device).unsqueeze(0)
            
        Xpred = torch.zeros((steps+1, *X0.shape), device=self.device)
        Xpred[0, :] = X0
        
        with torch.no_grad():
            Yencoded = self.model.ae.encoder(X0)
            for index in range(1, steps+1):
                Ypred = self.model.knet(Yencoded)
                Xpred[index, :] = self.model.ae.decoder(Ypred)
                Yencoded = Ypred
        return Xpred
    def predict_with_twosample(self, X0, threshold=3.1, window_size=20, statistic="Lepage", steps=50) -> tuple:
        """
        Predict future states using Koopman operator with reencoding triggered 
        by two-sample statistical tests (no AFT).
        
        Args:
            X0 (Tensor or array-like): Initial condition.
            threshold (float): Test statistic threshold for change detection.
            window_size (int): Sliding window size for two-sample test.
            statistic (str): Type of test statistic to use (default: "Lepage").
            steps (int): Number of prediction steps.

        Returns:
            tuple: (Xpred, Y_history, reenc_points)
                - Xpred: Predicted trajectory in observation space.
                - Y_history: List of latent states.
                - reenc_points: Time steps where reencoding occurred.
        """
        self.model.eval()
        if not isinstance(X0, torch.Tensor):
            X0 = torch.tensor(X0, device=self.device).unsqueeze(0)

        Xpred = torch.zeros((steps+1, *X0.shape), device=self.device)
        Xpred[0, :] = X0
        reenc_points = []

        detector = TwoSample(statistic=statistic, threshold=threshold, window_size=window_size)

        with torch.no_grad():
            Yencoded = self.model.ae.encoder(X0)
            Y_history = [Yencoded.squeeze(0)]

            for t in range(1, steps+1):
                Ypred = self.model.knet(Yencoded)
                Xpred[t, :] = self.model.ae.decoder(Ypred)

                # Decode and re-encode for consistency check
                Y_decoded = self.model.ae.decoder(Yencoded)
                Y_encoded_new = self.model.ae.encoder(Y_decoded)

                Ypred_after = self.model.knet(Y_encoded_new)

                # Compute discrepancy (like in AFT version, but without aft layer)
                mse_diff = torch.mean((Ypred_after - Ypred) ** 2)

                # Pass into two-sample detector
                is_changepoint = detector.process_one(float(mse_diff))

                if is_changepoint:
                    Yencoded = Y_encoded_new
                    Ypred = Ypred_after
                    reenc_points.append(t)

                Y_history.append(Ypred.squeeze(0))
                Yencoded = Ypred  # continue from predicted latent state

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