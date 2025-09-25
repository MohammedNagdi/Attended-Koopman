# data/data_handlers.py

import os
import torch
import pandas as pd
import numpy as np
import tqdm
from torch.utils.data import TensorDataset, DataLoader
from utils.data_loader import data_from_name

class KoopmanDataHandler:
    """Handles data preprocessing and dataloader generation for Koopman models."""
    def __init__(self, config):
        # Extract configuration parameters
        self.config = config
        self.data_dir = config.get("data_dir", os.path.join(os.getcwd(), 'data'))
        self.dataset_name = config.get("dataset", "simple")
        self.train_size = config.get("train_size", 0.8)
        self.val_size = config.get("val_size", 0.1)
        self.batch_size = config.get("batch_size", 32)
        self.noise = config.get("noise", 0.0)
        self.orthogonal_projection = config.get("orthogonal_projection", False)
        self.num_combinations = config.get("num_combinations", 100)
        self.num_samples = config.get("num_samples", 10)
        self.time_steps = config.get("time_steps", 100)
        self.max_time = config.get("max_time", 10)
        self.normalize = config.get("normalize", 'scale')
        self.device = config.get("device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.prediction_length = config.get("prediction_length", 10)
        self.stride = config.get("stride", None)
        self.data_parameters = config.get("data_parameters", None)
        self.shuffle = config.get("shuffle", True)

        # default parameters depending on the dataset
        default_params = {
            "isolated_repressilator": [0.03, 10, 40, 2, 0.3466, 0.0693, 10],
            "duffing_oscillator": [0, 1.0, -1.0, 0, 1.2],
            "goodwin_oscillator": [360, 43, 1.0, 12, 0.6, 1.0, 1.0, 1.0, 0.8],
            "Lorenz": [10.0, 28.0, 2.666],
            "LotkaVolterra" : [0.2, 0.2, 0.2, 0.2],
            "IRMA": {
            'alpha': [0, 1.49E-4, 3E-3, 7.4E-4, 6.1E-4],
            'v': [0.04, 0.026, 0.02, 0.014, 0.018],
            'k': [3.5E-4, 3.7E-2, 0.01, 1.884, 4.77E-2],
            'h': [1, 4, 4, 1, 4, 4],
            'd': [0.022, 0.047, 0.421, 0.098, 0.05],
            'gamma': 0.6
        },
            "Rossler": [0.2, 0.2, 5.7],
            "FluidFlow": {
            "mu": 0.1,
            "omega": 1.0,
            "A": -0.1,
            "lambda": 10.0
        } 
        }
        
        if self.data_parameters is None:
            if self.dataset_name in default_params:
                self.data_parameters = default_params[self.dataset_name]
            else:
                self.data_parameters = None

        
        # Initialize data attributes
        self.raw_data = None
        self.preprocessed_data = None
        self.Xscale = 1.0
        
    def load_data(self):
        """Load data based on the specified dataset name."""
        # Determine filename based on dataset
        if self.dataset_name == "discrete_spectrum":
            filename = 'discrete_spectrum.pkl'
        elif self.dataset_name == "pendulum":
            # pendulum_{num_samples}_{time_points}_{time_intervals}.pkl
            filename = f'pendulum_{self.num_samples}_{self.time_steps}_{self.max_time}.pkl'
        elif self.dataset_name == "simple":
            filename = 'simple.pkl'
        elif self.dataset_name == "isolated_repressilator":
            filename = f'isolated_repressilator_{self.num_combinations}_{self.num_samples}_{self.time_steps}_{self.max_time}_param_{self.data_parameters}.pkl'
        elif self.dataset_name == "duffing_oscillator":
            filename = f'duffing_oscillator_{self.num_combinations}_{self.num_samples}_{self.time_steps}_{self.max_time}_param_{self.data_parameters}.pkl'
        elif self.dataset_name == "goodwin_oscillator":
            filename = f'goodwin_oscillator_{self.num_combinations}_{self.num_samples}_{self.time_steps}_{self.max_time}_param_{self.data_parameters}.pkl'
        elif self.dataset_name == "Lorenz":
            filename = f'lorenz_{self.num_combinations}_{self.num_samples}_{self.time_steps}_{self.max_time}_param_{self.data_parameters}.pkl'
        elif self.dataset_name == "LotkaVolterra":
            filename = f'lotka_volterra_{self.num_combinations}_{self.num_samples}_{self.time_steps}_{self.max_time}_param_{self.data_parameters}.pkl'
        elif self.dataset_name == "IRMA":
            filename = f"irma_{self.num_combinations}_{self.num_samples}_{self.time_steps}_{self.max_time}.pkl"
        elif self.dataset_name == "Rossler":
            filename = f'rossler_{self.num_combinations}_{self.num_samples}_{self.time_steps}_{self.max_time}_param_{self.data_parameters}.pkl'
        elif self.dataset_name == "FluidFlow":
            filename = f'fluid_flow_{self.num_combinations}_{self.num_samples}_{self.time_steps}_{self.max_time}_param_{self.data_parameters}.pkl'
        elif self.dataset_name == "host_aware_repressilator":
            # Special case: load directly from CSV
            df = pd.read_csv("./results_perturbation_joint_induction_binding_rate.csv")
            
            data = pd.DataFrame()
            for idx in range(df.shape[0]):
                sol_df = pd.DataFrame()
                for col in df.columns:
                    temp_list = [float(elm) for elm in df.iloc[idx][col][1:-1].split(",")]
                    sol_df[col] = temp_list
                    sol_df.index = [idx] * len(temp_list)
                data = pd.concat([data, sol_df])
            
            self.raw_data = data
            return self.raw_data
        else:
            # Generic dataset handling
            print('Loading data...')
            
            X, Xclean, m, n = data_from_name(
                self.dataset_name, 
                noise=self.noise, 
                theta=self.config.get("theta", None),
                orthogonal_project=self.orthogonal_projection
            )
            self.raw_data = {"X": X, "Xclean": Xclean, "m": m, "n": n}
            return self.raw_data
        
        # Load or generate data for standard datasets
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.isfile(filepath):
            print('Generating data...')
            kwargs = {}
            if self.dataset_name in ["isolated_repressilator", "duffing_oscillator", "goodwin_oscillator", "Lorenz", "LotkaVolterra", "pendulum", "IRMA", "Rossler", "FluidFlow"]:
                kwargs = {
                    "combi_n": self.num_combinations,
                    "combi_n_samples": self.num_samples,
                    "time_points": self.time_steps,
                    "time_intervals": self.max_time,
                    "data_parameters": self.data_parameters
                }
            data = data_from_name(self.dataset_name, orthogonal_project=self.orthogonal_projection, **kwargs)

            # Ensure directory exists and save data
            os.makedirs(self.data_dir, exist_ok=True)
            pd.to_pickle(data, filepath)
        else:
            data = pd.read_pickle(filepath)

        data = self.discrete_data_format(data)
        print(f"the shape of the data is {data.shape}")
            
        self.raw_data = data
        return self.raw_data
    
    def discrete_data_format(self,df,chunk_size=1):
        X = []
        for i in tqdm.tqdm(df.index.unique()):
            x = torch.FloatTensor(df.loc[i].values)
            size = x.shape[0]
            if chunk_size > 1:
                size = int(size/chunk_size)
            x = torch.chunk(x,chunk_size)
            X.extend(x)
        X = torch.stack(X, 0)
        return X
        
    def preprocess(self):
        """Preprocess the loaded data."""
        if self.raw_data is None:
            self.load_data()
            
        print('Processing dataset:', self.dataset_name)
        
        # Split data into train/val/test sets
        train_idx, val_idx = self._get_split_indices(self.raw_data.shape[0])
        print(f"train_idx: {train_idx}, val_idx: {val_idx}")
        Xtrain = self.raw_data[:train_idx]
        Xval = self.raw_data[train_idx:val_idx]
        Xtest = self.raw_data[val_idx:]
        
        # Convert to tensors
        Xtrain = torch.tensor(Xtrain, dtype=torch.float32).to(self.device)
        Xval = torch.tensor(Xval, dtype=torch.float32).to(self.device)
        Xtest = torch.tensor(Xtest, dtype=torch.float32).to(self.device)

        # Apply normalization if configured
        if self.normalize == 'scale':
            self.Xscale = torch.max(torch.abs(Xtrain)).item()
            Xtrain = Xtrain / self.Xscale
            Xval = Xval / self.Xscale
            Xtest = Xtest / self.Xscale
        elif self.normalize == 'standard':
            self.Xmean = torch.mean(Xtrain).item()
            self.Xscale = torch.std(Xtrain).item()
            Xtrain = (Xtrain - self.Xmean) / self.Xscale
            Xval = (Xval - self.Xmean) / self.Xscale
            Xtest = (Xtest - self.Xmean) / self.Xscale
        elif self.normalize == 'minmax':
            self.Xmin = torch.min(Xtrain).item()
            self.Xmax = torch.max(Xtrain).item()
            Xtrain = (Xtrain - self.Xmin) / (self.Xmax - self.Xmin)
            Xval = (Xval - self.Xmin) / (self.Xmax - self.Xmin)
            Xtest = (Xtest - self.Xmin) / (self.Xmax - self.Xmin)
        elif self.normalize == 'max_per_dim':
            print(f" shape of Xtrain is {Xtrain.shape}")
            # Flatten all but last dimension, then take max per feature
            self.Xscale = torch.amax(Xtrain, dim=(0, 1))
            print(f"Xmax: {self.Xscale}")
            Xtrain = Xtrain / self.Xscale
            Xval = Xval / self.Xscale
            Xtest = Xtest / self.Xscale
        
        print(f'Processed data shapes - Train: {Xtrain.shape}, Val: {Xval.shape}, Test: {Xtest.shape}')
        
        # Store processed data
        self.preprocessed_data = {
            "Xtrain": Xtrain,
            "Xval": Xval,
            "Xtest": Xtest
        }
        
        return self.preprocessed_data
    
    @staticmethod
    def chunk_trajectory_data(data, prediction_length, stride=None):
        """Split trajectory data into chunks for training."""
        if len(data.shape) == 2:  # Handle 2D data
            num_samples, state_dim = data.shape
            # Reshape to simulate trajectory data
            data = data.reshape(1, num_samples, state_dim)
        
        num_trajectories, timesteps, state_dim = data.shape
        chunk_size = prediction_length + 1
        
        if chunk_size > timesteps:
            raise ValueError("Chunk size cannot be greater than number of timesteps")
        
        stride = prediction_length if stride is None else stride
        chunks_per_traj = (timesteps - chunk_size) // stride + 1
        total_chunks = num_trajectories * chunks_per_traj
        
        chunks = np.zeros((total_chunks, chunk_size, state_dim))
        
        chunk_idx = 0
        for traj in range(num_trajectories):
            for start in range(0, timesteps - chunk_size + 1, stride):
                chunks[chunk_idx] = data[traj, start:start + chunk_size]
                chunk_idx += 1
                
        return chunks
    
    def _get_split_indices(self, num_samples):
        """Get indices for train/val/test split."""
        train_idx = int(num_samples * self.train_size)
        val_idx = int(num_samples * (self.train_size + self.val_size))
        return train_idx, val_idx
    
    def create_dataloaders(self):
        """Create PyTorch DataLoaders from datasets."""
        if self.preprocessed_data is None:
            self.preprocess()

        # Prepare data for chunking
        Xtrain = self.preprocessed_data["Xtrain"].cpu().numpy()
        Xval = self.preprocessed_data["Xval"].cpu().numpy()
        Xtest = self.preprocessed_data["Xtest"].cpu().numpy()

        # Chunk data
        train_data = self.chunk_trajectory_data(Xtrain, self.prediction_length, self.stride)
        val_data = self.chunk_trajectory_data(Xval, self.prediction_length, self.stride)
        test_data = self.chunk_trajectory_data(Xtest, self.prediction_length, self.stride)

        # Convert to PyTorch tensors and move to device
        train_tensor = torch.tensor(train_data, dtype=torch.float32).to(self.device)
        val_tensor = torch.tensor(val_data, dtype=torch.float32).to(self.device)
        test_tensor = torch.tensor(test_data, dtype=torch.float32).to(self.device)


        # Create TensorDatasets
        train_dataset = TensorDataset(train_tensor)
        val_dataset = TensorDataset(val_tensor)
        test_dataset = TensorDataset(test_tensor)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
        )
        
        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }
    