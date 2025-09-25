# biokoopman : System dynamics prediction via Attention Free Transformer and Koopman Embeddings

A new class of physics-based methods related to Koopman theory has been introduced, offering an alternative for processing nonlinear dynamics systems. Koopman theory is based on the insight that a nonlinear dynamical system can be fully encoded using an operator that describes how scalar functions propagate in time. The Koopman operator is *linear*, and thus preferable to work with in practice, as tools from linear algebra can be directly applied. The Koopman operator maps between function spaces and thus it is infinite-dimensional and can not be represented on a computer. However, most machine learning approaches hypothesize that there exists a data transformation under which an approximate finite-dimensional Koopman operator is available. Typically, this map is represented via an autoencoder network, embedding the input onto a low-dimensional latent space.

Newly Introduced Methods

1. ### Attention-Free Transformer:
    This optimized attention mechanism is applied after processing the current time step in the encoder. It integrates the latest prediction used for the Koopman operator by incorporating a problem-specific time window. The most recent point is updated through attention from surrounding points, enabling the model to correct inaccurate predictions and improve long-term forecasting accuracy.

2. ### Dynamic Reencoding:
    Inspired by periodic reencoding, which ensures that latent space predictions remain within the Koopman invariant subspace and handles challenges like switching dynamics and multiple fixed points, Dynamic Reencoding automates this process. It does so by comparing the reencoding difference of the current point against the average difference over a previous window of predicted points, allowing the model to determine when reencoding is necessary.

3. ### Architectural Variation:

    Inspired by the Kolmogorov-Arnold representation theorem, Kolmogorov-Arnold Networks (KANs) have been proposed as a promising alternative to Multi-Layer Perceptrons (MLPs). While MLps have *fixed* activation functions on *nodes* ("neurons"), KANs have *learnable* activation functions on *edges* ("weights"). KANs have no linear weights at all -- every weight parameter is replaced by a univariate function parameterized as a spline. Here, we bring together these two worlds by  adding the ability to use either MLP or KANs as the backbone for the Koopman autoencoder. 


## Project Structure

```
biokoopman/
├── data_handler.py              # Data loading and preprocessing utilities
├── KoopmanTrainer.py           # Base Koopman trainer implementation
├── KoopmanTrainer_AFT.py       # Koopman trainer with Attention-Free Transformer
├── KoopmanTrainer_AFT_DR.py    # AFT trainer with Dynamic Reencoding
├── KoopmanAE_optimization.py   # Koopman autoencoder optimization routines
├── run_experiment.py           # Main experiment runner script
├── run_experiment.ipynb        # Jupyter notebook for experiments
├── Project.md                  # Project description and objectives
├── requirement.txt             # Python dependencies
├── data/                       # Generated datasets directory
├── experiments/                # Experimental results and outputs
│   └── FluidFlow_AFT_DR/      # Example experiment results
├── models/                     # Saved model checkpoints
├── notebooks/                  # Example notebooks for various dynamical systems
└── utils/                      # Utility functions and helper modules
```

## Core Components

### Trainers
- [`KoopmanTrainer.py`](KoopmanTrainer.py): Base implementation of Koopman operator training
- [`KoopmanTrainer_AFT.py`](KoopmanTrainer_AFT.py): Enhanced with Attention-Free Transformer layers
- [`KoopmanTrainer_AFT_DR.py`](KoopmanTrainer_AFT_DR.py): Includes dynamic reencoding capabilities

### Data Handling
- [`data_handler.py`](data_handler.py): Utilities for loading, preprocessing, and managing dynamical system datasets

### Optimization
- [`KoopmanAE_optimization.py`](KoopmanAE_optimization.py): Autoencoder optimization routines for Koopman operators

## Usage

### Running Experiments

1. **Using Python Script**:
   ```bash
   python run_experiment.py
   ```

2. **Using Jupyter Notebook**:
   Open [`run_experiment.ipynb`](run_experiment.ipynb) for interactive experimentation

3. **Custom Training**:
   Explore the [`notebooks/`](notebooks/) directory for system-specific examples

### Data Generation

Generated datasets are automatically saved to the [`data/`](data/) directory. Example format:
```
duffing_oscillator_1_6000_201_10_param_[0, 1.0, -1.0, 0, 1.2].pkl
```

### Results

Experimental results and model outputs are stored in the [`experiments/`](experiments/) directory, organized by experiment type and configuration.

## Installation

Install required dependencies:
```bash
pip install -r requirement.txt
```