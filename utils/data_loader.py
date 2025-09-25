import os
import pickle
import random
import math
import tqdm
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import ellipj, ellipk
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import torch

#******************************************************************************
# Read in data util functions
#******************************************************************************
def data_from_name(name, combi_n = 1, combi_n_samples = 5000, time_points = 50, time_intervals = 10, noise = 0.0, theta=2.4,data_parameters=[0.03, 10, 40, 2, 0.3466, 0.1933, 10], orthogonal_project=False,path = 'data/'):
    """
    Retrieve dataset based on the provided name.

    This function loads or generates datasets for various dynamical systems based on the given name.
    Each dataset is configured with specific parameters and can be customized based on the input arguments.

    Parameters:
    ----------
    name : str
        The name of the dataset to retrieve. Options include:
        - 'simple': Simple dynamical system
        - 'pendulum': Nonlinear pendulum system
        - 'discrete_spectrum': System with discrete eigenvalue spectrum
        - 'isolated_repressilator': Gene regulatory network model
        - 'duffing_oscillator': Duffing oscillator system
    combi_n : int, optional
        Number of combinations of parameters of the ODE for applicable dynamical systems, default is 50
    combi_n_samples : int, optional
        Number of initial conditions per combination for applicable datasets, default is 100
    time_points : int, optional
        Number of time points for time series data, default is 50, that will be the number of timesteps for np.linspace(0,time_intervals,time_points)
    time_intervals : int, optional
        Number of time intervals for sampling, default is 10 that will be from (0,10)
    noise : float, optional
        Noise level to add to the data, default is 0.0
    theta : float, optional
        Parameter for pendulum systems, default is 2.4
    orthogonal_project : bool, optional
        Whether to apply orthogonal projection to the data, default is False
    data_parameters : list, optional
        Parameters for the system
    path : str, optional
        Directory path to save/load datasets, default is 'data/'

    Returns:
    -------
    tuple
        Dataset in the format specific to the dynamical system requested, 
        typically containing state variables, time derivatives, and time points
        (number of trajecties, number of time points, number of state variables)

    Raises:
    ------
    ValueError
        If the specified dataset name is not recognized
    """
    path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        os.makedirs(path)
    if name == 'pendulum':
        return generate_pendulum_data(num_samples=combi_n_samples, time_points=time_points, time_intervals=time_intervals, path=path)
    elif name == 'simple':
        return simple()      
    elif name == 'discrete_spectrum':
        x1range = [-0.5, 0.5]
        x2range = [-0.5, 0.5]
        tSpan = np.linspace(0, 1, 51)
        mu = -0.05
        lamda = -1
        numICs = combi_n_samples
        seed = 42
        return DiscreteSpectrumExampleFn(x1range, x2range, numICs, tSpan, mu, lamda, seed, path=path)
    elif name == "isolated_repressilator":
        return isolated_repressilator_fn(n=combi_n, num_samples=combi_n_samples, time_points=time_points, time_intervals=time_intervals,data_parameters=data_parameters, path=path)
    elif name == "duffing_oscillator":
        return duffing_oscillator_fn(n=combi_n, num_samples=combi_n_samples, time_points=time_points, time_intervals=time_intervals,data_parameters=data_parameters, path=path)
    elif name == "goodwin_oscillator":
        return goodwin_oscillator_fn(n=combi_n, num_samples=combi_n_samples, time_points=time_points, time_intervals=time_intervals,data_parameters=data_parameters, path=path)
    elif name == "Lorenz":
        return lorenz_fn(n=combi_n, num_samples=combi_n_samples, time_points=time_points, time_intervals=time_intervals,data_parameters=data_parameters, path=path)
    elif name == "LotkaVolterra":
        return lotka_volterra_fn(n=combi_n, num_samples=combi_n_samples, time_points=time_points, time_intervals=time_intervals,data_parameters=data_parameters, path=path) 
    elif name == "IRMA":
        return irma_fn(n=combi_n, num_samples=combi_n_samples, time_points=time_points, time_intervals=time_intervals,data_parameters=data_parameters, path=path)
    elif name == "Rossler":
        return rossler_dataset_fn(n=combi_n, num_samples=combi_n_samples, time_points=time_points, time_intervals=time_intervals,data_parameters=data_parameters, path=path)
    elif name == "FluidFlow":
        return fluidflow_fn(n=combi_n, num_samples=combi_n_samples, time_points=time_points, time_intervals=time_intervals,data_parameters=data_parameters, path=path)
    else:
        raise ValueError('dataset {} not recognized'.format(name))
    
#******************************************************************************
# Data processing functions
#******************************************************************************


def rotate_scale(samples_array, dims = (64,2)):

    # Rotate to high-dimensional space and scale

    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    # rotate
    samples_array = samples_array.T.dot(Q.T)        
    
    # scale 
    samples_array = 2 * (samples_array - np.min(samples_array)) / np.ptp(samples_array) - 1

    return samples_array



#******************************************************************************
# Dynamical Systems functions -- the actual generators
#******************************************************************************

#******************************************************************************
# Pendulum system
###******************************************************************************
class PendulumModel:
    """
    Simple pendulum model with damping.

    ODE system:
        dtheta/dt = omega
        domega/dt = -(g / l) * sin(theta) - b * omega

    Parameters
    ----------
    y0 : array-like
        Initial state of the system [theta (angle in radians), omega (angular velocity)]
    g : float
        Gravitational constant (default: 9.81 m/s²)
    l : float
        Length of the pendulum (default: 1.0 m)
    b : float
        Damping coefficient (default: 0.2)
    """

    def __init__(self, y0=None, g=9.81, l=1.0, b=0.2):
        if y0 is None:
            # Default: 10 degree offset from inverted position => pi ± 10°
            self._y0 = np.array([np.pi + np.deg2rad(10), 0.0])
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 2:
                raise ValueError("Initial value must have size 2.")
        self.g = g
        self.l = l
        self.b = b

    def _rhs(self, S, t):
        theta, omega = S
        dtheta_dt = omega
        domega_dt = -(self.g / self.l) * np.sin(theta) - self.b * omega
        return [dtheta_dt, domega_dt]

    def simulate(self, times):
        return odeint(self._rhs, self._y0, times)

    def suggested_times(self):
        return np.linspace(0, 20, 1000)  # simulate for 20 seconds with more points

def generate_pendulum_data(num_samples, time_points, time_intervals, path='data/'):
    """
    Generate simulated pendulum trajectories as a NumPy array:
    shape = (num_samples, time_points, 2)

    Returns
    -------
    np.ndarray
        Array of trajectories (theta, omega)
    """
    times = np.linspace(0, time_intervals, time_points)
    

    data = pd.DataFrame()

    for idx in range(num_samples):
        theta0 = np.random.uniform(-np.pi, np.pi)
        omega0 = 0.0
        model = PendulumModel(y0=[theta0, omega0], g=1, l=1.0, b=0)
        sol = model.simulate(times)  # shape = (time_points, 2)
        df = pd.DataFrame(sol)
        df.index = [idx]*len(times)
        data = pd.concat([data, df])

        

    os.makedirs(path, exist_ok=True)
    filename = f'pendulum_{num_samples}_{time_points}_{time_intervals}.pkl'
    np.save(filename, data)

    return data




#******************************************************************************
# Simple  system
#******************************************************************************
def simple():
    """
    Generates a dataset for a simple linear dynamical system.
    The system is defined by the linear differential equation dx/dt = A*x where A is a 3x3 matrix:
    A = [
    ]
    The function:
    1. Creates 500 random initial conditions in the range [-1, 1]
    2. Simulates each initial condition for 51 time steps from t=0 to t=10
    3. Combines all trajectories into a single pandas DataFrame
    4. Saves the DataFrame to a pickle file at 'data/simple.pkl'
    Returns:
        pandas.DataFrame: A DataFrame containing all simulated trajectories.
            Each row is indexed by the initial condition number,
            and columns represent the three state variables.
    """
    A = np.array([

    [0, -1, 0],    # First row

    [1, 0, 0],     # Second row

    [0, 0, -1]     # Third row - stable pole

])
    def system(state, t):

        return A @ state
    
    t = np.linspace(0, 10, 51)

    initial_conditions = (2 * np.random.rand(500, 3) - 1).tolist()

    numICs = len(initial_conditions)

    data = pd.DataFrame()

    for i in range(numICs):

        sol = odeint(system, initial_conditions[i], t)

        df = pd.DataFrame(sol)

        df.index = [i]*len(t)

        data = pd.concat([data, df])
        
    data.to_pickle(os.path.join('data/', 'simple.pkl'))

    return data


#******************************************************************************
# Discrete Spectrum
#******************************************************************************
class DiscreteSpectrum():

    def __init__(self, y0=None):
        super(DiscreteSpectrum, self).__init__()

        # Check initial values
        if y0 is None:

            self._y0 = np.array([1,1])

        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 2:
                raise ValueError('Initial value must have size 2.')

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return 2

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 2

    def _rhs(self, y, t, mu, lamda):
        """
        Calculates the model RHS.
        """
        dy = np.zeros(2)
        dy[0] = mu * y[0]
        dy[1] = lamda*(y[1] - y[0]**2)

        return dy

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        mu, lamda = parameters
        y = odeint(self._rhs, self._y0, times, (mu, lamda))
        return y[:, :]


    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        # Toni et al.:
        return np.array([-0.05, -1])


    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """
        # Toni et al.:
        return np.linspace(0, 1, 51)

def DiscreteSpectrumExampleFn(x1range, x2range, numICs, tSpan, mu, lamda, seed, path = 'data/'):

    # try some initial conditions for x1, x2
    random.seed(seed)

    # randomly start from x1range(1) to x1range(2)
    x1 = np.random.uniform(x1range[0], x1range[1], numICs)

    # randomly start from x2range(1) to x2range(2)
    x2 = np.random.uniform(x2range[0], x2range[1], numICs)

    lenT = len(tSpan)

    # make an empty dataframe
    data = pd.DataFrame()

    count = 1
    for j in range(numICs):
        # x1 and x2 are the initial conditions
        y1 = x1[j]
        y2 = x2[j]
        y0 = np.array([y1, y2])
        model = DiscreteSpectrum(y0=y0)
        xhat = model.simulate([mu, lamda], tSpan)
        # make xhat into pandas dataframe without column names
        xhat = pd.DataFrame(xhat)
        # make the data have the index 
        xhat.index = [j]*lenT
        # append the data
        data = pd.concat([data, xhat])
    print('The shape of the dataframe')
    print(data.shape)

    # save data to file as pickle
    data.to_pickle(os.path.join(path, 'discrete_spectrum.pkl'))

    return data



#******************************************************************************
# Isolated Repressilator
#******************************************************************************
# Repressilator model
class RepressilatorModel():
    """
    The "Repressilator" model describes oscillations in a network of proteins
    that suppress their own creation [1]_, [2]_.

    The formulation used here is taken from [3]_ and analysed in [4]_. It has
    three protein states (:math:`p_i`), each encoded by mRNA (:math:`m_i`).
    Once expressed, they suppress each other:

    .. math::
        \\dot{m_0} = -m_0 + \\frac{\\alpha}{1 + p_2^n} + \\alpha_0

        \\dot{m_1} = -m_1 + \\frac{\\alpha}{1 + p_0^n} + \\alpha_0

        \\dot{m_2} = -m_2 + \\frac{\\alpha}{1 + p_1^n} + \\alpha_0

        \\dot{p_0} = -\\beta (p_0 - m_0)

        \\dot{p_1} = -\\beta (p_1 - m_1)

        \\dot{p_2} = -\\beta (p_2 - m_2)

    With parameters ``alpha_0``, ``alpha``, ``beta``, and ``n``.

    Parameters
    ----------
    y0
        The system's initial state, must have 6 entries all >=0.

    References
    ----------
    .. [1] A Synthetic Oscillatory Network of Transcriptional Regulators.
          Elowitz, Leibler (2000) Nature.
          https://doi.org/10.1038/35002125

    .. [2] https://en.wikipedia.org/wiki/Repressilator

    .. [3] Dynamic models in biology. Ellner, Guckenheimer (2006) Princeton
           University Press

    .. [4] Approximate Bayesian computation scheme for parameter inference and
           model selection in dynamical systems. Toni, Welch, Strelkowa, Ipsen,
           Stumpf (2009) J. R. Soc. Interface.
           https://doi.org/10.1098/rsif.2008.0172
    """

    def __init__(self, y0=None):
        super(RepressilatorModel, self).__init__()

        # Check initial values
        if y0 is None:
            # Toni et al.:
            self._y0 = np.array([0, 0, 0, 2, 1, 3])
            # Figure 42 in book
            #self._y0 = np.array([0.2, 0.1, 0.3, 0.1, 0.4, 0.5], dtype=float)
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 6:
                raise ValueError('Initial value must have size 6.')
            if np.any(self._y0 < 0):
                raise ValueError('Initial states can not be negative.')



    def n_outputs(self):
        return 6


    def n_parameters(self):
        return 4



    def _rhs(self, y, t, alpha0, alpha, K, n, delta_m, delta_p, beta):
        """
        Calculates the model RHS.
        """
        dy = np.zeros(6)
        # equations
        dy[0] = alpha0 + alpha / (1 + (y[5]/K)**n) - delta_m * y[0]
        dy[1] = alpha0 + alpha / (1 + (y[3]/K)**n) - delta_m * y[1]
        dy[2] = alpha0 + alpha / (1 + (y[4]/K)**n) - delta_m * y[2]
        dy[3] = beta * y[0] - delta_p * y[3]
        dy[4] = beta * y[1] - delta_p * y[4]
        dy[5] = beta * y[2] - delta_p * y[5]

        return dy

    def simulate(self, parameters, times):
        alpha0, alpha, K, n, delta_m, delta_p, beta = parameters
        y = odeint(self._rhs, self._y0, times, (alpha0, alpha, K, n, delta_m, delta_p, beta))
        return y[:, :]

    def suggested_parameters(self):
        # Toni et al.:
        return np.array([0, 1, 40, 2, 0.3466, 0.1933, 10])

        # Figure 42 in book:
        #return np.array([0, 50, 0.2, 2])

    def suggested_times(self):
        # Toni et al.:
        return np.linspace(0,1200, 1000)

        # Figure 42 in book:
        #return np.linspace(0, 300, 600)

# Function to generate parameter combinations
def CombinationGenerator_repr(n):
    # Define the parameter ranges
    alpha_0_range = (0, 5)
    alpha_range = (100, 5000)
    beta_range = (0, 10)
    n_range = (0, 10)

    # Set the number of combinations
    num_combinations = n

    # Use numpy to generate the combinations with uniform distribution
    alpha_0_values = np.random.uniform(alpha_0_range[0], alpha_0_range[1], num_combinations)
    alpha_values = np.random.uniform(alpha_range[0], alpha_range[1], num_combinations)
    beta_values = np.random.uniform(beta_range[0], beta_range[1], num_combinations)
    n_values = np.random.uniform(n_range[0], n_range[1], num_combinations)

    # Combine the values into a DataFrame
    combinations = pd.DataFrame({
        'alpha_0': alpha_0_values,
        'alpha': alpha_values,
        'beta': beta_values,
        'n': n_values
    })

    # Ensure uniqueness (though with uniform random generation, collisions are highly unlikely)
    combinations = combinations.drop_duplicates()

    # If there are not enough unique combinations, regenerate until we have enough
    while len(combinations) < num_combinations:
        additional_combinations = pd.DataFrame({
            'alpha_0': np.random.uniform(alpha_0_range[0], alpha_0_range[1], num_combinations - len(combinations)),
            'alpha': np.random.uniform(alpha_range[0], alpha_range[1], num_combinations - len(combinations)),
            'beta': np.random.uniform(beta_range[0], beta_range[1], num_combinations - len(combinations)),
            'n': np.random.uniform(n_range[0], n_range[1], num_combinations - len(combinations))
        })
        combinations = pd.concat([combinations, additional_combinations]).drop_duplicates()

    # Ensure we have exactly the desired number of unique combinations
    combinations = combinations.head(num_combinations)

    return combinations

# simulator function that takes the combination and the number of samples per combination and the suggested times
def simulator_repr(combination,y,times=None):
    # If times is not provided, use the suggested times
    if times is None:
        times = RepressilatorModel().suggested_times()
    
    # Create the model
    model = RepressilatorModel(y0=y)
    # Simulate the model
    data = model.simulate(combination, times)
    
    # Add noise to the data
    #noise = np.random.normal(0, 0.1, data.shape)
    #data += noise
    # Return the data
    return data

# combination & number of samples
def generate_data_repr(n, num_samples,data_parameters):
    # Generate the combinations
    if n == 1:
        combinations = pd.DataFrame([data_parameters])
    else:
        combinations = CombinationGenerator_repr(n)
    # transform combinations dataframe into a list of list
    combinations = combinations.values.tolist()
    # initial conditions
    if num_samples == 1:
        y = np.array([0, 0, 0, 2, 1, 3]).tolist()
    else:
        y = []
        for i in range(num_samples):
            y0 = np.random.uniform(0, 100, 6)
            y0 = y0.tolist()
            y.append(y0)
    
    # for each combination make a tuple of each combination and each member of y
    data = []
    for combination in combinations:
        for y0 in y:
            data.append((combination, y0))
    
    # shuffle the data
    np.random.shuffle(data)
    return data

def isolated_repressilator_fn(n, num_samples, time_points, time_intervals,data_parameters, path = 'data/'):
    # generate time points
    times = np.linspace(0, time_intervals, time_points)
    # generate parameter and initial values combinations
    combinations = generate_data_repr(n, num_samples,data_parameters)
    dataset = pd.DataFrame()
    for idx, combi in enumerate(combinations):
        # call the ODE system solver
        sol = simulator_repr(combi[0], combi[1], times)
        # pass sol object into a dataframe
        sol_df = pd.DataFrame(sol)
        sol_df.index = [idx]*len(times)
        # concatenate existing dataframes
        dataset = pd.concat([dataset, sol_df])
    
    # save data to file as pickle the name should be isolated_repressilator_nparameters_num_smaples_time_points_time_intervals.pkl
    dataset.to_pickle(os.path.join(path, 'isolated_repressilator_{}_{}_{}_{}_param_{}.pkl'.format(n, num_samples, time_points, time_intervals,data_parameters)))

    return dataset


#******************************************************************************
# Duffing Oscillator
#******************************************************************************
class ForcedDuffingOscillator:
    """
    The Duffing oscillator is a nonlinear dynamic system characterized by a second-order differential equation
    that includes both linear and nonlinear restoring forces, described in [1] [2].

    \frac{d^2 x[0]}{dt^2} + \delta \frac{dx[0]}{dt} + \beta x[0]^3 + \alpha x[0] = \gamma \cos(\omega t)

    where delta is the damping force, beta is the nonlinear restoring force, alpha is the spring constant, 
    gamma is the amplitude of the driving force, and omega is the angular frequency.

    \dot{y[0]} = y[1]
    \dot{y[1]} = -\delta y[1] - \beta y[0]^3 - \alpha y[0] + \gamma \cos(\omega t)

    Parameters
    ----------
    y0
        The system's initial state, must have 2 entries all >=0.

    delta : float
        The damping force
    beta : float
        The nonlinear restoring force
    alpha : float
        The spring constant
    gamma : float
        The amplitude of the driving force
    omega : float
        The angular frequency

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Duffing_equation
    .. [2] State Identification of Duffing Oscillator Based on Extreme Learning Machine
           https://ieeexplore.ieee.org/abstract/document/8080207
    """

    def __init__(self, y0=None):
        # Check initial values
        if y0 is None:
            self._y0 = np.array([0, 0])
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 2:
                raise ValueError('Initial value must have size 2.')
            #if np.any(self._y0 < 0):
            #    raise ValueError('Initial states can not be negative.')

    def n_outputs(self):
        """Returns the number of state variables."""
        return 2

    def n_parameters(self):
        """Returns the number of parameters."""
        return 5

    def _rhs(self, y, t, delta, beta, alpha, gamma, omega):
        """
        Calculates the right-hand side of the differential equations.
        
        Parameters
        ----------
        y : array_like
            Current state vector [position, velocity]
        t : float
            Current time
        delta, beta, alpha, gamma, omega : float
            System parameters
            
        Returns
        -------
        array_like
            Derivatives [dy[0]/dt, dy[1]/dt]
        """
        dy = np.zeros(2)
        dy[0] = y[1]
        dy[1] = -delta * y[1] - beta * y[0]**3 - alpha * y[0] + gamma * np.cos(omega * t)
        return dy

    def simulate(self, parameters, times):
        """
        Simulate the system for given parameters and time points.
        
        Parameters
        ----------
        parameters : array_like
            System parameters [delta, beta, alpha, gamma, omega]
        times : array_like
            Time points for simulation
            
        Returns
        -------
        array_like
            Solution array with shape (len(times), 2)
        """
        delta, beta, alpha, gamma, omega = parameters
        y = odeint(self._rhs, self._y0, times, args=(delta, beta, alpha, gamma, omega))
        return y

    def suggested_parameters(self):
        """Returns a set of example parameters that produce interesting behavior."""
        return np.array([0.2, 1.0, -1.0, 0.3, 1.2])

    def suggested_times(self):
        """Returns a suggested time array for simulation."""
        return np.linspace(0, 50, 1000)

# Function to generate parameter combinations
def CombinationGenerator_dfn(n):
    # Define the parameter ranges
    # delta, beta, alpha, gamma, omega
    delta_range = (0, 1)
    beta_range = (0, 1)
    alpha_range = (-1, 1)
    gamma_range = (0, 1)
    omega_range = (0, 2)

    # Set the number of combinations
    num_combinations = n

    # Use numpy to generate the combinations with uniform distribution
    delta_values = np.random.uniform(delta_range[0], delta_range[1], num_combinations)
    beta_values = np.random.uniform(beta_range[0], beta_range[1], num_combinations)
    alpha_values = np.random.uniform(alpha_range[0], alpha_range[1], num_combinations)
    gamma_values = np.random.uniform(gamma_range[0], gamma_range[1], num_combinations)
    omega_values = np.random.uniform(omega_range[0], omega_range[1], num_combinations)

    # Combine the values into a DataFrame
    combinations = pd.DataFrame({
        'delta': delta_values,
        'beta': beta_values,
        'alpha': alpha_values,
        'gamma': gamma_values,
        'omega': omega_values
    })

    # Ensure uniqueness (though with uniform random generation, collisions are highly unlikely)
    combinations = combinations.drop_duplicates()

    # If there are not enough unique combinations, regenerate until we have enough
    while len(combinations) < num_combinations:
        additional_combinations = pd.DataFrame({
            'delta': np.random.uniform(delta_range[0], delta_range[1], num_combinations - len(combinations)),
            'beta': np.random.uniform(beta_range[0], beta_range[1], num_combinations - len(combinations)),
            'alpha': np.random.uniform(alpha_range[0], alpha_range[1], num_combinations - len(combinations)),
            'gamma': np.random.uniform(gamma_range[0], gamma_range[1], num_combinations - len(combinations)),
            'omega': np.random.uniform(omega_range[0], omega_range[1], num_combinations - len(combinations))
        })
        combinations = pd.concat([combinations, additional_combinations]).drop_duplicates()

    # Ensure we have exactly the desired number of unique combinations
    combinations = combinations.head(num_combinations)

    return combinations

# simulator function that takes the combination and the number of samples per combination and the suggested times
def simulator_dfn(combination,y,times=None):
    # If times is not provided, use the suggested times
    if times is None:
        times = ForcedDuffingOscillator().suggested_times()
    
    # Create the model
    model = ForcedDuffingOscillator(y0=y)
    # Simulate the model
    data = model.simulate(combination, times)
    
    # Add noise to the data
    #noise = np.random.normal(0, 0.1, data.shape)
    #data += noise
    # Return the data
    return data

# combination & number of samples
def generate_data_dfn(n, num_samples,data_parameters):
    # Generate the combinations


    # this part didn't exist before, but it is needed to make the code work to get the choatic behavior 
    # and also y0 will be changed from 0 to 10 to -2,2 
    # and also I will remove the condition that data shouldn't be negative
    if n == 1:
        combinations = pd.DataFrame([data_parameters])
    else:
        combinations = CombinationGenerator_dfn(n)
    # transform combinations dataframe into a list of list
    combinations = combinations.values.tolist()
    # initial conditions
    if num_samples == 1:
        y = np.array([0, 0]).tolist()
    else:
        y = []
        for i in range(num_samples):
            y0 = np.random.uniform(-2, 2, 2)
            y0 = y0.tolist()
            y.append(y0)
    
    # for each combination make a tuple of each combination and each member of y
    data = []
    for combination in combinations:
        for y0 in y:
            data.append((combination, y0))
    
    # shuffle the data
    np.random.shuffle(data)
    return data

def duffing_oscillator_fn(n, num_samples, time_points, time_intervals,data_parameters, path = 'data/'):
    # generate time points
    times = np.linspace(0, time_intervals, time_points)
    # generate parameter and initial values combinations
    combinations = generate_data_dfn(n, num_samples,data_parameters)
    dataset = pd.DataFrame()
    for idx, combi in enumerate(combinations):
        # call the ODE system solver
        sol = simulator_dfn(combi[0], combi[1], times)
        # pass sol object into a dataframe
        sol_df = pd.DataFrame(sol)
        sol_df.index = [idx]*len(times)
        # concatenate existing dataframes
        dataset = pd.concat([dataset, sol_df])
    # save dataset
    dataset.to_pickle(os.path.join(path, 'duffing_oscillator_{}_{}_{}_{}_param_{}.pkl'.format(n, num_samples, time_points, time_intervals,data_parameters)))

    return dataset


#******************************************************************************
# Goodwin's ODE system
#******************************************************************************


class GoodwinOscillator:
    """
    Goodwin oscillator model.

    ODE system:
        dX/dt = a1 / (kappa1 + k1 * Z^n) - b1 * X
        dY/dt = alpha1 * X - beta1 * Y
        dZ/dt = gamma1 * Y - delta1 * Z

    Parameters
    ----------
    y0 : array-like
        Initial state of the system [X, Y, Z]
    """

    def __init__(self, y0=None):
        super(GoodwinOscillator, self).__init__()
        if y0 is None:
            self._y0 = np.array([1.0, 1.0, 1.0])
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 3:
                raise ValueError("Initial value must have size 3.")

    def n_outputs(self):
        return 3

    def n_parameters(self):
        return 8  # a1, kappa1, k1, n, b1, alpha1, beta1, gamma1, delta1

    def _rhs(self, S, t, a1, kappa1, k1, n, b1, alpha1, beta1, gamma1, delta1):
        """
        Right-hand side of the system.
        """
        X, Y, Z = S
        dS = np.zeros(3)
        dS[0] = a1 / (kappa1 + k1 * Z**n) - b1 * X
        dS[1] = alpha1 * X - beta1 * Y
        dS[2] = gamma1 * Y - delta1 * Z
        return dS

    def simulate(self, parameters, times):
        return odeint(self._rhs, self._y0, times, args=tuple(parameters))

    def suggested_parameters(self):
        # Sample parameters that generate oscillations
        return [360, 43, 1.0, 12, 0.6, 1.0, 1.0, 1.0, 0.8]

    def suggested_times(self):
        return np.linspace(0, 20, 200)
    

# Placeholder Combination Generator
def CombinationGenerator_gdw(n):
    """
    Placeholder function to generate parameter combinations for Goodwin oscillator.
    """
    # Define parameter ranges for each parameter
    a1_range = (100, 500)
    kappa1_range = (10, 100)
    k1_range = (0.5, 2)
    n_range = (6, 15)
    b1_range = (0.1, 1.0)
    alpha1_range = (0.5, 2.0)
    beta1_range = (0.5, 2.0)
    gamma1_range = (0.5, 2.0)
    delta1_range = (0.2, 1.0)

    combinations = pd.DataFrame({
        'a1': np.random.uniform(*a1_range, n),
        'kappa1': np.random.uniform(*kappa1_range, n),
        'k1': np.random.uniform(*k1_range, n),
        'n': np.random.uniform(*n_range, n),
        'b1': np.random.uniform(*b1_range, n),
        'alpha1': np.random.uniform(*alpha1_range, n),
        'beta1': np.random.uniform(*beta1_range, n),
        'gamma1': np.random.uniform(*gamma1_range, n),
        'delta1': np.random.uniform(*delta1_range, n),
    })

    return combinations.head(n)



# Simulator Function
def simulator_gdw(combination, y, times=None):
    if times is None:
        times = GoodwinOscillator().suggested_times()

    model = GoodwinOscillator(y0=y)
    data = model.simulate(combination, times)

    # Optional: Add noise here if needed
    return data



# Generate Parameter & Initial Condition Combinations
def generate_data_gdw(n, num_samples, data_parameters):
    if n == 1:
        combinations = pd.DataFrame([data_parameters])
    else:
        combinations = CombinationGenerator_gdw(n)

    combinations = combinations.values.tolist()

    if num_samples == 1:
        y = np.array([1.0, 1.0, 1.0]).tolist()
    else:
        y = [np.random.uniform(-2, 2, 3).tolist() for _ in range(num_samples)]

    data = []
    for combination in combinations:
        for y0 in y:
            data.append((combination, y0))

    np.random.shuffle(data)
    return data



# Dataset Generator Function
def goodwin_oscillator_fn(n, num_samples, time_points, time_intervals, data_parameters, path='data/'):
    times = np.linspace(0, time_intervals, time_points)
    combinations = generate_data_gdw(n, num_samples, data_parameters)

    dataset = pd.DataFrame()
    for idx, combi in enumerate(combinations):
        sol = simulator_gdw(combi[0], combi[1], times)
        sol_df = pd.DataFrame(sol)
        sol_df.index = [idx] * len(times)
        dataset = pd.concat([dataset, sol_df])

    os.makedirs(path, exist_ok=True)
    filename = f'goodwin_oscillator_{n}_{num_samples}_{time_points}_{time_intervals}_param_{str(data_parameters).replace(" ", "").replace(",", "_")}.pkl'
    dataset.to_pickle(os.path.join(path, filename))

    return dataset

#******************************************************************************
# The Lorenz System
#******************************************************************************

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class LorenzSystem:
    """
    The Lorenz system is a classic system of three coupled nonlinear differential
    equations originally developed to model atmospheric convection.

    It is known for its chaotic behavior and the iconic "butterfly" attractor.

    The system is defined as:

    .. math::
        \\dot{x} = \\sigma(y - x) \\\\
        \\dot{y} = x(\\rho - z) - y \\\\
        \\dot{z} = xy - \\beta z

    Parameters
    ----------
    y0 : array-like
        Initial state of the system [x, y, z].

    References
    ----------
    .. [1] Edward N. Lorenz (1963). "Deterministic Nonperiodic Flow". Journal
           of the Atmospheric Sciences.

    .. [2] https://en.wikipedia.org/wiki/Lorenz_system
    """

    def __init__(self, y0=None):
        super(LorenzSystem, self).__init__()

        if y0 is None:
            self._y0 = np.array([1.0, 1.0, 1.0])
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 3:
                raise ValueError("Initial value must have size 3.")

    def n_outputs(self):
        return 3

    def n_parameters(self):
        return 3

    def _rhs(self, y, t, sigma, rho, beta):
        """
        Calculates the right-hand side of the Lorenz system.
        """
        x, y_, z = y
        dxdt = sigma * (y_ - x)
        dydt = x * (rho - z) - y_
        dzdt = x * y_ - beta * z
        return [dxdt, dydt, dzdt]

    def simulate(self, parameters, times):
        sigma, rho, beta = parameters
        y = odeint(self._rhs, self._y0, times, args=(sigma, rho, beta))
        return y[:, :]

    def suggested_parameters(self):
        return np.array([10.0, 28.0, 8.0/3.0])

    def suggested_times(self):
        return np.linspace(0, 40, 10000)
    


# Parameter Combination Generator
def CombinationGenerator_lrz(n):
    """
    Generates `n` random parameter combinations for the Lorenz system.
    """
    sigma_range = (5, 15)
    rho_range = (20, 40)
    beta_range = (2, 3.5)

    combinations = pd.DataFrame({
        'sigma': np.random.uniform(*sigma_range, n),
        'rho': np.random.uniform(*rho_range, n),
        'beta': np.random.uniform(*beta_range, n)
    })

    return combinations.head(n)



# Simulator Function
def simulator_lrz(combination, y, times=None):
    if times is None:
        times = LorenzSystem().suggested_times()

    model = LorenzSystem(y0=y)
    return model.simulate(combination, times)



# Generate Parameter & Initial State Combinations
def generate_data_lrz(n, num_samples, data_parameters):
    if n == 1:
        combinations = pd.DataFrame([data_parameters])
    else:
        combinations = CombinationGenerator_lrz(n)

    combinations = combinations.values.tolist()

    if num_samples == 1:
        y = np.array([1.0, 1.0, 1.0]).tolist()
    else:
        y = [(np.array([0.0, 1.0, 1.05]) + np.random.normal(0, 1, 3)).tolist() for _ in range(num_samples)]

    data = []
    for combination in combinations:
        for y0 in y:
            data.append((combination, y0))

    np.random.shuffle(data)
    return data



# Dataset Generator Function
def lorenz_fn(n, num_samples, time_points, time_intervals, data_parameters, path='data/'):
    times = np.linspace(0, time_intervals, time_points)
    combinations = generate_data_lrz(n, num_samples, data_parameters)

    dataset = pd.DataFrame()
    for idx, combi in enumerate(combinations):
        sol = simulator_lrz(combi[0], combi[1], times)
        sol_df = pd.DataFrame(sol)
        sol_df.index = [idx] * len(times)
        dataset = pd.concat([dataset, sol_df])

    os.makedirs(path, exist_ok=True)
    filename = f'lorenz_{n}_{num_samples}_{time_points}_{time_intervals}_param_{str(data_parameters).replace(" ", "").replace(",", "_")}.pkl'
    dataset.to_pickle(os.path.join(path, filename))

    return dataset

#******************************************************************************
# The Lotka-Volterra System
#******************************************************************************

class LotkaVolterraSystem:
    """
    The Lotka-Volterra system models predator-prey dynamics in ecological systems.

    Equations:
        dx/dt = alpha*x - beta*x*y
        dy/dt = delta*x*y - gamma*y

    Parameters:
        alpha : prey birth rate
        beta : predation rate
        delta : predator reproduction rate
        gamma : predator death rate
    """

    def __init__(self, y0=None):
        super(LotkaVolterraSystem, self).__init__()

        if y0 is None:
            self._y0 = np.array([1.0, 1.0])
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 2:
                raise ValueError("Initial value must have size 2.")

    def n_outputs(self):
        return 2

    def n_parameters(self):
        return 4

    def _rhs(self, y, t, alpha, beta, delta, gamma):
        x, y_ = y
        dxdt = alpha * x - beta * x * y_
        dydt = delta * x * y_ - gamma * y_
        return [dxdt, dydt]

    def simulate(self, parameters, times):
        alpha, beta, delta, gamma = parameters
        y = odeint(self._rhs, self._y0, times, args=(alpha, beta, delta, gamma))
        return y[:, :]

    def suggested_parameters(self):
        return np.array([1.5, 1.0, 1.0, 3.0])

    def suggested_times(self):
        return np.linspace(0, 200, 200)
    

# simulator function
def simulator_lv(combination, y, times=None):
    if times is None:
        times = LotkaVolterraSystem().suggested_times()

    model = LotkaVolterraSystem(y0=y)
    return model.simulate(combination, times)

# Generate Parameter Combinations
def CombinationGenerator_lv(n):
    """
    Generate `n` random parameter combinations for the Lotka-Volterra system.
    """
    alpha_range = (0.5, 2.0)
    beta_range = (0.01, 1.5)
    delta_range = (0.01, 1.5)
    gamma_range = (0.5, 3.0)

    combinations = pd.DataFrame({
        'alpha': np.random.uniform(*alpha_range, n),
        'beta': np.random.uniform(*beta_range, n),
        'delta': np.random.uniform(*delta_range, n),
        'gamma': np.random.uniform(*gamma_range, n),
    })

    return combinations.head(n)


# data generation function
def generate_data_lv(n, num_samples, data_parameters):
    if n == 1:
        combinations = pd.DataFrame([data_parameters])
    else:
        combinations = CombinationGenerator_lv(n)

    combinations = combinations.values.tolist()

    if num_samples == 1:
        y = np.array([1.0, 1.0]).tolist()
    else:
        y = [np.random.uniform(0.02, 3, size=2).tolist() for _ in range(num_samples)]

    data = []
    for combination in combinations:
        for y0 in y:
            data.append((combination, y0))

    np.random.shuffle(data)
    return data


# Dataset Generator Function
def lotka_volterra_fn(n, num_samples, time_points, time_intervals, data_parameters, path='data/'):
    times = np.linspace(0, time_intervals, time_points)
    combinations = generate_data_lv(n, num_samples, data_parameters)

    dataset = pd.DataFrame()
    for idx, combi in enumerate(combinations):
        sol = simulator_lv(combi[0], combi[1], times)
        sol_df = pd.DataFrame(sol)
        sol_df.index = [idx] * len(times)
        dataset = pd.concat([dataset, sol_df])

    os.makedirs(path, exist_ok=True)
    filename = f'lotka_volterra_{n}_{num_samples}_{time_points}_{time_intervals}_param_{str(data_parameters).replace(" ", "").replace(",", "_")}.pkl'
    dataset.to_pickle(os.path.join(path, filename))

    return dataset

    



#******************************************************************************
# The IRMA System
#******************************************************************************

class IRMASystem:
    """
    The IRMA (In vivo Reverse-engineering and Modeling Assessment) synthetic gene regulatory network.

    Simulates the dynamics of five interacting genes (CBF1, GAL4, SWI5, GAL80, ASH1).
    """

    def __init__(self, y0=None):
        if y0 is None:
            self._y0 = np.random.rand(5) # CBF1, GAL4, SWI5, GAL80, ASH1
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 5:
                raise ValueError("Initial condition must have size 5.")

    def n_outputs(self):
        return 5

    def n_parameters(self):
        return 6  # alpha, v, k, h, d, gamma

    def _rhs(self, y, t, alpha, v, k, h, d, gamma):
        x1, x2, x3, x4, x5 = y

        dx1_dt = alpha[0] + v[0] * (k[0] ** h[0]) / (k[0] ** h[0] + x5 ** h[0]) - d[0] * x1
        dx2_dt = alpha[1] + v[1] * (x1 ** h[1]) / (k[1] ** h[1] + x1 ** h[1]) - d[1] * x2
        dx3_dt = alpha[2] + v[2] * (x2 ** h[2]) / (
            k[2] ** h[2] + x2 ** h[2] * (1 + (x4 ** h[5]) / (gamma ** h[5]))
        ) - d[2] * x3
        dx4_dt = alpha[3] + v[3] * (x3 ** h[3]) / (k[3] ** h[3] + x3 ** h[3]) - d[3] * x4
        dx5_dt = alpha[4] + v[4] * (x3 ** h[4]) / (k[4] ** h[4] + x3 ** h[4]) - d[4] * x5

        return [dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt]

    def simulate(self, parameters, times):
        alpha = parameters["alpha"]
        v = parameters["v"]
        k = parameters["k"]
        h = parameters["h"]
        d = parameters["d"]
        gamma = parameters["gamma"]

        result = odeint(self._rhs, self._y0, times, args=(alpha, v, k, h, d, gamma))
        return result[:, :]

    def suggested_parameters(self):
        return {
            'alpha': [0, 1.49E-4, 3E-3, 7.4E-4, 6.1E-4],
            'v': [0.04, 0.026, 0.02, 0.014, 0.018],
            'k': [3.5E-4, 3.7E-2, 0.01, 1.884, 4.77E-2],
            'h': [1, 4, 4, 1, 4, 4],
            'd': [0.022, 0.047, 0.421, 0.098, 0.05],
            'gamma': 0.6
        }

    def suggested_times(self):
        return np.linspace(0, 1500, 1500)

#******************************************************************************
# Simulator Function
#******************************************************************************

def simulator_irma(parameter_dict, y0, times=None):
    if times is None:
        times = IRMASystem().suggested_times()
    
    model = IRMASystem(y0=y0)
    return model.simulate(parameter_dict, times)

#******************************************************************************
# Parameter Combination Generator
#******************************************************************************

def CombinationGenerator_irma(n):
    """
    Generate `n` random parameter combinations for the IRMA system.
    """
    combinations = []

    for _ in range(n):
        combo = {
            'alpha': list(np.random.uniform(0, 0.005, 5)),
            'v': list(np.random.uniform(0.01, 0.05, 5)),
            'k': list(np.random.uniform(1e-4, 2.0, 5)),
            'h': [1, 4, 4, 1, 4, 4],  # Fixed values, only one source of oscillation
            'd': list(np.random.uniform(0.01, 0.5, 5)),
            'gamma': np.random.uniform(0.1, 1.0)
        }
        combinations.append(combo)

    return combinations

#******************************************************************************
# Data Generator Function
#******************************************************************************

def generate_data_irma(n, num_samples, data_parameters=None):
    if n == 1 and data_parameters is not None:
        combinations = [data_parameters]
    else:
        combinations = CombinationGenerator_irma(n)

    if num_samples == 1:
        y = np.random.rand(5).tolist()
        y_list = [y]
    else:
        y_list = [np.random.rand(5).tolist() for _ in range(num_samples)]

    data = []
    for comb in combinations:
        for y0 in y_list:
            data.append((comb, y0))

    np.random.shuffle(data)
    return data

#******************************************************************************
# Dataset Generator Function
#******************************************************************************

def irma_fn(n, num_samples, time_points, time_intervals, data_parameters=None, path='data/'):
    times = np.linspace(0, time_intervals, time_points)
    combinations = generate_data_irma(n, num_samples, data_parameters)

    dataset = pd.DataFrame()
    for idx, (params, y0) in enumerate(combinations):
        sol = simulator_irma(params, y0, times)
        sol_df = pd.DataFrame(sol)
        sol_df.index = [idx] * len(times)
        dataset = pd.concat([dataset, sol_df])

    os.makedirs(path, exist_ok=True)
    filename = f'irma_{n}_{num_samples}_{time_points}_{time_intervals}.pkl'
    dataset.to_pickle(os.path.join(path, filename))

    return dataset




#******************************************************************************
# The Rossler System
#****************************************************************************** 
class RosslerSystem:
    """
    The Rössler system is a simple model for chaos in a continuous-time dynamical system.

    Equations:
        dx/dt = -y - z
        dy/dt = x + a*y
        dz/dt = b + z*(x - c)

    Parameters:
        a, b, c : system parameters controlling the chaotic behavior
    """

    def __init__(self, y0=None):
        super(RosslerSystem, self).__init__()
        if y0 is None:
            self._y0 = np.array([1.0, 1.0, 1.0])
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 3:
                raise ValueError("Initial value must have size 3.")

    def n_outputs(self):
        return 3

    def n_parameters(self):
        return 3

    def _rhs(self, y, t, a, b, c):
        x, y_, z = y
        dxdt = -y_ - z
        dydt = x + a * y_
        dzdt = b + z * (x - c)
        return [dxdt, dydt, dzdt]

    def simulate(self, parameters, times):
        a, b, c = parameters
        y = odeint(self._rhs, self._y0, times, args=(a, b, c))
        return y[:, :]

    def suggested_parameters(self):
        return np.array([0.2, 0.2, 5.7])

    def suggested_times(self):
        return np.linspace(0, 100, 10000)


# simulator function
def simulator_rossler(combination, y, times=None):
    if times is None:
        times = RosslerSystem().suggested_times()

    model = RosslerSystem(y0=y)
    return model.simulate(combination, times)


# Generate Parameter Combinations
def CombinationGenerator_rossler(n):
    """
    Generate `n` random parameter combinations for the Rössler system.
    """
    a_range = (0.1, 0.4)
    b_range = (0.1, 0.4)
    c_range = (4.0, 10.0)

    combinations = pd.DataFrame({
        'a': np.random.uniform(*a_range, n),
        'b': np.random.uniform(*b_range, n),
        'c': np.random.uniform(*c_range, n),
    })

    return combinations.head(n)


# Data generation function
def generate_data_rossler(n, num_samples, data_parameters):
    if n == 1:
        combinations = pd.DataFrame([data_parameters])
    else:
        combinations = CombinationGenerator_rossler(n)

    combinations = combinations.values.tolist()

    if num_samples == 1:
        y = np.array([1.0, 1.0, 1.0]).tolist()
    else:
        y = [np.random.uniform(0.02, 3, size=3).tolist() for _ in range(num_samples)]

    data = []
    for combination in combinations:
        for y0 in y:
            data.append((combination, y0))

    np.random.shuffle(data)
    return data


# Dataset Generator Function
def rossler_dataset_fn(n, num_samples, time_points, time_intervals, data_parameters, path='data/'):
    times = np.linspace(0, time_intervals, time_points)
    combinations = generate_data_rossler(n, num_samples, data_parameters)

    dataset = pd.DataFrame()
    for idx, combi in enumerate(combinations):
        sol = simulator_rossler(combi[0], combi[1], times)
        sol_df = pd.DataFrame(sol)
        sol_df.index = [idx] * len(times)
        dataset = pd.concat([dataset, sol_df])

    os.makedirs(path, exist_ok=True)
    filename = f'rossler_{n}_{num_samples}_{time_points}_{time_intervals}_param_{str(data_parameters).replace(" ", "").replace(",", "_")}.pkl'
    dataset.to_pickle(os.path.join(path, filename))

    return dataset



###******************************************************************************
#******************************************************************************
# Fluid Flow Attractor System
#******************************************************************************

import numpy as np
import pandas as pd
import os
from scipy.integrate import odeint

class FluidFlowSystem:
    """
    Dynamical system modeling fluid flow on an attractor.
    Simulates a 3D dynamical system with variables x1, x2, x3.
    """

    def __init__(self, y0=None):
        if y0 is None:
            self._y0 = np.array([
                np.random.uniform(-1.1, 1.1),
                np.random.uniform(-1.1, 1.1),
                np.random.uniform(0, 2.42)
            ])
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 3:
                raise ValueError("Initial condition must have size 3.")

    def n_outputs(self):
        return 3

    def n_parameters(self):
        return 4  # mu, omega, A, lambda

    def _rhs(self, y, t, mu, omega, A, lam):
        x1, x2, x3 = y

        dx1_dt = mu * x1 - omega * x2 + A * x1 * x3
        dx2_dt = omega * x1 + mu * x2 + A * x2 * x3
        dx3_dt = -lam * (x3 - x1**2 - x2**2)

        return [dx1_dt, dx2_dt, dx3_dt]

    def simulate(self, parameters, times, clip_x3=True):
        mu = parameters["mu"]
        omega = parameters["omega"]
        A = parameters["A"]
        lam = parameters["lambda"]

        result = odeint(self._rhs, self._y0, times, args=(mu, omega, A, lam))

        if clip_x3:
            result = result[result[:, 2] <= 2.5]  # eliminate overflow cases

        return result

    def suggested_parameters(self):
        return {
            "mu": 0.1,
            "omega": 1.0,
            "A": -0.1,
            "lambda": 10.0
        }

    def suggested_times(self, T=100, n=1000):
        return np.linspace(0, T, n)


#******************************************************************************
# Simulator Function
#******************************************************************************

def simulator_fluidflow(parameter_dict, y0, times=None):
    if times is None:
        times = FluidFlowSystem().suggested_times()
    
    model = FluidFlowSystem(y0=y0)
    return model.simulate(parameter_dict, times)


#******************************************************************************
# Parameter Combination Generator
#******************************************************************************

def CombinationGenerator_fluidflow(n):
    """
    Generate `n` random parameter combinations for the fluid flow system.
    """
    combinations = []

    for _ in range(n):
        combo = {
            "mu": np.random.uniform(0.05, 0.15),
            "omega": np.random.uniform(0.8, 1.2),
            "A": np.random.uniform(-0.15, -0.05),
            "lambda": np.random.uniform(8.0, 12.0)
        }
        combinations.append(combo)

    return combinations


#******************************************************************************
# Data Generator Function
#******************************************************************************

def generate_data_fluidflow(n, num_samples, data_parameters=None):
    if n == 1 and data_parameters is not None:
        combinations = [data_parameters]
    else:
        combinations = CombinationGenerator_fluidflow(n)

    if num_samples == 1:
        y = [
            np.random.uniform(-1.1, 1.1),
            np.random.uniform(-1.1, 1.1),
            np.random.uniform(0, 2.42)
        ]
        y_list = [y]
    else:
        y_list = [
            [
                np.random.uniform(-1.1, 1.1),
                np.random.uniform(-1.1, 1.1),
                np.random.uniform(0, 2.42)
            ]
            for _ in range(num_samples)
        ]

    data = []
    for comb in combinations:
        for y0 in y_list:
            data.append((comb, y0))

    np.random.shuffle(data)
    return data


#******************************************************************************
# Dataset Generator Function
#******************************************************************************

def fluidflow_fn(n, num_samples, time_points, time_intervals, data_parameters=None, path='data/'):
    times = np.linspace(0, time_intervals, time_points)
    combinations = generate_data_fluidflow(n, num_samples, data_parameters)

    dataset = pd.DataFrame()
    for idx, (params, y0) in enumerate(combinations):
        sol = simulator_fluidflow(params, y0, times)
        if len(sol) == 0:
            continue  # skip trajectories that overflow
        sol_df = pd.DataFrame(sol, columns=['x1', 'x2', 'x3'])
        sol_df.index = [idx] * len(sol_df)
        dataset = pd.concat([dataset, sol_df])

    os.makedirs(path, exist_ok=True)
    filename = f'fluidflow_{n}_{num_samples}_{time_points}_{time_intervals}.pkl'
    dataset.to_pickle(os.path.join(path, filename))

    return dataset