import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from numpy.typing import NDArray
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

def initialize_lattice(L : int):
    """randomly initialize a lattice of spins of size LxL

    Args:
        L (int): dimension of the lattice

    Returns:
        ndarray: initialized lattice of size LxL
    """
    lattice = np.random.choice([-1,1], size=(L,L))
    
    return lattice

def H(row : int, col : int, lattice : NDArray, params : dict):
    """calculates the nearest-neighbor interaction energies of a spin
    at location (row,col) in a lattice assuming periodic boundary conditions

    Args:
        row (_type_): _description_
        col (_type_): _description_
        lattice (_type_): _description_
    """
    L = lattice.shape[0]
    K = params["interaction_strength"]
    h = params["external_field"]
    # store central spin
    s = lattice[row][col]
    # store neighboring spins
    neighbors = np.zeros(4)
    neighbors[0] = lattice[(row - 1)%L][col]        # up
    neighbors[1] = lattice[(row + 1)%L][col]        # down
    neighbors[2] = lattice[row][(col - 1)%L]        # left
    neighbors[3] = lattice[row][(col + 1)%L]        # right
    # calculate the interaction energy and return
    energy = -K*np.sum(neighbors*s) + h*s
    return energy
    

def update(lattice : NDArray, params : dict):
    """updates the lattice by randomly selecting a spin
    and evolving it according to the Metropolis-Hastings algorithm

    Args:
        lattice (ndarray): current state of the system
        T (float): temperature of the system
    
    Returns:
        ndarray: new state of the system
    """
    L = lattice.shape[0]
    T = params["temperature"]
    row = np.random.randint(0,L)
    col = np.random.randint(0,L)
    
    E = H(row, col, lattice, params)
    boltzmann_weight = np.exp(-E/T)
    flip_boltzmann_weight = np.exp(E/T)
    
    accept_ratio = flip_boltzmann_weight/boltzmann_weight
    # if the energy is positive, then flipping is better
    if accept_ratio > 1:
        lattice[row][col] = -lattice[row][col]
    # otherwise, flip with probability given by the Boltzmann factor
    elif np.random.rand() < accept_ratio:
        lattice[row][col] = -lattice[row][col]
        
    return lattice

def sweep(lattice : NDArray, params : dict):
    L = lattice.shape[0]
    new_lattice = lattice.copy()
    for _ in range(L*L):
        new_lattice = update(new_lattice, params)
    return new_lattice

def run_simulation(n_sweeps : int, L : int, T : float, K : float, h : float):
    params = {"temperature" : T, "interaction_strength" : K, "external_field" : h}
    lattice = initialize_lattice(L)
    for i in range(n_sweeps):
        lattice = sweep(lattice, params)
    return lattice

def unpack_run_simulation(args):
    return run_simulation(*args)

def snapshot_series(n_sweeps : int, L : int, T : float, K : float, h : float):
    params = {"temperature" : T, "interaction_strength" : K, "external_field" : h}
    lattice = initialize_lattice(L)
    snapshots = []
    snapshots.append(lattice)
    for i in range(n_sweeps):
        lattice = sweep(lattice, params)
        snapshots.append(lattice)
    snapshots = np.asarray(snapshots)
    return snapshots

def ensemble_simulation(n_trials : int, n_sweeps : int, L : int, T : float, K : float, h : float):
    args = [(n_sweeps, L, T, K, h) for _ in range(n_trials)]
    final_snapshots = process_map(unpack_run_simulation,
                        args,
                        chunksize=1,        # tweak for best throughput
                        desc="Simulations")
    final_snapshots = np.asarray(final_snapshots)

    return final_snapshots