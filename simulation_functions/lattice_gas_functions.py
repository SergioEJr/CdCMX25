# Clubes de Ciencia Mexico 2025
# by Sergio Eraso

import numpy as np
from numpy.typing import NDArray

def initialize_lattice(L : int):
    """initialize a lattice of size LxL with half filled with ones and half with zeros

    Args:
        L (int): dimension of the lattice

    Returns:
        ndarray: initialized lattice of size LxL with half filled with ones and half with zeros
    """
    lattice = np.ones((L,L))
    lattice[:,:L//2] = 0
    return lattice

def choose_neighbor(row : int, col : int, L : int):
    """select a random neighbor of a site in a lattice of size LxL assuming
    periodic boundary conditions in the horizontal direction and 
    closed boundary conditions in the vertical direction

    Args:
        row (int): row of the central site
        col (int): column of the central site
        L (int): dimension of the lattice
        
    Returns:
        tuple: coordinates of the chosen neighbor
    """
    #   0
    # 2 x 3
    #   1
    # if at the top row, choose a random neighbor except the one above
    if row == 0:
        neighbor = np.random.choice([1,2,3])
    # if at the bottom row, choose a random neighbor except the one below
    elif row == L-1:
        neighbor = np.random.choice([0,2,3])
    # otherwise, choose a random neighbor
    else:
        neighbor = np.random.choice([0,1,2,3])
    if neighbor == 0:  # up
        n_row = row - 1
        n_col = col
    elif neighbor == 1:  # down
        n_row = row + 1
        n_col = col
    elif neighbor == 2:  # left
        n_row = row
        n_col = (col - 1) % L  # wrap around horizontally
    elif neighbor == 3:  # right
        n_row = row
        n_col = (col + 1) % L  # wrap around horizontally
    return n_row, n_col 

def update(lattice : NDArray):
    """evolves the lattice by randomly selecting a site and swapping it with one of its neighbors

    Args:
        lattice (ndarray): current state of the lattice

    Returns:
        ndarray: new state of the lattice after one update
    """
    L = lattice.shape[0]
    # choose a random site
    row = np.random.randint(0,L)
    col = np.random.randint(0,L)
    # choose a random neighbor respecting the boundary conditions
    n_row, n_col = choose_neighbor(row, col, L)
    # store the current site and the neighbor
    site = lattice[row,col]
    neighbor = lattice[n_row, n_col]
    # swap the current site with the neighbor
    lattice[row][col] = neighbor
    lattice[n_row][n_col] = site
    return lattice

def sweep(lattice : NDArray):
    """performs a sweep over the lattice, that is LxL updates
    
    Args:
        lattice (ndarray): current state of the lattice
        
    Returns:
        ndarray: new state of the lattice after LxL updates
    """
    L = lattice.shape[0]
    new_lattice = lattice.copy()
    for _ in range(L*L):
        new_latice = update(new_lattice)
    return new_lattice

def x_density(lattice, window_size : int = 1):
    """Calculates the coarse-grained density of a given
    lattice gas configuration. The larger window_size, the
    smoother the density profile.

    Args:
        lattice (NDArray): current state
        window_size (int) : rolling average window size
        
    Returns:
        NDArray : coarse-grained x-density profile
    """
    L = lattice.shape[0]
    # density of each column
    dens = np.mean(lattice, axis = 0)
    # concatenate the array three times for periodic boundary conditions
    dens3 = np.concatenate([dens,dens,dens])
    window = np.ones(window_size)/window_size
    # calculate the rolling average density for each x location
    dens = np.convolve(window, dens3, mode="same")[L:2*L]
    return dens