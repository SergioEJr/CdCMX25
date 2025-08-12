# Clubes de Ciencia Mexico 2025
# by Sergio Eraso

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy.typing import NDArray

def initialize_lattice(L : int):
    """randomly initialize a lattice of spins of size LxL

    Args:
        L (int): dimension of the lattice

    Returns:
        ndarray: initialized lattice of size LxL
    """
    lattice = np.random.choice([-1,1], size=(L,L))
    
    return lattice

def convert_img(path : str, side : int):
    """makes an image into a square and converts its data
    into a binary string of 1s and -1s

    Args:
        path (str): image file location
        side (int): length of side of image

    Returns:
        ndarray: flattened binary image
    """
    img = Image.open(path)
    img = img.resize((side, side))          # make the image square  
    img = img.convert('1')                  # converts the image to binary
    img = 2 * np.array(img, int) - 1        # shifts pixels to be 1 and -1
    return img.flatten()

def show_img(img_array : NDArray):
    """shows an image

    Args:
        img_array (NDArray): flattened image array
    """
    side = int(np.sqrt(img_array.shape[0]))
    img_array = img_array.reshape((side, side))
    plt.figure(figsize=(3, 3))
    plt.imshow(img_array, cmap="grey")
    plt.axis('off')
    plt.show()

def corrupt_img(img : NDArray, p : float = 0.5):
    """corrupt an image by randomly flipping (or not)
    a proportion of its bits

    Args:
        img (ndarray): flattened image array
        p (float): proportion of data to corrupt, number between 0 and 1

    Returns:
        ndarray: corrupted image
    """
    N = img.size
    num_to_flip = int(p*N)
    indices = np.random.choice(N, size=num_to_flip, replace=False)
    c_img = img.copy()
    # flip the selected spins
    for i in indices:
        if np.random.rand() < 0.5:
            c_img[i] *= -1
    return c_img

def calculate_K(img : NDArray): 
    """create a weight matrix to store the information of the
    image in the network

    Args:
        img (ndarray): flattened image array

    Returns:
        ndarray: weight matrix
    """
    K = np.outer(img,img)
    return K

def delta_H(i : int, state : NDArray, K : NDArray):
    """Calculates the change in energy of a Hopfield network
    after flipping spin i

    Args:
        i (int): index of the neuron
        state (ndarray): current state
        K (ndarray): connectivity matrix

    Returns:
        ndarray: _description_
    """
    return state[i]*np.dot(K[i],state)

def recall(state : NDArray, K : NDArray): 
    """recall a pattern using the weight (connectivity) matrix w

    Args:
        state (ndarray): corrupted pattern
        K (ndarray): connectivity matrix

    Returns:
        ndarray: recovered pattern
    """
    n = state.size
    indices = np.random.permutation(n)
    r_img = state.copy()
    for i in range(n):
        flipped = False
        for index in indices:
            dE = delta_H(index, r_img, K)
            # go down the energy landscape
            if dE < 0:
                r_img[index] *= -1
                flipped = True
        if not flipped:
            print(f"converged in {i} iterations")
            break
        
    return r_img

def recall_snapshots(state : NDArray, K : NDArray, iterations : int = 1):
    """recall a pattern using the connectivity matrix K

    Args:
        state (ndarray): corrupted pattern
        K (ndarray): connectivity matrix
        iterations (int) : number of loops. Defaults to 1

    Returns:
        ndarray: recovered pattern
    """
    snapshots = []
    n = state.size
    r_img = state.copy()
    snapshots.append(r_img.copy())
    for i in range(iterations):
        flipped = False
        # update the pixels randomly
        indices = np.random.permutation(n)
        for index in indices:
            dE = delta_H(index, r_img, K)
            if dE < 0:
                r_img[index] *= -1
                flipped = True
            snapshots.append(r_img.copy())
        if not flipped:
            print(f"converged in {i} iterations")
            break
    return snapshots
