"""
Functions for calc_coalescence_densities
"""


import os
import numpy as np
import pandas as pd
import warnings


def get_filepaths(directory):
    """
    Get the full file paths of all files in a directory.

    Args:
        directory (str): The directory path.

    Returns:
        list: A list of full file paths.
    """
    filenames = os.listdir(directory)
    filepaths = [os.path.join(directory, filename) for filename in filenames]
    return filepaths


def extract_parameter_values(filename, which_params=None):
    """
    Extract parameter values from a filename.

    Args:
        filename (str): The filename.

    Returns:
        dict or list: A dictionary containing the parameter names and their corresponding values.
            If 'which_params' are given, will return a list without the names, but in conserved
            order.
    """
    parameter_dict = {}
    filename = os.path.basename(filename)  # Extract filename without path
    filename = os.path.splitext(filename)[0]  # Remove file extension if present

    # Split filename by comma and iterate over parts
    parts = filename.split(',')
    for part in parts:
        key, value = part.split('_')
        try:
            parameter_dict[key] = int(value) if value.isdigit() else float(value)
        except ValueError:
            parameter_dict[key] = value

            
    if not which_params is None:
        if isinstance(which_params, str):
            which_params = [which_params]
        
        results = []
        for param in which_params:
            results.append(parameter_dict[param])
        
        
    else:
        results = parameter_dict
            
            
    return results


def calc_density(my_file, options: dict={}):
    """Calculates the density of a simulation based on the provided file.

    Parameters:
        my_file (str): The file path of the simulation data.
        options (dict, optional): Additional options for the calculation. 
            Defaults to an empty dictionary.

    Returns:
        tuple: A tuple containing the following elements:
            - simprobs (numpy.ndarray): The density probabilities calculated from the simulation.
            - bin_widths (numpy.ndarray): The widths of the bins used for density calculation.
            - bin_edges (numpy.ndarray): The edges of the bins used for density calculation.
            
    Note:
        - N is assumed to be the number of individuals in slim, which will be halved as
          only half of the individuals are of a single sex.
        
    """
    # provide params from options
    nbin = options.get("nbin", 20)
    tmax = options.get("tmax", 3)
    
    
    # extract the parameters
    N, s, U = extract_parameter_values(
        my_file, which_params=("popsize", "selcoeff", "mutrate")
    )
    if s <= 0:
        s = -s
    else:
        warnings.warn(f"{s} Selcoef was extracted from filename as being positive. This can be true. The value did not get flipped to be positive. Make sure, you know we talk about deleterious mutations though.")
    
    
    N = int(round(N / 2))  # N in slim is number of individuals, but only half of them are of the desired sex
    
    
    # param dict
    param_dict = {
        "popsize": N,
        "selcoef": s,
        "deleterious_mutrate": U
    }
    
    
    # define bins
    bin_edges = time_intervals(nsam=nbin, max_tmrca=tmax, popsize=2*N)
    bin_widths = np.diff(bin_edges)
    
    
    # get coaldens
    simdat = pd.read_feather(my_file)
    
    
    simcoal, simweights = simdat["time"].to_numpy(), simdat["count"].to_numpy()
    hist_counts, _ = np.histogram(simcoal, bins=bin_edges, weights=simweights)
    simprobs = hist_counts / (hist_counts.sum() * bin_widths)
    
    
    return simprobs, bin_widths, bin_edges, param_dict


def time_intervals(nsam=40, max_tmrca=2, popsize=1, include_0=True, include_inf=True):
    """Generate time intervals for a given number of samples and maximum time.

    Parameters:
        nsam (int): Number of samples. Default is 40.
        max_tmrca (float): Maximum time. Default is 2.
        popsize (int or float): Population size. Default is 1. Used to scale the time.
        include_0 (bool): Whether to include the time interval 0. Default is True.
        include_inf (bool): Whether to include the infinite time interval. Default is True.

    Returns:
        numpy.ndarray: Array of time intervals.
    """
    my_breaks = np.zeros((nsam,))

    for my_index in range(nsam):
        my_breaks[my_index] = (
            0.1 * np.exp((my_index + 1) / nsam * np.log(1 + 10 * max_tmrca)) - 0.1
        )

    if include_0:
        my_breaks = np.array([0] + list(my_breaks))

    if include_inf:
        my_breaks = np.array(list(my_breaks) + [np.inf])

    return my_breaks * popsize


def phi(
    population_size: int or float,
    selcoef: float,
    deleterious_mutation_rate: float,
    to_str: bool = True,
    strfmt: str = r".3g",
) -> str:
    """Calculate the independent lineage criterion based on Nicolaisen and
    Desai (2012).

    The independent lineage criterion, denoted as phi (φ), is used to quantify the impact of
    selection and mutation on genetic diversity in a population.

    Args:
        population_size (float): The effective population size.
        deleterious_mutation_rate (float): The rate of deleterious mutations per generation.
        selcoef (float): The selection coefficient.

    Returns:
        str: The string-formatted calculated value of the independent lineage criterion.

    Raises:
        AssertionError: If any of the input arguments are not greater than zero.

    Examples:
        >>> phi(1000, 0.01, 0.1)
        9.04837418035918

    References:
        - Nicolaisen and Desai (2012). Distortions in genealogies due to purifying selection. Mol.
          Biol. Evol. 29(11): 3589-3600
    """
    assert selcoef >= 0, "Selection coefficient must be greater (or eq) than zero."


    if selcoef > 0:
        my_phi = population_size * selcoef * np.exp(-deleterious_mutation_rate / selcoef)
    elif selcoef == 0:
        my_phi = 0
    else:
        assert False, "unknown selcoef, please check the code of this function"


    if to_str:
        my_phi = f"{my_phi:{strfmt}}"

    return my_phi

