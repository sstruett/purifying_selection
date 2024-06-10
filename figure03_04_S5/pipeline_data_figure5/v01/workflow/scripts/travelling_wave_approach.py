"""Functions for the according jupyter-notebook."""


import sys
import math
import warnings

import numpy as np
import scipy as sp
import sklearn.linear_model
import tqdm


def fitness_class_distribution_hk(
    fitness_class_k: int, deleterious_mutation_rate: float, selcoef: float
) -> float:
    """Calculates the frequency of the fitness class using Nikolaisen and Desai (2012) equation (1).

    Args:
        fitness_class_k (int): The fitness class, i.e. the number of deleterious mutations.
        deleterious_mutation_rate (float): The rate of deleterious mutations per locus,
                                           e.g., length * per_site_mutation_rate.
        selcoef (float): The selection coefficient.

    Returns:
        float: The calculated fitness class distribution.

    Raises:
        None

    Calculates the fitness class distribution based on the given parameters using equation (1)
    from the paper by Nikolaisen and Desai (2012). The fitness class distribution is determined
    by the relative mutation rate, the fitness class, and the selection coefficient. This analytical
    expression assumes mutation-selection balance and is independent of the demography. The function
    can also be called alternatively with hk().
    """
    relmutrate = deleterious_mutation_rate / selcoef
    return (
        (relmutrate**fitness_class_k)
        * np.exp(-1 * relmutrate)
        / math.factorial(fitness_class_k)
    )


# alias to fitness_class_distribution_hk
hk = fitness_class_distribution_hk


def expected_lineage_weight_distribution(
    time_steps,
    deleterious_mutation_rate,
    selcoef,
    fitness_class_distribution=None,
    options: dict = {},
):
    """Calculates the expected lineage weight distribution over time based on Nd (2012)
        equation (5).

    Args:
        time_steps (float or list): The time or list of times at which to calculate the lineage
            weight distribution.
        deleterious_mutation_rate (float): The rate of deleterious mutations.
        selcoef (float): The selection coefficient.
        fitness_class_distribution (list or None): The pre-calculated fitness class distribution.
                                                   If None, it will be calculated using Nd.
        options (dict):
            threshold (float): The threshold to stop calculating the fitness class distribution.
                If not provided, default is 1e-12
            k_max (int): The maximum fitness class. If not provided, default is 500

    Returns:
        numpy.ndarray: The expected lineage weight distribution over time.

    Raises:
        None

    Calculates the expected lineage weight distribution over time based on equation (5) from the
    paper by Nd (2012).

    If a single time `time_steps` is provided, it is converted to a list to ensure compatibility
    with multiple time points.

    If the fitness class distribution is not provided, it will be calculated based on the
    `fitness_class_distribution_hk()` function for fitness classes up to `k_max`. The calculation
    stops when the fitness class distribution falls below the specified threshold.

    The function then precalculates the nuisance parameters related to the relative mutation rate
    and selection coefficient.

    Finally, the lineage weight is calculated for each time point and fitness class using the
    precalculated parameters and the expected lineage weight equation. The results are returned as a
    numpy array.

    Example usage:
        time_steps = [1, 2, 3]
        deleterious_mutation_rate = 0.1
        selcoef = 0.05
        expected_lineage_weight_distribution(time_steps, deleterious_mutation_rate, selcoef)
    """
    # options
    threshold = options.get("threshold", 1e-12)
    k_max = options.get("k_max", 500)

    if isinstance(time_steps, (float, int)):
        time_steps = [time_steps]

    if fitness_class_distribution is None:
        fitness_class_distribution = []
        for k in range(k_max + 1):
            fitness_class_distribution.append(
                fitness_class_distribution_hk(k, deleterious_mutation_rate, selcoef)
            )
            if fitness_class_distribution[-1] <= threshold:
                fitness_class_distribution = np.array(fitness_class_distribution)
                break

    relmutrate = deleterious_mutation_rate / selcoef

    lineage_weight_t = [[] for _ in range(len(time_steps))]
    for timix, my_time in enumerate(time_steps):
        selcoef_t = -1 * selcoef * my_time

        for k in range(len(fitness_class_distribution)):
            numerator = (relmutrate * np.exp(selcoef_t)) ** k * np.exp(
                -1 * relmutrate * np.exp(selcoef_t)
            )
            denominator = math.factorial(k)
            lineage_weight_t[timix].append(numerator / denominator)

    return np.array(lineage_weight_t)


def effective_population_size(
    time_steps: float or list,
    deleterious_mutation_rate: float,
    selcoef: float,
    population_size: int or float,
):
    """Calculate the effective population size under the assumptions of
    Nikolaisen and Desai (2012): constant population size and selection-
    mutation equilibrium.

    Parameters:
        time_steps (float or list): The time or list of times at which to calculate the effective
            population size.
        deleterious_mutation_rate (float): The rate of deleterious
        mutations.     selcoef (float): The selection coefficient.
        population_size (int or float): The size of the population.

    Returns:
        numpy.ndarray: The calculated effective population size for each time step.

    The function effective_population_size() calculates the effective
    population size based on the assumptions of Nikolaisen and Desai
    (2012). It assumes a constant population size and a selection-
    mutation equilibrium.

    If a single time step `time_steps` is provided, it is converted to a
    list to ensure compatibility with multiple time steps.

    The effective population size is calculated analytically using the
    formula: N_e = N * exp(-((U / s) * (1 - exp(-s * t))^2)) where N_e
    is the effective population size, N is the population size, U is the
    deleterious mutation rate, s is the selection coefficient, and t is
    the time step.

    The calculation is performed for each time step in `time_steps`, and
    the results are returned as a numpy array.

    Note that this function assumes the availability of the
    fitness_class_distribution_hk() function for dependent calculations.
    Please ensure that the fitness_class_distribution_hk() function is
    implemented separately and imported correctly.

    Example usage:
        time_steps = [1, 2, 3]
        deleterious_mutation_rate = 0.1
        selcoef = 0.05
        population_size = 1000
        effective_population_size(time_steps, deleterious_mutation_rate, selcoef, population_size)
    """
    if isinstance(time_steps, (float, int)):
        time_steps = [time_steps]

    return np.array(
        [
            population_size
            * np.exp(
                (-1 * deleterious_mutation_rate / selcoef)
                * (1 - np.exp(-1 * selcoef * t)) ** 2
            )
            for t in time_steps
        ]
    )


def expected_coalescence_time_distribution(
    time_steps: list,
    effective_population_size_history: float or list,
    nsam: int = 2,
    assume_sorted: bool = True,
    time_steps_coalescence=None,
):
    """Calculate the distribution of the expected time to coalescence for
    arbitrary population size histories.

    Parameters:
        time_steps (list): List of time steps for which to calculate the expected time to
            coalescence.
        effective_population_size_history (float or list): Effective population size history.
            Can be a single value or a list of values.
        nsam (int): Number of samples.
        assume_sorted (bool, optional): Whether to assume that the time_steps are already sorted in
            ascending order. Defaults to True.
        time_steps_coalescence (None or list, optional): On which times to provide the coalescence
            time. If None, calculates the coalescence at the provided times from the pop size
            history

    Returns:
        numpy.ndarray: Array containing the distribution of expected coalescence times for each time
            step.

    The function expected_coalescence_time() calculates the distribution
    of the expected time to coalescence for arbitrary population size
    histories. It uses numerical integrals to estimate the distribution.

    The `time_steps` parameter is a list of time steps for which to
    calculate the expected time to coalescence. The
    `effective_population_size_history` parameter represents the
    effective population size history. It can be a single value if the
    effective population size is constant, or a list of values
    corresponding to the population size at each time step. The length
    of `effective_population_size_history` should be the same as the
    length of `time_steps`.

    The `nsam` parameter specifies the number of samples.

    The `assume_sorted` parameter indicates whether the `time_steps` are
    already sorted in ascending order. By default, it is assumed to be
    True. If set to False, the function will sort the `time_steps` and
    `effective_population_size_history` arrays in ascending order.

    The function uses the scipy.interpolate.interp1d function to define
    a function `pop_size_t` that provides the population size as a
    function of time. Linear interpolation is used to approximate the
    population size between time steps. Extrapolation is performed
    assuming constant to the nearest value. The `pop_size_t` function is
    then integrated using scipy.integrate.quad to calculate the expected
    time to coalescence for each time step.

    The resulting distribution of expected coalescence times is returned
    as a numpy array.
    """
    if isinstance(effective_population_size_history, (float, int)):
        effective_population_size_history = np.array(
            [effective_population_size_history for _ in range(len(time_steps))]
        )

    assert len(time_steps) == len(
        effective_population_size_history
    ), "time steps should be informative for the pop size history"

    # preparse parameters
    effective_population_size_history = np.array(effective_population_size_history)
    time_steps = np.array(time_steps)
    ncomb = math.comb(nsam, 2)

    if not assume_sorted:
        sorted_indices = np.argsort(time_steps)
        time_steps = time_steps[sorted_indices]
        effective_population_size_history = effective_population_size_history[
            sorted_indices
        ]
        time_steps_coalescence.sort()

    if time_steps_coalescence is None:
        time_steps_coalescence = time_steps
    else:
        time_steps_coalescence = sorted(time_steps_coalescence)

    # define functions that provides the pop size and coalrate by time
    pop_size_t = sp.interpolate.interp1d(
        x=time_steps,
        y=effective_population_size_history,
        kind="linear",
        bounds_error=False,
        fill_value=(
            effective_population_size_history[0],
            effective_population_size_history[-1],
        ),
    )

    def coal_rate_t(my_time):
        return 1 / pop_size_t(my_time)

    # calculate the ccoalescence probabilities
    integral_values = np.array(
        [
            sp.integrate.quad(coal_rate_t, 0, my_time)[0]
            for my_time in time_steps_coalescence
        ]
    )
    coalescence_probabilities = np.where(
        integral_values == 0,
        None,
        ncomb
        * coal_rate_t(time_steps_coalescence)
        * np.exp(-1 * ncomb * integral_values),
    )

    return coalescence_probabilities


expected_coalescence_time = (
    expected_coalescence_time_distribution  # alias to the old function name
)


def population_size_by_eps(popsize, deleterious_mutation_rate: float, selcoef: float):
    """Using Charlesworth effective population size approximation under strong
    selection."""
    return np.array(popsize) * np.exp(-1 * deleterious_mutation_rate / selcoef)


def effective_population_size_manually(
    lineage_weight_k_at_time_t: list,
    population_size: int or float,
    fitness_distribution_hk_at_time_t: list,
):
    """Calculate the effective population size as the inverse of the summed
    coalescence rate.

    Parameters:
        lineage_weight_k_at_time_t (list): Lineage probability of being in class k at a given
            time t.
        population_size (int or float): Population size at time 0.
        fitness_distribution_hk (list): Fitness distribution.

    Returns:
        float: The calculated effective population size.

    The function calculates the effective population size by summing the
    coalescence rate for each fitness class weighted by the squared
    lineage probability and the corresponding population size at each
    class. The inverse of the summed coalescence rate is then returned
    as the effective population size.

    The `prob_k_t` list should have the same length as the
    `fitness_distribution_hk` list.
    """
    assert len(lineage_weight_k_at_time_t) == len(
        fitness_distribution_hk_at_time_t
    ), "prob_k_t and fitness_distribution_hk must have the same length"

    coalescence_rates = [
        (p**2) / (population_size * fhk)
        for p, fhk in zip(
            lineage_weight_k_at_time_t.T, fitness_distribution_hk_at_time_t.T
        )
    ]
    summed_coal_rate = sum(coalescence_rates)

    return 1 / summed_coal_rate


def expected_lineage_weight_distribution_linalg(
    time_steps: int or list,
    pi_0,
    migration_matrix,
    deleterious_mutation_rate=None,
    selcoef=None,
):
    """Calculate the lineage distribution over time using matrix operations.

    Parameters:
        time_steps (int or list): Time step or list of time steps at which to calculate the lineage
            distribution.
        pi_0 (numpy.ndarray): Initial lineage distribution at time 0.
        migration_matrix (str or numpy.ndarray): Migration matrix or a string specifying a specific
            migration pattern. If a string, the following patterns are supported:
                - "Nd": N-deme model with deleterious mutation rate. Requires
                    `deleterious_mutation_rate` parameter. The migration matrix is constructed with
                    diagonal elements (1 - deleterious_mutation_rate) and off-diagonal elements as
                    deleterious_mutation_rate.
        deleterious_mutation_rate (float, optional): Deleterious mutation rate. Required if
            migration_matrix is "Nd".

    Returns:
        numpy.ndarray: The expected lineage weight distribution at the specified time steps.

    Raises:
        ValueError: If an unknown migration pattern is provided.

    This function calculates the lineage distribution over time using
    matrix operations. It accepts the initial lineage distribution
    `pi_0`, the migration matrix, and the time steps at which to
    calculate the distribution.

    If the `time_steps` parameter is an integer, the function calculates
    the lineage distribution at a single time step. If it is a list of
    time steps, the function calculates the lineage distribution at
    multiple time steps.

    The `migration_matrix` parameter can be a numpy array representing
    the migration matrix, or it can be a string specifying a specific
    migration pattern. If a string is provided, the following patterns
    are supported:

    - "Nd": N-deme model with deleterious mutation rate. This pattern
    requires the `deleterious_mutation_rate`   parameter. The migration
    matrix is constructed with diagonal elements (1 -
    deleterious_mutation_rate) and   off-diagonal elements as
    deleterious_mutation_rate.
    """
    if isinstance(time_steps, int):
        time_steps = np.array([time_steps])
    else:
        time_steps = np.array(time_steps).astype(int)

    if isinstance(migration_matrix, str):
        if migration_matrix == "Nd":
            migration_matrix = create_migration_matrix(
                parameters={
                    "selcoef": selcoef,
                    "deleterious_mutation_rate": deleterious_mutation_rate,
                    "number_fitness_classes": len(pi_0),
                    "fitness_class_freqs": pi_0
                },
                mode="Nd",
            )
        else:
            raise ValueError("Unknown migration pattern")

    expected_distribution = np.array(
        [pi_0 @ np.linalg.matrix_power(migration_matrix, t) for t in time_steps]
    )

    return expected_distribution


def create_bwd_migration_matrix(
    parameters: dict = {},
    options: dict = {},
    mode: str="neutral"
) -> np.ndarray:
    """Create a migration matrix for the Nikolaisen and Desai (2012) approach
    based on weak purifying selection.

    Parameters:
        number_fitness_classes (int): Number of fitness classes.
        deleterious_mutation_rate (float): Rate of deleterious mutation.
        wave_velocity (float): Velocity of the wave in relative terms compared to the mutation rate.
        mode (str): Migration pattern mode. Available options: "Nd", "Nd_weak_selection",
            "Nd_weak_selection_reduced".

    Returns:
        numpy.ndarray: Migration matrix.

    Notes:
        - For mode "Nd_weak_selection_reduced", the migration matrix is only filled for the diagonal
            elements one position above the main diagonal.

    Raises:
        ValueError: If an unknown migration pattern mode is provided.
        AssertionError: If the wave travels faster than the mutation rate or if the wave velocity is
            negative.
    """
    # params
    class params:
        population_size = parameters.get("population_size", 100)
        selcoef = parameters.get("selcoef", 0.0)
        deleterious_mutation_rate = parameters.get("deleterious_mutation_rate", 0.0)
        fitness_class_freqs = parameters.get("fitness_class_freqs")
        number_fitness_classes = parameters.get("number_fitness_classes", len(fitness_class_freqs))
        wave_velocity = parameters.get("wave_velocity", 0)
    

    # options
    class opt:
        # assertations and warnings
        assert_wave_velocity = options.get("force_wave_velocity", True)
        warn_small_wave_velocity = options.get("warn_small_wave_velocity", False)
        warn_small_wave_velocity_threshold = options.get("warn_small_wave_velocity_threshold", 1e-2)


    # assertations and checks
    if opt.assert_wave_velocity:
        assert 0 <= params.wave_velocity <= 1, "relative wave velocity must be >= 0 and <= 1"
    if opt.warn_small_wave_velocity:
        if params.wave_velocity <= opt.warn_small_wave_velocity_threshold:
            warnings.warn("The relative wave velocity is very small, make sure it is the relative wave velocity (and not the absolute wave velocity)")


    # create empty migmat; migmat always meant to be bwd-in-time
    migmat = np.zeros(
        (params.number_fitness_classes, params.number_fitness_classes)
    )  # Empty migration matrix

    if mode == "Nd":
        migrates_bwd_in_time = [
            params.selcoef * k for k in range(1, params.number_fitness_classes)
        ]
        np.fill_diagonal(migmat[1:], migrates_bwd_in_time)
    elif (
        mode == "Nd_Ud"
    ):  # use mutrate to determine the migrates, this can be a corrected migration rate
        if params.fitness_class_freqs is None:
            fitness_class_freqs = [
                fitness_class_distribution_hk(
                    fitness_class_k=this_k,
                    deleterious_mutation_rate=params.deleterious_mutation_rate,
                    selcoef=params.selcoef,
                )
                for this_k in range(params.number_fitness_classes)
            ]
        else:
            fitness_class_freqs = params.fitness_class_freqs
        migrates_bwd_in_time = [
            fitness_class_freqs[k - 1]
            * params.deleterious_mutation_rate
            / fitness_class_freqs[k]
            for k in range(1, params.number_fitness_classes)
        ]
        np.fill_diagonal(migmat[1:], migrates_bwd_in_time)
    elif mode == "Udv":  # Ud (deleterious mutation rate); v (wave_velocity)
        fitness_class_freqs = params.fitness_class_freqs
        
        migrates_bwd_in_time_to_fitter_classes = [
            fitness_class_freqs[k - 1]
            * params.deleterious_mutation_rate
            / fitness_class_freqs[k]
            for k in range(1, params.number_fitness_classes)
        ]
        np.fill_diagonal(migmat[1:], migrates_bwd_in_time_to_fitter_classes)
        
        
        migrates_bwd_in_time_to_lower_classes = [
            fitness_class_freqs[k+1]
            * params.wave_velocity * params.deleterious_mutation_rate
            / fitness_class_freqs[k]
            for k in range(params.number_fitness_classes-1)
        ]
        np.fill_diagonal(migmat[:,1:], migrates_bwd_in_time_to_lower_classes)
    else:
        raise ValueError("Unknown migration pattern mode.")

    # must sum up to 1, this is migration to the same class
    np.fill_diagonal(migmat, 1 - migmat.sum(axis=1))

    return migmat


create_migration_matrix = create_bwd_migration_matrix  # old name of the function


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
        my_phi = (
            population_size * selcoef * np.exp(-deleterious_mutation_rate / selcoef)
        )
    elif selcoef == 0:
        my_phi = 0
    else:
        assert False, "unknown selcoef, please check the code of this function"

    if to_str:
        my_phi = f"{my_phi:{strfmt}}"

    return my_phi


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


def throw_a_stone(
    population_size: int or float,
    selcoef: float,
    deleterious_mutation_rate: float,
    options: dict = {},
    seed: int = None,
) -> list:
    """Calculate relative wave velocity

    Args:
        population_size (float or list): Population size, number of haploids.
        deleterious_mutation_rate (float): The rate of deleterious mutations.
        selcoef (float): The selection coefficient.
        options (dict):
            mode (str): Defines what to be calculated. Default is "velocity_only"
            burnin (int): Number of generations to leave out. Default is
                7 * population_size.
            burnin_fitter (int): Number of generations to leave out for the linear
                model, if not given, then is same as burnin
            generations (int): Number of generations to simulate. Default is
                20 * population_size.
            thinning (int): How much to thin. When thinning is n, then every n-th
                generation will be used to calculate the velocity. Default is 12.
            progress_bar (bool): Show progress bar of simulations. Default is True.
            repeat (int): Number of independtendly run simulations. Default is 1.
        seed (int): Random seed to be used for the simulation. Default is 1.

    Returns:
        list: Dependend on the chosen model.
            - "velocity_only": list of floats: velocities
            - "velocity_and_distribution": list of 2-tuples: (velocity, distribution)s
                the distribution is a 1d-numpy-array with providing the frequency
            - "fitness_coalescence_distribution": list of 3-tuples: First two as in
                "velocity_and_distribution". The third entry provides the proportion
                of coalescence events in each class. These will be calculated on all
                unthinned generations

    Raises:
        ValueError: If unknown mode is provided in the options.


    Under an all constant WF model with multiplicative fitness, we calculate the relative
    wave velocity.


    We first choose the parents based on their relative fitness. Then, we mutate the
    children, such that they carry the new mutations added to the existing ones from
    their parents.
    """
    np.random.seed(seed)

    # options
    class opt:
        mode = options.get("mode", "velocity_only")
        burnin = options.get("burnin", 3 * population_size)
        burnin_fitter = options.get("burnin_fitter", None)
        generations = options.get("generations", 10 * population_size)
        thinning = options.get(
            "thinning", 12
        )  # use every 12-th generation after burnin
        progress_bar = options.get("progress_bar", True)
        repeat = options.get("repeat", 1)
        return_wave = options.get("return_wave", False)
        return_fitter = options.get("return_fitter", False)  
        return_sim = options.get("return_sim", False)
        return_parents = options.get("return_parents", False)
        return_rawsim = options.get("return_rawsim", False)


    # create result 'container'
    result = []


    # create parent 'container'
    if opt.return_parents:
        wf_parents = []
    

    for replicate_ix in range(opt.repeat):
        if opt.mode == "velocity_only":
            # create fitness array for the simulation
            sim = np.zeros((opt.generations, population_size))

            # init first sim with mutations
            sim[0, :] = np.random.poisson(deleterious_mutation_rate, population_size)

            # simulate generations
            for generation_index in tqdm.tqdm(
                range(1, opt.generations),
                desc=f" fitness wave ({replicate_ix+1}/{opt.repeat})",
                disable=not opt.progress_bar,
            ):
                # choose parents
                fitness_weight = (1 - selcoef) ** sim[generation_index - 1, :]
                parent_index = np.random.choice(
                    range(population_size),
                    size=population_size,
                    replace=True,
                    p=fitness_weight / fitness_weight.sum(),
                )

                if opt.return_parents:
                    wf_parents.append(parent_index)

                # mutate
                sim[generation_index, :] = (
                    sim[generation_index - 1, parent_index]
                ) + np.random.poisson(deleterious_mutation_rate, population_size)


            # create sim and time for the fitter in case there is a burnin_fitter
            if not opt.burnin_fitter is None:
                sim_time_fitter = np.arange(sim.shape[0])[opt.burnin_fitter :: opt.thinning]
                sim_fitter = sim[opt.burnin_fitter :: opt.thinning, :]
            else:
                sim_time_fitter = np.arange(sim.shape[0])[opt.burnin :: opt.thinning]
                sim_fitter = sim[opt.burnin :: opt.thinning, :]


            # remove the burnin and do the thinning
            sim_time_thinned = np.arange(sim.shape[0])[opt.burnin :: opt.thinning]
            sim = sim[opt.burnin :: opt.thinning, :]


            # copy the raw wave when it shall be returned
            if opt.return_wave:
                sim_raw = sim.copy()


            # calculate the velocity as a linear regression model
            sim_time_thinned_flat = np.repeat(sim_time_fitter, sim_fitter.shape[1])
            sim = sim_fitter.flatten("C")

            model_full = sklearn.linear_model.LinearRegression().fit(
                sim_time_thinned_flat.reshape(-1, 1), sim
            )
            wave_velocity = model_full.coef_[0]

            relative_wave_velocity = wave_velocity / deleterious_mutation_rate

            result.append(relative_wave_velocity)
        elif opt.mode == "velocity_and_distribution":
            # create fitness array for the simulation
            sim = np.zeros((opt.generations, population_size))

            # init first sim with mutations
            sim[0, :] = np.random.poisson(deleterious_mutation_rate, population_size)

            # simulate generations
            for generation_index in tqdm.tqdm(
                range(1, opt.generations),
                desc=f" fitness wave ({replicate_ix+1}/{opt.repeat})",
                disable=not opt.progress_bar,
            ):
                # choose parents
                fitness_weight = (1 - selcoef) ** sim[generation_index - 1, :]
                parent_index = np.random.choice(
                    range(population_size),
                    size=population_size,
                    replace=True,
                    p=fitness_weight / fitness_weight.sum(),
                )

                if opt.return_parents:
                    wf_parents.append(parent_index)

                # mutate
                sim[generation_index, :] = (
                    sim[generation_index - 1, parent_index]
                ) + np.random.poisson(deleterious_mutation_rate, population_size)



            # create sim and time for the fitter in case there is a burnin_fitter
            if not opt.burnin_fitter is None:
                sim_time_fitter = np.arange(sim.shape[0])[opt.burnin_fitter :: opt.thinning]
                sim_fitter = sim[opt.burnin_fitter :: opt.thinning, :]
            else:
                sim_time_fitter = np.arange(sim.shape[0])[opt.burnin :: opt.thinning]
                sim_fitter = sim[opt.burnin :: opt.thinning, :]


            # remove the burnin and do the thinning
            sim_time_thinned = np.arange(sim.shape[0])[opt.burnin :: opt.thinning]
            sim = sim[opt.burnin :: opt.thinning, :]


            # copy the raw wave when it shall be returned
            if opt.return_wave:
                sim_raw = sim.copy()


            # calculate the velocity as a linear regression model
            sim_time_thinned_flat = np.repeat(sim_time_fitter, sim_fitter.shape[1])
            sim_flat = sim_fitter.flatten("C")

            model_full = sklearn.linear_model.LinearRegression().fit(
                sim_time_thinned_flat.reshape(-1, 1), sim_flat
            )
            wave_velocity = model_full.coef_[0]

            relative_wave_velocity = wave_velocity / deleterious_mutation_rate

            # predict the mean k-fitness to normalize the distribution using our linear model
            mean_fitness_pred = model_full.predict(sim_time_thinned.reshape(-1, 1))

            # center the values and add its min value to provide only positive k-classes
            sim_raw = sim.copy()
            sim = sim - mean_fitness_pred[:, np.newaxis]
            sim = sim - sim.min()

            # make histogram; generous upper bound
            bin_edges = np.arange(-0.05, np.ceil(2*sim.max()) + .5, 1)
            simhk, _ = np.histogram(sim, bins=bin_edges)
            simhk = np.trim_zeros(simhk, trim="b")  # remove trailing zeros from back
            simhk = simhk/simhk.sum()  # get frequency
           
            result_tuple = relative_wave_velocity, simhk

            result.append(result_tuple)
        elif opt.mode == "fitness_coalescence_distribution":
            # create fitness array for the simulation
            sim = np.zeros((opt.generations, population_size))
            
            # track also the parent indexes; but only after burnin
            parents = np.zeros((opt.generations - opt.burnin, population_size))

            # init first sim with mutations
            sim[0, :] = np.random.poisson(deleterious_mutation_rate, population_size)

            # simulate generations
            for generation_index in tqdm.tqdm(
                range(1, opt.generations),
                desc=f" fitness wave ({replicate_ix+1}/{opt.repeat})",
                disable=not opt.progress_bar,
            ):
                # choose parents
                fitness_weight = (1 - selcoef) ** sim[generation_index - 1, :]
                parent_index = np.random.choice(
                    range(population_size),
                    size=population_size,
                    replace=True,
                    p=fitness_weight / fitness_weight.sum(),
                )
                
                # track parents
                if generation_index >= opt.burnin:
                    parents[generation_index - opt.burnin, :] = parent_index

                # mutate
                sim[generation_index, :] = (
                    sim[generation_index - 1, parent_index]
                ) + np.random.poisson(deleterious_mutation_rate, population_size)

                
                # fitness coalescence distribution or which fitness do parents of more than 1 child have
                parent_index_unique, parent_index_counts = np.unique(parent_index, return_counts=True)
                



            # create sim and time for the fitter in case there is a burnin_fitter
            if not opt.burnin_fitter is None:
                sim_time_fitter = np.arange(sim.shape[0])[opt.burnin_fitter :: opt.thinning]
                sim_fitter = sim[opt.burnin_fitter :: opt.thinning, :]
            else:
                sim_time_fitter = np.arange(sim.shape[0])[opt.burnin :: opt.thinning]
                sim_fitter = sim[opt.burnin :: opt.thinning, :]


            # remove the burnin and do the thinning; keep unthinned for coalescence distributions
            sim_time_unthinned = np.arange(sim.shape[0])[opt.burnin  ::]
            sim_time_thinned = np.arange(sim.shape[0])[opt.burnin :: opt.thinning]
            sim_unthinned = sim[opt.burnin::, :]
            sim = sim[opt.burnin :: opt.thinning, :]


            # copy the raw wave when it shall be returned
            if opt.return_wave:
                sim_raw = sim.copy()


            # calculate the velocity as a linear regression model
            sim_time_thinned_flat = np.repeat(sim_time_fitter, sim_fitter.shape[1])
            sim_flat = sim_fitter.flatten("C")

            model_full = sklearn.linear_model.LinearRegression().fit(
                sim_time_thinned_flat.reshape(-1, 1), sim_flat
            )
            wave_velocity = model_full.coef_[0]

            relative_wave_velocity = wave_velocity / deleterious_mutation_rate

            # predict the mean k-fitness to normalize the distribution using our linear model
            mean_fitness_pred = model_full.predict(sim_time_thinned.reshape(-1, 1))
            mean_fitness_pred_unthinned = model_full.predict(sim_time_unthinned.reshape(-1, 1))

            # center the values and add its min value to provide only positive k-classes
            sim_raw = sim.copy()
            sim = sim - mean_fitness_pred[:, np.newaxis]
            sim = sim - sim_unthinned.min()
            sim_unthinned = sim_unthinned - mean_fitness_pred_unthinned[:, np.newaxis]
            sim_unthinned = sim_unthinned - sim_unthinned.min()

            # make histogram; generous upper bound
            bin_edges = np.arange(-0.5, np.ceil(2*sim.max()) + .5, 1)
            simhk, _ = np.histogram(sim, bins=bin_edges)
            simhk = np.trim_zeros(simhk, trim="b")  # remove trailing zeros from back
            simhk = simhk/simhk.sum()  # get frequency
            
            # calculate relative coalescence distribution (in which hk does most of the coalescence happen)
            relative_coalescence_distribution = np.zeros(simhk.shape)
            bin_edges = bin_edges[:(len(simhk)+1)]
            
            # loop through generations
            for generation_index in tqdm.tqdm(
                range(1, len(sim_unthinned)),
                desc=" counting coalescences",
                disable=not opt.progress_bar,
            ):
                # find parents occuring more than once
                this_parents = parents[generation_index, :]
                this_parents_unique, this_parents_count = np.unique(this_parents, return_counts=True)
                this_parents_unique = this_parents_unique[this_parents_count>1]
                this_parents_count = this_parents_count[this_parents_count>1] - 1
                
                # get fitness class of those parents (in previous generation)
                this_parents_fitness = sim_unthinned[generation_index-1, this_parents_unique.astype(int)]
                this_parents_fitness_flat = np.repeat(this_parents_fitness, this_parents_count)
                this_parents_coalescence, _ = np.histogram(this_parents_fitness_flat, bins=bin_edges)
                relative_coalescence_distribution += this_parents_coalescence
                del this_parents_coalescence
                
            # normalize coalescence counts
            relative_coalescence_distribution /= relative_coalescence_distribution.sum()

                
            result_triple = relative_wave_velocity, simhk, relative_coalescence_distribution

            result.append(result_triple)
        else:
            raise ValueError(f"Mode {opt.mode} is not implemented yet.")


        # prepare and append the simulated wave if the option has been chosen
        if opt.return_wave:
            # make histogram
            bin_edges = np.arange(-0.5, np.ceil(sim_raw.max()) + 1.5, 1)
            simhk_t = np.zeros((sim.shape[0], len(bin_edges)-1))


            for generation_index in tqdm.trange(
                simhk_t.shape[0],
                desc=f" calc wave ({replicate_ix+1}/{opt.repeat})",
                disable=not opt.progress_bar,
            ):
                simhk_t_i, _ = np.histogram(sim_raw[generation_index, :], bins=bin_edges)
                simhk_t[generation_index, :] = simhk_t_i


            result.append(simhk_t)


        # return fitter if option is activated
        if opt.return_fitter:
            result.append(model_full)


        # return the simulated population over time if option is activated
        if opt.return_sim:
            result.append(sim_raw)

        # return the parent indices if option is activated
        if opt.return_parents:
            result.append(wf_parents)

        if opt.return_rawsim:
            result.append(sim_raw)


    return result

def imported():
    """Test if import of module was successful."""
    return True
