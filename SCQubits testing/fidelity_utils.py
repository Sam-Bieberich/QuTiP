import qutip as qt
import numpy as np
import scqubits as scq
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import itertools
import warnings

warnings.filterwarnings('ignore', module='qutip') #remove the random qutip warnings
scq.settings.T1_DEFAULT_WARNING=False

c_ops = None  # Global placeholder
fluxonium = scq.Fluxonium(EJ=8.9, EC=2.5, EL=0.5, flux=0.48, cutoff=110)  # Already done

def init_c_ops():
    gamma_ij = {}
    for j in range(1, levels):
        for i in range(j):
            t1 = fluxonium.t1_capacitive(j, i, Q_cap=1e5)
            if t1 is not None and t1 > 0:
                rate = 1.0 / t1
                gamma_ij[(i, j)] = rate
                gamma_ij[(j, i)] = rate

    c_ops_local = []
    for (i, j), gamma in gamma_ij.items():
        cop = (np.sqrt(gamma)) * qt.basis(levels, i) * qt.basis(levels, j).dag()
        c_ops_local.append(cop)
    return c_ops_local


def evolve(omega_d, t_g):
    global c_ops
    if c_ops is None:
        c_ops = init_c_ops()
    fluxonium = scq.Fluxonium(EJ = 8.9,
                               EC = 2.5,
                               EL = 0.5,
                               flux = 0.48,
                               cutoff = 110)
    levels = 6
    evals, evecs = fluxonium.eigensys(evals_count=levels)
    n_op_energy_basis = qt.Qobj(fluxonium.process_op(fluxonium.n_operator(), energy_esys=(evals, evecs)))
    H0 = qt.Qobj(np.diag(evals))
    A = 0.1
    drive_op = n_op_energy_basis
    H = [H0, [A * drive_op, 'cos(wd * t)']]
    args = {'wd': omega_d}
    options = qt.Options(nsteps=10000000, store_states=True, atol=1e-12, rtol=1e-11)
    propagator = qt.propagator(H, t_g, args=args, options=options, c_ops=c_ops)
    propagator_kraus = qt.to_kraus(propagator)
    propagator_2x2 = [qt.Qobj(k.full()[:2, :2]) for k in propagator_kraus]
    p_2x2_super = qt.kraus_to_super(propagator_2x2)
    fidelity = qt.average_gate_fidelity(p_2x2_super, qt.sigmax())
    return fidelity

def evolve_wrapped(params):
    omega_d, t_g = params
    return evolve(omega_d, t_g)

def _default_kwargs():
    return {"num_cpus": os.cpu_count() or 1}

def parallel_map_qutip_cleaned(task, values, task_args=tuple(), task_kwargs={}, **kwargs):
    """
    Parallel map with tqdm progress bar using pathos (mp) for multiprocessing.

    Parameters
    ----------
    task : function
        Function to apply to each element in `values`.
    values : list
        List of values to iterate over.
    task_args : tuple
        Extra positional args to pass to `task`.
    task_kwargs : dict
        Extra keyword args to pass to `task`.
    num_cpus : int (via kwargs)
        Number of parallel processes to use.

    Returns
    -------
    List of results from task(value, *task_args, **task_kwargs)
    """
    import pathos.multiprocessing as mp  # moved inside to avoid notebook import error

    os.environ["QUTIP_IN_PARALLEL"] = "TRUE"

    # Set number of CPUs
    kw = _default_kwargs()
    if "num_cpus" in kwargs:
        kw["num_cpus"] = kwargs["num_cpus"]

    # Create pool
    pool = mp.Pool(processes=kw["num_cpus"])

    try:
        # Wrap task with extra arguments
        def wrapped(val):
            return task(val, *task_args, **task_kwargs)

        # Run with tqdm for progress tracking
        results = list(tqdm(pool.imap(wrapped, values), total=len(values)))

        pool.close()
        pool.join()
    except KeyboardInterrupt as e:
        pool.terminate()
        pool.join()
        os.environ["QUTIP_IN_PARALLEL"] = "FALSE"
        raise e

    os.environ["QUTIP_IN_PARALLEL"] = "FALSE"
    return results


def varg_opt(data, axis=None, opt_fun=np.nanargmin):
    """
    Return an index of a (possibly) multi-dimensional array of the element that
    optimizes a given function along with the optimal value.
    """
    index = arg_opt(data, axis=axis, opt_fun=opt_fun)
    return index, data[index]

######################################################################################

fluxonium = scq.Fluxonium(EJ = 8.9,
                               EC = 2.5,
                               EL = 0.5,
                               flux = 0.48,
                               cutoff = 110)

levels = 6

evals, evecs = fluxonium.eigensys(evals_count=levels)

n_op_energy_basis = qt.Qobj(fluxonium.process_op(fluxonium.n_operator(), energy_esys=(evals, evecs)))

H0 = qt.Qobj(np.diag(evals))

A = 0.1
drive_op = n_op_energy_basis

omega_d = evals[1] - evals[0]
H = [H0, [A * drive_op, 'cos(wd * t)']]
args = {'wd': omega_d}



################################################################

# for parallel map
try:
    # pathos implementation is much more robust - should install if not present
    import pathos.multiprocessing as mp
except ImportError:
    # but default to std library version
    print(
        "using std lib version of multiprocessing; consider installing pathos; it's much more robust"
    )
    import multiprocessing as mp


if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    #Noise
    gamma_ij = {}
    for j in range(1, levels):
        for i in range(j):
            t1 = fluxonium.t1_capacitive(j, i, Q_cap=1e5)
            if t1 is not None and t1 > 0:
                rate = 1.0 / t1
                gamma_ij[(i, j)] = rate
                gamma_ij[(j, i)] = rate  

    c_ops = []
    for (i, j), gamma in gamma_ij.items():
        # |i><j| operator
        cop = (np.sqrt(gamma)) * qt.basis(levels, i) * qt.basis(levels, j).dag()
        c_ops.append(cop)


    #arrays
    omega_d_array = np.linspace(omega_d - 0.1, omega_d + 0.1, 20)
    peak_time_noise = 559.5559555955596 #defined from earlier runs
    t_g_array = np.linspace(0.8 * peak_time_noise, 1.2 * peak_time_noise, 20)
    param_pairs = list(itertools.product(omega_d_array, t_g_array))
    print(f"Total simulations to run: {len(param_pairs)}")


    results_flat = parallel_map_qutip_cleaned(evolve_wrapped, param_pairs, num_cpus=12)
    results = np.reshape(results_flat, (len(omega_d_array), len(t_g_array)))

    print(results)

    max_idx = np.unravel_index(np.argmax(results), results.shape)
    max_value = results[max_idx]
    omega_d_best = omega_d_array[max_idx[0]]
    t_g_best = t_g_array[max_idx[1]]

    print(f"Best fidelity: {max_value}")
    print(f"Found at omega_d = {omega_d_best}, t_g = {t_g_best}")
    print(f"Indices in results array: {max_idx}")