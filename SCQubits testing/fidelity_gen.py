import qutip as qt
import numpy as np
import scqubits as scq
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import itertools
import warnings

warnings.filterwarnings('ignore', module='qutip')
scq.settings.T1_DEFAULT_WARNING=False

# Global variables to store precomputed values
_fluxonium_cache = {}
_system_cache = {}

def get_fluxonium_system(EJ=8.9, EC=2.5, EL=0.5, flux=0.48, cutoff=110, levels=6):
    """Cache fluxonium system to avoid recomputation"""
    cache_key = (EJ, EC, EL, flux, cutoff, levels)
    
    if cache_key not in _fluxonium_cache:
        fluxonium = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, flux=flux, cutoff=cutoff)
        evals, evecs = fluxonium.eigensys(evals_count=levels)
        n_op_energy_basis = qt.Qobj(fluxonium.process_op(fluxonium.n_operator(), energy_esys=(evals, evecs)))
        H0 = qt.Qobj(np.diag(evals))
        
        _fluxonium_cache[cache_key] = {
            'fluxonium': fluxonium,
            'evals': evals,
            'evecs': evecs,
            'n_op_energy_basis': n_op_energy_basis,
            'H0': H0
        }
    
    return _fluxonium_cache[cache_key]

def get_collapse_operators(fluxonium, levels, Q_cap=1e5):
    """Cache collapse operators to avoid recomputation"""
    cache_key = (id(fluxonium), levels, Q_cap)
    
    if cache_key not in _system_cache:
        gamma_ij = {}
        for j in range(1, levels):
            for i in range(j):
                t1 = fluxonium.t1_capacitive(j, i, Q_cap=Q_cap)
                if t1 is not None and t1 > 0:
                    rate = 1.0 / t1
                    gamma_ij[(i, j)] = rate
                    gamma_ij[(j, i)] = rate

        c_ops = []
        for (i, j), gamma in gamma_ij.items():
            cop = (np.sqrt(gamma)) * qt.basis(levels, i) * qt.basis(levels, j).dag()
            c_ops.append(cop)
        
        _system_cache[cache_key] = c_ops
    
    return _system_cache[cache_key]

def evolve(omega_d, t_g, system_data, c_ops, A=0.1):
    """
    Optimized evolve function that reuses precomputed system data
    """
    H0 = system_data['H0']
    drive_op = system_data['n_op_energy_basis']
    
    # Build Hamiltonian
    H = [H0, [A * drive_op, 'cos(wd * t)']]
    args = {'wd': omega_d}
    
    # Use fewer options for better performance
    options = qt.Options(nsteps=1000000, store_states=False, atol=1e-10, rtol=1e-9)
    
    # Compute propagator with collapse operators
    propagator = qt.propagator(H, t_g, args=args, options=options, c_ops=c_ops)
    propagator_kraus = qt.to_kraus(propagator)
    
    # Extract 2x2 subspace
    propagator_2x2 = [qt.Qobj(k.full()[:2, :2]) for k in propagator_kraus]
    p_2x2_super = qt.kraus_to_super(propagator_2x2)
    
    # Calculate fidelity
    fidelity = qt.average_gate_fidelity(p_2x2_super, qt.sigmax())
    return fidelity

def evolve_wrapped(params):
    """Wrapper for parallel execution"""
    omega_d, t_g = params
    # Get cached system data and collapse operators
    system_data = get_fluxonium_system()
    c_ops = get_collapse_operators(system_data['fluxonium'], 6)
    return evolve(omega_d, t_g, system_data, c_ops)

def _default_kwargs():
    return {"num_cpus": os.cpu_count() or 1}

def parallel_map_qutip_cleaned(task, values, task_args=tuple(), task_kwargs={}, **kwargs):
    """
    Parallel map with tqdm progress bar using pathos for multiprocessing.
    """
    import pathos.multiprocessing as mp

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

def save_results(results, omega_d_array, t_g_array, filename='optimization_results.npz'):
    """Save results to avoid recomputation"""
    np.savez(filename, 
             results=results,
             omega_d_array=omega_d_array,
             t_g_array=t_g_array)

def load_results(filename='optimization_results.npz'):
    """Load previously computed results"""
    try:
        data = np.load(filename)
        return data['results'], data['omega_d_array'], data['t_g_array']
    except FileNotFoundError:
        return None, None, None

######################################################################################

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # System parameters
    EJ, EC, EL, flux, cutoff, levels = 8.9, 2.5, 0.5, 0.48, 110, 6
    
    # Initialize system (cached)
    system_data = get_fluxonium_system(EJ, EC, EL, flux, cutoff, levels)
    fluxonium = system_data['fluxonium']
    evals = system_data['evals']
    omega_d = evals[1] - evals[0]
    
    # Get collapse operators (cached)
    c_ops = get_collapse_operators(fluxonium, levels)
    
    # Check if results already exist
    results_file = 'optimization_results.npz'
    results, omega_d_array, t_g_array = load_results(results_file)
    
    if results is None:
        print("No cached results found. Running optimization...")
        
        # Parameter arrays
        omega_d_array = np.linspace(omega_d - 0.1, omega_d + 0.1, 20)
        peak_time_noise = 559.5559555955596
        t_g_array = np.linspace(0.8 * peak_time_noise, 1.2 * peak_time_noise, 20)
        param_pairs = list(itertools.product(omega_d_array, t_g_array))
        
        print(f"Total simulations to run: {len(param_pairs)}")
        
        # Run parallel optimization
        results_flat = parallel_map_qutip_cleaned(evolve_wrapped, param_pairs, num_cpus=4)
        results = np.reshape(results_flat, (len(omega_d_array), len(t_g_array)))
        
        # Save results for future use
        save_results(results, omega_d_array, t_g_array, results_file)
        print(f"Results saved to {results_file}")
    else:
        print("Loaded cached results from previous run.")
    
    print("Results shape:", results.shape)
    print("Results:\n", results)
    
    # Find optimal parameters
    max_idx = np.unravel_index(np.argmax(results), results.shape)
    max_value = results[max_idx]
    omega_d_best = omega_d_array[max_idx[0]]
    t_g_best = t_g_array[max_idx[1]]
    
    print(f"\nOptimization Results:")
    print(f"Best fidelity: {max_value:.6f}")
    print(f"Found at omega_d = {omega_d_best:.6f}, t_g = {t_g_best:.6f}")
    print(f"Indices in results array: {max_idx}")
    
    # Optional: Plot results
    # try:
    #     plt.figure(figsize=(10, 8))
    #     plt.contourf(t_g_array, omega_d_array, results, levels=50)
    #     plt.colorbar(label='Fidelity')
    #     plt.xlabel('Gate Time (t_g)')
    #     plt.ylabel('Drive Frequency (omega_d)')
    #     plt.title('Fidelity Optimization Results')
    #     plt.plot(t_g_best, omega_d_best, 'r*', markersize=15, label=f'Best: {max_value:.4f}')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
    #     plt.show()
    # except Exception as e:
    #     print(f"Could not create plot: {e}")