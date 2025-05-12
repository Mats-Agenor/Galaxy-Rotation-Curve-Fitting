#===================================================================\
# GALAXY ROTATION CURVE FITTING PROGRAM BASED ON SOFUE DATA (1999)   \
# Version 4.2 (includes multiprocessing) -- From: Agenor (2025).     /
#===================================================================/

import os
import emcee  
import corner
import warnings
import numpy as np
from tqdm import tqdm
import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.constants import M_sun
from scipy import optimize, constants, special

### Plot Settings ####
# Configure various matplotlib parameters for good quality plots
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams['font.size'] = 16
plt.rc('font', **{'family':'serif', 'serif':['Times']})
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['xtick.direction'] = 'in' 
mpl.rcParams['ytick.direction'] = 'in' 
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 0.79
mpl.rcParams['xtick.minor.width'] = 0.79
mpl.rcParams['ytick.major.width'] = 0.79
mpl.rcParams['ytick.minor.width'] = 0.79
plt.rcParams['figure.constrained_layout.use'] = True

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")

# Input and output directories
input_dir = './input'          # Directory containing input data files
output_dir = './output_4'      # Main output directory for results
pdf_dir = os.path.join(output_dir, 'pdfs')  # Subdirectory for PDF figures
os.makedirs(pdf_dir, exist_ok=True)        # Create directory if it doesn't exist

# Galaxy IDs and their disk masses (in solar masses)
galaxies = {
    '1097': 18e10,  # NGC 1097 with disk mass of 18×10¹⁰ M☉
    '1365': 10e10,  # NGC 1365
    '2903': 5e10,   # NGC 2903
    '4303': 8e10,   # NGC 4303 (M61)
    '5236': 11e10   # NGC 5236 (M83)
}
# These mass parameters were calculated in the TIMER survey by Gadotti et al, 2019 

# Physical constants
G = 4.30091e-6  # Gravitational constant in kpc * (km/s)² / M_sun

# Exponential disk velocity function (unchanged)
def vd(r, a, M):
    x = r / (2.0 * a)
    x = np.where(x < 1e-4, 1e-4, x)
    term = x**2 * (special.i0(x) * special.k0(x) - special.i1(x) * special.k1(x))
    v = np.sqrt(2 * G * M / a) * np.sqrt(term)
    return v

# NFW halo velocity function (unchanged)
def v_halo(r, M_halo, r_s):
    r = np.where(r < 1e-4, 1e-4, r)
    x = r / r_s
    term = np.log(1 + x) - x / (1 + x)
    denom = np.log(1 + 1) - 1 / (1 + 1)
    v = np.sqrt(G * M_halo / r * term / denom)
    return v

# Chi-squared function (unchanged)
def chi_squared(params, r, v_obs, v_err, M_disk):
    a_disco, M_halo, r_s = params
    if a_disco <= 0 or M_halo <= 0 or r_s <= 0:
        return np.inf
    v_d = vd(r, a_disco, M_disk)
    v_h = v_halo(r, M_halo, r_s)
    v_model = np.sqrt(v_d**2 + v_h**2)
    chi2 = np.sum(((v_obs - v_model) / v_err)**2)
    return chi2

# Main processing loop through all galaxies
print('Hello, we are starting the program! 😊')
for gal_id, M_disk in galaxies.items():
    print(f'\nProcessing galaxy NGC{gal_id} 🧐 ...')
    
    # Read observational data (unchanged)
    filename = f'NGC{gal_id}.dat'
    filepath = os.path.join(input_dir, filename)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = [line.strip().split() for line in lines if not line.startswith('#') and line.strip()]
    r = np.array([float(row[0]) for row in data])       # Radial distances in kpc
    v_obs = np.array([float(row[1]) for row in data])   # Observed rotation velocities
    v_err = np.full_like(v_obs, 0.5)                   # Assume constant velocity errors

    # 1. Initial optimization with new bounds
    print('1. 🤏 Performing initial minimization...')
    initial_guess = [3.0, 1e11, 10.0]  # Initial guess for [a_disk, M_halo, r_s]
    bounds = [(0.1, 20.0), (1e9, 1e13), (0.1, 100.0)]  # Physical bounds for parameters
    result = optimize.minimize(chi_squared, initial_guess, args=(r, v_obs, v_err, M_disk), 
                              bounds=bounds, method='L-BFGS-B')
    best_params = result.x

    # 2. Grid search for better initial parameters
    print('2. 🔍 Performing grid search...')
    ranges = (
        slice(0.1, 20.0, 0.1),         # Disk scale length range with small step
        slice(1e9, 1e13, 1e10),        # Halo mass range with reduced step
        slice(0.1, 100.0, 1.0)         # Scale radius range with new upper limit
    )
    grid_result = optimize.brute(chi_squared, ranges, args=(r, v_obs, v_err, M_disk),
                                full_output=True, finish=optimize.fmin, workers=20)  # Using 20 processors
    best_params = grid_result[0]

    # 3. MCMC setup for corner plots and parameter uncertainties
    print('3. 🧮 Running Markov Chains...')
    nwalkers = 20      # Number of walkers in the ensemble
    ndim = 3           # Number of parameters being fit
    nsteps = 50000     # 50,000 steps per walker (total 1,000,000 samples)

    # Log probability function for MCMC
    def log_probability(params):
        a, M, r_s = params
        # Check physical bounds
        if not (0.1 < a < 20.0 and 1e9 < M < 1e13 and 0.1 < r_s < 100.0):
            return -np.inf
        return -0.5 * chi_squared(params, r, v_obs, v_err, M_disk)

    # Initialize walkers around best parameters with small random offsets
    pos = best_params + 1e-4 * np.random.randn(nwalkers, ndim)
    
    # Parallel execution using 20 processors
    with multiprocessing.Pool(20) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    # 4. Generate corner plot showing parameter correlations
    print('4. 📈 Plotting parameter correlations...')
    samples = sampler.get_chain(discard=1000, flat=True)  # Discard burn-in samples
    
    fig = corner.corner(
        samples,
        labels=[r'R$_d$ [kpc]', r'M$_h$ [M$_{\odot}$]', r'r$_s$ [kpc]'],
        truths=best_params,
        quantiles=[0.16, 0.5, 0.84],  # Show median and 1σ ranges
        show_titles=True,
        title_kwargs={"fontsize": 10}
    )
    plt.savefig(os.path.join(output_dir, f'NGC{gal_id}_corner.png'), dpi=300)
    plt.savefig(os.path.join(pdf_dir, f'NGC{gal_id}_corner.pdf'))
    plt.close()

    # 5. Plot the final rotation curve with components
    print('5. 📉 Plotting Rotation Curve...')
    v_d = vd(r, best_params[0], M_disk)           # Disk component
    v_h = v_halo(r, best_params[1], best_params[2])  # Halo component
    v_model = np.sqrt(v_d**2 + v_h**2)            # Total model

    plt.figure()
    plt.errorbar(r, v_obs, yerr=v_err, fmt='o', color='gray', label='Data')
    plt.plot(r, v_d, 'b--', label='Disk')
    plt.plot(r, v_h, 'r--', label='Halo NFW')
    plt.plot(r, v_model, 'k-', label='Total')
    plt.xlabel('Radius [kpc]')
    plt.ylabel('Velocity [km/s]')
    plt.title(f'NGC{gal_id}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'NGC{gal_id}_curva.png'))
    plt.savefig(os.path.join(pdf_dir, f'NGC{gal_id}_curva.pdf'))
    plt.close()

    # 6. Save best-fit parameters to file
    with open(os.path.join(output_dir, f'NGC{gal_id}_parametros.txt'), 'w') as f:
        f.write(f'a_disco: {best_params[0]:.4f} kpc\n')        # Disk scale length
        f.write(f'M_halo: {best_params[1]:.4e} M_sun\n')      # Halo mass
        f.write(f'r_s: {best_params[2]:.4f} kpc\n')           # Scale radius
        f.write(f'Chi²: {grid_result[1]:.4f}\n')              # Minimum chi-squared

    print(f'Galaxy NGC{gal_id} processed successfully! ✅')

print('\nProgram completed! 👍')
