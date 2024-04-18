from Mass_Spec_Simulator import NativeMassSpecSimulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cProfile
import time

if __name__ == '__main__':
    print(f'Running simulator at {time.time()}')
    simulator = NativeMassSpecSimulator(
        monomer_masses=[25644, 30885],  # Example: TNFa and TNFR1
        resolution=1000,
        chargewidth=10,
        maxcharge=50,
        noise_level=0.000,
        Q=0.1,
        F=1,
        AO=1,
        VA=1
    )

    n_proteins = 6
    num_spectra = 1000
    print(f'Generating {num_spectra} spectra for {n_proteins} proteins')

    # Generate spectra and interaction matrices
    spectra_data = simulator.generate_spectra_parallel(n_proteins, num_spectra)
    interaction_matrices = np.array([simulator.create_interaction_matrix(n_proteins) for _ in range(num_spectra)])

    # Assuming spectra_data is an array of tuples (binned_spectrum) for each spectrum
    # Let's stack all spectra into a single array for efficient storage
    all_spectra = np.array([spectrum for spectrum in spectra_data])  # Each spectrum is assumed to be an array of length 2000

    # Save the data as an NPZ file
    np.savez('spectra.npz', spectra=all_spectra, interaction_matrices=interaction_matrices)
    print(f'Finished simulator at {time.time()}')
