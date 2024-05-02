from Mass_Spec_Simulator import NativeMassSpecSimulator
import numpy as np
import torch 
import pickle
import psutil
import time

if __name__ == '__main__':
    print(f'Running simulator at {time.time()}')
    simulator = NativeMassSpecSimulator(
        monomer_masses=[25644, 90885],  # Masses of the monomers
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
    num_spectra = 100000
    print(f'Generating {num_spectra} spectra for {n_proteins} proteins')
    
    
    spectra_dataset = torch.zeros((num_spectra, 2000))
    interaction_matrices_dataset = torch.zeros((num_spectra, n_proteins, n_proteins))
    
# this could go in a fx in the class to reduce verbosity of the code. Can also just save as PT tensors instead of pkl
# saving this way on my own laptop for 10k took 1.4MB for interaction matrices and 800MB for spectra - pretty good.
# took 2 hours to generate 100k spectra - not bad 
# 17 mins per 10k spectra but around 40 mins less in practise 


    for i in range(num_spectra):
        binned_mz_range, binned_normalized_spectrum = simulator.generate_single_spectrum(n_proteins)
        spectra_dataset[i] = torch.tensor(binned_normalized_spectrum)
        interaction_matrix = simulator.create_interaction_matrix(n_proteins)
        interaction_matrices_dataset[i] = torch.tensor(interaction_matrix)

with open('spectra_dataset_10binned.pkl', 'wb') as f, open('interaction_matrices_10binned.pkl', 'wb') as g:
    pickle.dump(spectra_dataset, f)
    pickle.dump(interaction_matrices_dataset, g)
        

print(f'Finished simulator at {time.time()}')