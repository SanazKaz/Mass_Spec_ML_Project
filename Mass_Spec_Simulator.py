import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NativeMassSpecSimulator:
    def __init__(self, monomer_masses, resolution, chargewidth, maxcharge, noise_level, Q, F, AO, VA):
        self.monomer_masses = monomer_masses
        self.resolution = resolution
        self.chargewidth = chargewidth
        self.maxcharge = maxcharge
        self.noise_level = noise_level
        self.Q = Q
        self.F = F
        self.AO = AO
        self.VA = VA
        #self.run_counter = 0  # Initialize run counter

    def simulate_complex_spectrum(self, complex_mass):
        """ Simulate a mass spectrum for a single complex"""

        mz_range = np.arange(1, 20001)
        spectrum = np.zeros_like(mz_range, dtype=float)

        MS = complex_mass
        MA = self.Q * MS**0.76
        ME = MS + MA
        ZA = 0.0467 * ME**0.533 + self.F
        TT = np.exp(-(np.arange(self.maxcharge) - 50)**2 / self.chargewidth)
        sumT = np.sum(TT)
        WC = np.zeros(self.maxcharge)
        DE = np.zeros_like(WC)

        for charge in range(self.maxcharge):
            WC[charge] = np.exp(-(charge + 1 - ZA)**2 / self.chargewidth) / sumT
            DE[charge] = (1 - np.exp(-1620 * (9.1 * (charge + 1) / ME)**1.75)) * self.AO * self.VA if ME > 0 else 0

        WD = WC * DE

        for charge in range(1, self.maxcharge + 1):
            mz = ME / charge
            if mz < len(mz_range):
                lower_limit = max(1, int(mz - self.resolution / 10))
                upper_limit = min(len(mz_range), int(mz + self.resolution / 10))
                for axis in range(lower_limit, upper_limit):
                    spread = np.exp(-((axis - mz)**2) / (2 * (self.resolution / 100)**2))
                    spectrum[axis] += WD[charge - 1] * spread

        noise = np.random.normal(0, self.noise_level, size=spectrum.size)
        spectrum += noise

        return spectrum
    
    def simulate_mass_spectrum(self, interaction_matrix):
        """ Simulate a mass spectrum with all complexes based on the interaction matrix"""
        mz_range = np.arange(1, 20001)
        combined_spectrum = np.zeros_like(mz_range, dtype=float)
        #peak_labels = []

        for i, j in np.argwhere(interaction_matrix > 0):
            print("non zero from def:", np.argwhere(interaction_matrix > 0))

            stoich_A = i  # Stoichiometry of Protein A
            stoich_B = j  # Stoichiometry of Protein B
            complex_mass = stoich_A * self.monomer_masses[0] + stoich_B * self.monomer_masses[1]
            
            
            spectrum = self.simulate_complex_spectrum(complex_mass)

            # Scale the spectrum by the interaction matrix value
            scaled_spectrum = spectrum * interaction_matrix[i, j] 

            # Generate the peak label based on stoichiometry
            #peak_label = f"{stoich_A +1}A_{stoich_B +1}B_{interaction_matrix[i, j]}"
            #peak_labels.append(peak_label)

            combined_spectrum += scaled_spectrum
            
        # Normalize the combined spectrum after summing contributions from all interactions
        total_intensity = np.sum(combined_spectrum)
        normalized_spectrum = combined_spectrum / total_intensity if total_intensity > 0 else combined_spectrum
            ## if there is only 1 element in the matrix then it is is scaled to 1 by the normlisation
            ## need to sort this out


        #print("normalized_spectrum:", normalized_spectrum)
        #print("peak_labels:", peak_labels)
        
        return mz_range, normalized_spectrum #, peak_labels
    

    def simulate_single_scaled(self, interaction_matrix, spectra):

        """to simulate a single spectrum from the predicted 
        matrices for overlaying with  original spectra"""


        mz_range = np.arange(1, 20001)
        combined_spectrum = np.zeros_like(mz_range, dtype=float)

        for i, j in np.argwhere(interaction_matrix > 0):

            stoich_A = i  # Stoichiometry of Protein A
            stoich_B = j  # Stoichiometry of Protein B
            complex_mass = stoich_A * self.monomer_masses[0] + stoich_B * self.monomer_masses[1]
            
            
            spectrum = self.simulate_complex_spectrum(complex_mass)

            # Scale the spectrum by the interaction matrix value
            scaled_spectrum = spectrum * interaction_matrix[i, j] 

            combined_spectrum += scaled_spectrum
            total_intensity = np.sum(spectra) # adding all 2000 intensities then dividing new spec by this
            
        # Normalize the combined spectrum after summing contributions from all interactions
        normalized_spectrum = combined_spectrum / total_intensity
            ## if there is only 1 element in the matrix then it is is scaled to 1 by the normlisation
            ## need to sort this out
        
        ## need to bin it here 
        binned_normalised_spectrum = np.zeros(2001)
        bin_counts = np.zeros(2001)

        for mz, intensity in zip(mz_range, normalized_spectrum):
            bin_idx = int(mz // 10)
            if bin_idx < 2001:
                binned_normalised_spectrum[bin_idx] += intensity ## this is added the intensity to the bin index ? kind of dum
                bin_counts[bin_idx] += 1
        
        mask = bin_counts > 0
        binned_normalised_spectrum[mask] /= bin_counts[mask]

        binned_mz_range = np.arange(0, 20010, 10)
        return binned_mz_range, binned_normalised_spectrum




    def create_interaction_matrix(self, n_proteins): 
        """Create a random square interaction matrix with x by x """

        # Protein A is column B is the row
        interaction_matrix = np.random.uniform(0, 1, (n_proteins, n_proteins))
        #interaction_matrix = np.where(interaction_matrix > 0, np.random.randint(1, 7, interaction_matrix.shape), interaction_matrix)
        nonzeros = np.random.randint(3, 12) # this determines how many non zero stoichs you'll have 
        if nonzeros > n_proteins * n_proteins: # if the number of non zeros is greater than the number of elements in the matrix then set it between the range of rows and cols
            nonzeros = np.random.randint(1, n_proteins * n_proteins) # to avoid the case where the number of non zeros is greater than the number of elements in the matrix
        idx = np.random.choice(np.arange(n_proteins * n_proteins), size=(n_proteins * n_proteins - nonzeros), replace=False)
        interaction_matrix.ravel()[idx] = 0
        interaction_matrix[0, 0] = 0

        return interaction_matrix


    def generate_single_spectrum(self, n_proteins):
        """Generate a single mass spectrum for n_proteins n_proteins
        main def that is called to generate the spectra for the dataset"""


        interaction_matrix = self.create_interaction_matrix(n_proteins)
        mz_range, normalized_spectrum = self.simulate_mass_spectrum(interaction_matrix)
        
        binned_normalised_spectrum = np.zeros(2001)
        bin_counts = np.zeros(2001)

        for mz, intensity in zip(mz_range, normalized_spectrum):
            bin_idx = int(mz // 10)
            if bin_idx < 2001:
                binned_normalised_spectrum[bin_idx] += intensity ## this is added the intensity to the bin index ? kind of dum
                bin_counts[bin_idx] += 1
        
        mask = bin_counts > 0
        binned_normalised_spectrum[mask] /= bin_counts[mask]

        binned_mz_range = np.arange(0, 20010, 10)
        return binned_mz_range, binned_normalised_spectrum




    def generate_spectrum_from_pred(self, matrix):
        interaction_matrix = matrix
        mz_range, normalized_spectrum = self.simulate_mass_spectrum(interaction_matrix)
        
        binned_normalised_spectrum = np.zeros(2001)
        bin_counts = np.zeros(2001)

        for mz, intensity in zip(mz_range, normalized_spectrum):
            bin_idx = int(mz // 10)
            if bin_idx < 2001:
                binned_normalised_spectrum[bin_idx] += intensity ## this is added the intensity to the bin index ? kind of dum
                bin_counts[bin_idx] += 1
        
        mask = bin_counts > 0
        binned_normalised_spectrum[mask] /= bin_counts[mask]

        binned_mz_range = np.arange(0, 20010, 10)
        return binned_mz_range, binned_normalised_spectrum
        
    