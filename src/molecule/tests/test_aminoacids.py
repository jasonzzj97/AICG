# Author: Brooke Husic

import numpy as np
import itertools

from cgnet.molecule import (CGMolecule, RESIDUE_RADII,
                            calculate_hard_sphere_minima)


def test_angstrom_conversion():
    # This tests in a somewhat roundabout way whether the angstrom
    # conversion from the master dictionary is correct

    # Create a CG model using alpha carbons only
    all_residues = list(RESIDUE_RADII.keys())
    doubled_res_list = np.concatenate(np.vstack([all_residues,
                                                 all_residues]).T)
    names = ['CA'] * len(doubled_res_list)
    resseq = np.arange(1, len(doubled_res_list)+1)
    resmap = {i+1: doubled_res_list[i] for i in range(len(doubled_res_list))}
    mypeptide = CGMolecule(names, resseq, resmap)

    # Enumerate only (CA, CA) pairs when each CA corresponds to the same
    # type of residue
    same_res_pairs = [(i, i+1) for i in range(len(doubled_res_list))
                      if i % 2 == 0]

    # Calculate the minima with a prefactor of 1.0
    same_res_minima = calculate_hard_sphere_minima(same_res_pairs,
                                                   mypeptide,
                                                   units='Angstroms',
                                                   prefactor=1.0)

    # The minima should be the radii in nanometers after a factor of 1/20
    single_nm_radii = [i/20 for i in same_res_minima]

    values_from_dict = [RESIDUE_RADII[res] for i, res in
                        enumerate(doubled_res_list) if i % 2 == 0]

    np.testing.assert_allclose(values_from_dict, single_nm_radii)


def test_minima_calculation_values():
    # This is a manual test of the minima calculations

    # Shuffle the twenty amino acids. We'll used the firt entrie of the
    # shuffled lit to make a random peptide.
    possible_residues = list(RESIDUE_RADII.keys())
    np.random.shuffle(possible_residues)

    # Make a CA only CGMolecule object with a random number of residues
    num_residues = np.random.randint(3, 10)
    names = ['CA'] * num_residues
    resseq = np.arange(1, num_residues+1)
    resmap = {i+1: possible_residues[i] for i in range(num_residues)}
    mypeptide = CGMolecule(names, resseq, resmap)

    # Enumerate all the residue-residue pairs
    pairs = list(itertools.combinations(np.arange(num_residues), 2))

    # Designate a random prefactor (i.e., scaling factor for each radius
    # in the calculation
    prefactor = np.random.uniform(0.5, 1.3)

    # Perform the manual calculation
    manual_distances = []
    for pair in pairs:
        # The +1 is needed because the resmap isn't zero-indexed
        rad1 = RESIDUE_RADII[resmap[pair[0] + 1]]
        rad2 = RESIDUE_RADII[resmap[pair[1] + 1]]
        # The *10 converts to angstroms
        manual_distances.append(prefactor*rad1*10 + prefactor*rad2*10)

    # Perform the automatic calculation
    distances = calculate_hard_sphere_minima(pairs, mypeptide,
                                             prefactor=prefactor)

    # The high tolerance is due to the significant figures in the
    # master list
    np.testing.assert_allclose(manual_distances, distances, rtol=1e-4)


def test_CA_vs_CB_minima_correspondence():
    # This tests that CA-CA distances are the same as CB-CB for the same
    # residue pair

    # Shuffle the twenty amino acids. We'll used the firt entrie of the
    # shuffled lit to make a random peptide.
    possible_residues = list(RESIDUE_RADII.keys())
    np.random.shuffle(possible_residues)

    # Make a CA+CB CGMolecule object with a random number of residues
    # Note that this might involve a GLY having a CB - this is fine
    num_residues = np.random.randint(3, 10)
    names = ['CA', 'CB'] * num_residues
    resseq = list(np.concatenate([np.repeat(i+1, 2)
                                  for i in range(num_residues)]))
    resmap = {i+1: possible_residues[i] for i in range(num_residues)}
    mypeptide = CGMolecule(names, resseq, resmap)

    # Enumerate each set of inds
    CA_inds = [i for i in range(num_residues*2) if i % 2 == 0]
    CB_inds = [i for i in range(num_residues*2) if i % 2 == 1]

    # Enumerate one set of all CA-CA pairs and one set of all CB-CB pairs
    CA_CA_pairs = list(itertools.combinations(CA_inds, 2))
    CB_CB_pairs = list(itertools.combinations(CB_inds, 2))

    # Calculate each set of minima
    CA_CA_minima = calculate_hard_sphere_minima(CA_CA_pairs, mypeptide)
    CB_CB_minima = calculate_hard_sphere_minima(CB_CB_pairs, mypeptide)

    # Ensure equality
    np.testing.assert_array_equal(CA_CA_minima, CB_CB_minima)


def test_intra_residue_zeros():
    # This tests that the minimum distance between atoms within the same
    # residue returns zero

    # Shuffle the twenty amino acids. We'll used the firt entrie of the
    # shuffled lit to make a random peptide.
    possible_residues = list(RESIDUE_RADII.keys())
    np.random.shuffle(possible_residues)

    # Make a CA+CB CGMolecule object with a random number of residues
    num_residues = np.random.randint(3, 10)
    names = ['CA', 'CB'] * num_residues
    resseq = list(np.concatenate([np.repeat(i+1, 2)
                                  for i in range(num_residues)]))
    resmap = {i+1: possible_residues[i] for i in range(num_residues)}
    mypeptide = CGMolecule(names, resseq, resmap)

    # Enumerate the intraresidue CA-CB pairs
    intra_res_pairs = [(i, i+1) for i in range(num_residues - 1) if i % 2 == 0]

    should_be_zeros = calculate_hard_sphere_minima(intra_res_pairs, mypeptide)

    # Test that a zero is returned for each residue
    np.testing.assert_array_equal(should_be_zeros,
                                  np.zeros(num_residues // 2))
