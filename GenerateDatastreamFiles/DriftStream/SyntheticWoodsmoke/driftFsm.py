import random
import numpy as np

def generate_fsm_drift(num_concepts, num_drifts, seed):
    """Generates dirfts accourding to some underlying probabilities.
    
    Our system uses the assumption that there is some reason behind drifts,
    I.E a 'real' probability of moving from one state to anouther.
    Note this is not a hard assumption, but is a reasoning for extracting empirical
    statistics on drift. If there are no underlying real stats, the empirical
    ones don't mean much.
    
    Parameters
    ----------
    num_concepts: int
        The number of possible concepts
    
    num_drifts: int
        The number of concept switches
        
    seed: int
        The random seed.
        
    Returns
    -------
    
    A List of ints representing concept IDs. No spacing information."""

    np.random.seed(seed)
    transition_matrix = []
    for from_id in range(0, num_concepts):
        transition_matrix.append([])
        transition_probabilities = transition_matrix[from_id]
        for to_id in range(0, num_concepts):
            transition_probabilities.append(np.random.random())
        normalize_factor = sum(transition_probabilities)
        for i, v in enumerate(transition_probabilities):
            transition_probabilities[i] = v / normalize_factor
    
    concept_chain = []
    concept_id = 0
    for drift_index in range(0, num_drifts):
        concept_chain.append(concept_id)
        transition_probabilities = transition_matrix[concept_id]
        concept_id = np.random.choice(num_concepts, p=transition_probabilities)
    
    return concept_chain