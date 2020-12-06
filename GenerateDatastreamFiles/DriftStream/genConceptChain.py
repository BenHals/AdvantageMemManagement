import random
import math

class datastreamOptions:
    def __init__(self, noise, num_concepts, hard_diff, easy_diff, hard_appear, easy_appear, hard_prop, examples_per_appearence,
            stream_type, seed, gradual):
        self.noise = noise
        self.num_concepts = num_concepts
        self.hard_diff = hard_diff
        self.easy_diff = easy_diff
        self.hard_appearences = hard_appear
        self.easy_appearences = easy_appear
        self.hard_proportion = hard_prop
        self.examples_per_appearence = examples_per_appearence
        self.stream_type = stream_type
        self.seed = seed
        self.gradual = gradual

class conceptOccurence:
    """
    Represents a concept in a stream
    """
    def __init__(self, id, difficulty, noise, appearences, examples_per_appearence):
        self.id = id
        self.difficulty = difficulty
        self.noise = noise
        self.appearences = appearences
        self.examples_per_appearence = examples_per_appearence
    
    def __repr__(self):
        # return f"{self.difficulty}"
        return f"<id: {self.id}, difficulty: {self.difficulty}, noise: {self.noise}, appearences: {self.appearences}, e_p_a: {self.examples_per_appearence}"

def genConceptChain(concept_desc, sequential):
    """
    Given a list of availiable concepts, generate a list of indexes representing a random ordering.
    """
    concept_chain = []
    num_samples = 0
    more_appearences = True
    appearence = 0
    while more_appearences:
        concepts_still_to_appear = []
        for cID in concept_desc:
            concept = concept_desc[cID]
            if concept.appearences > appearence:
                concepts_still_to_appear.append(concept)
        more_appearences = len(concepts_still_to_appear) > 0
        for concept in concepts_still_to_appear:
            concept_chain.append(concept.id)
            num_samples += concept.examples_per_appearence
        appearence += 1
    # for cID in concept_desc:
    #     concept = concept_desc[cID]
    #     concept_chain += [concept.id] * (concept.appearences)
    #     num_samples += (concept.appearences * concept.examples_per_appearence)
    if not sequential:
        random.shuffle(concept_chain)
    return concept_chain, num_samples


def generateExperimentConceptChain(ds_options, sequential):
    """
    Generates a list of concepts for a datastream given.
    """
    num_hard = math.floor(ds_options.hard_proportion * ds_options.num_concepts)
    num_easy = ds_options.num_concepts - num_hard

    concept_desc = {}
    cID = 0
    for i in range(num_hard):
        concept = conceptOccurence(cID, ds_options.hard_diff, ds_options.noise, ds_options.hard_appearences, ds_options.examples_per_appearence) 
        concept_desc[cID] = concept
        cID += 1
    for i in range(num_easy):
        concept = conceptOccurence(cID, ds_options.easy_diff, ds_options.noise, ds_options.easy_appearences, ds_options.examples_per_appearence) 
        concept_desc[cID] = concept
        cID += 1
    cc, ns = genConceptChain(concept_desc, sequential)
    return cc, ns, concept_desc


if __name__ == '__main__':
    ds_options = datastreamOptions(0, 10, 3, 0, 10, 10, 0.2, 4000)
    cc, disc = generateExperimentConceptChain(ds_options)
    print([str(disc[x]) for x in cc])