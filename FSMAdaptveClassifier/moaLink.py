import os
MOA_EVALUATORS = {
    'preq': 'EvaluatePrequential',
    'int':  'EvaluateInterleavedTestThenTrain',
}
MOA_BASELINES = {
    'rcd-unlimited': '(meta.RCD -c 0)',
    'rcd-limited-15': '(meta.RCD -c 15)',
    'rcd-limited-50': '(meta.RCD -c 50)',
    'rcd-limited-5': '(meta.RCD -c 5)',
    'HT': 'trees.HoeffdingTree',
}

MOA_LEARNERS = {
    # 'rcd': 'meta.RCD',
    'rcdHTEDDM': 'meta.RCD -l trees.HoeffdingTree -d EDDM',
    'rcdHT': 'meta.RCD -l trees.HoeffdingTree',
    'rcdEDDM': 'meta.RCD -d EDDM',
    'rcdBase': 'meta.RCD',
    'nb': 'bayes.NaiveBayes',
    # 'rcd': 'meta.RCD -l trees.HoeffdingTree',
    # 'rcd': 'meta.RCD -l trees.HoeffdingTree -d (ADWINChangeDetector -a 0.05)',
    'arf': 'meta.AdaptiveRandomForest',
    'obag': 'meta.OzaBagAdwin',
    'ht': 'trees.HoeffdingTree',
    'ecpf': 'meta.ECPF',
    'gp': 'meta.WEKAClassifier -l (weka.classifiers.meta.Graph -G '
}
chi_table = {
    1: 3.84,
    2: 5.99,
    3: 7.81,
    4: 9.48,
    5: 11.07,
    6: 12.59,
    7: 14.07,
    8: 15.50,
    9: 16.91,
    10: 18.31,
    11: 19.68,
    12: 21.026,
    13: 22.36,
    14: 23.68,
    15: 25.0,
    16: 26.30,
    17: 27.58,
    18: 28.87,
    19: 30.14,
    20: 31.41,
    21: 32.67,
    22: 33.92,
    23: 35.17,
    24: 36.41,
    25: 37.65,
    26: 38.89,
    27: 40.11,
    28: 41.34,
    29: 42.56,
    30: 43.77,
    31: 44.99,
    63: 82.53,
    127: 154.30,
    255: 293.25,
    511: 564.70,
    1023: 1098.52

}
def get_learner_string(learner, concept_limit, num_features = 8):
    learner_string = MOA_LEARNERS[learner]
    print(learner)
    if learner == 'rcd':
        if concept_limit != 0:
            concept_limit = max(0, concept_limit)
        concept_string = f"-c {concept_limit}" if concept_limit != None else f""
    elif learner == 'arf':
        if concept_limit != 0:
            concept_limit = max(1, concept_limit)
        concept_string = f"-s {concept_limit}" if concept_limit != None else f""
    elif learner == 'gp':
        compare_values = []
        for chi_compare in chi_table.keys():
            compare_values.append((abs(chi_compare - num_features), chi_table[chi_compare]))
        compare_values.sort(key = lambda x: x[0])
        concept_string = f"{compare_values[0][1]})"
    elif learner == 'ecpf':
        if concept_limit != 0:
            return f'(meta.ECPF -f -p {concept_limit} -l trees.HoeffdingTree -d (ADWINChangeDetector -a 0.05))'
        else:
            return f'(meta.ECPF -l trees.HoeffdingTree -d (ADWINChangeDetector -a 0.05))'
    else:
        concept_string = ""
    print(f'{learner_string} {concept_string}')
    return f'({learner_string} {concept_string})'

def make_moa_command(stream_string, learner, concept_limit, evaluator, length, report_length, directory, is_bat = True, name = "arf", num_features = 8):
    return f'java -cp {"%" if is_bat else "$"}1\moa.jar{";" if is_bat else ":"}{"%" if is_bat else "$"}1\weka-dev-3.9.2.jar -javaagent:{"%" if is_bat else "$"}1\sizeofag-1.0.4.jar moa.DoTask "{MOA_EVALUATORS[evaluator]} -l {get_learner_string(learner, concept_limit, num_features)} -s {stream_string} -e (BenPerformanceEvaluator -w {report_length})-i {length} -f {10000}" > "{directory}{os.sep}{name}.csv"'

def get_moa_stream_from_file(directory, is_bat = True):
    return f"(ArffFileStream -f ({'%' if is_bat else '$'}cd{'%' if is_bat else '$'}\saved_stream.ARFF))"
def get_moa_stream_from_filename(directory, filename):
    return f"(ArffFileStream -f ({directory}{os.sep}{filename}))"
def get_moa_stream_string(concepts):
    if len(concepts) < 1:
        return ""
    if len(concepts) == 1:
        c = concepts[0]
        concept = c[0]
        start = c[1]
        end = c[2]
        return concept.get_moa_string(start, end)
    else:
        c = concepts[0]
        concept = c[0]
        start = c[1]
        end = c[2]
        return f"(ConceptDriftStream -s {concept.get_moa_string(start, end)} -d {get_moa_stream_string(concepts[1:])} -p {end - start} -w 1)"

def save_moa_bat(moa_command, filename, is_bat = True):

    print(f"{moa_command}\n")
    with open(filename, 'w') as f:
        if not is_bat:
            f.write(f"#!/bin/sh\n")
        f.write(f"{moa_command}\n")