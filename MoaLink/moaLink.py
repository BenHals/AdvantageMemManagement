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
    'rcd': 'meta.RCD',
    'arf': 'meta.AdaptiveRandomForest',
    'obag': 'meta.OzaBagAdwin',
    'ht': 'trees.HoeffdingTree'
}
def get_learner_string(learner, concept_limit):
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
    else:
        concept_string = ""
    print(f'{learner_string} {concept_string}')
    return f'({learner_string} {concept_string})'

def make_moa_command(stream_string, learner, concept_limit, evaluator, length, report_length, directory, is_bat = True):
    return f'java -cp {"%" if is_bat else "$"}1\moa.jar -javaagent:{"%" if is_bat else "$"}1\sizeofag-1.0.4.jar moa.DoTask "{MOA_EVALUATORS[evaluator]} -l {get_learner_string(learner, concept_limit)} -s {stream_string} -e (BenPerformanceEvaluator -w {report_length})-i {length} -f {50}" > "{directory}{os.sep}{learner}-{concept_limit}.csv"'

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