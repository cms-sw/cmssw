def lumi_to_run(runs, events_in_sample, events_per_job):
    '''Print tuple for use in firstLuminosityBlockForEachRun'''
    n_iovs = len(runs)
    n_lumis = events_in_sample // events_per_job
    if n_lumis % n_iovs != 0:
        raise Exception('n_lumis should be evenly divisible by n_iovs.')
    pairs = []
    for i, run in enumerate(runs):
        pairs.append((run, 1 + i*n_lumis//n_iovs))
    return tuple(pairs)
