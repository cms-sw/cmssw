from __future__ import print_function
import FWCore.ParameterSet.Config as cms

# function to modify an existing Cms Source in order to set the run number
# aimed at setting non trivial run number in MC production - be it GEN-SIM or DIGI step

def throwAndSetRandomRun(source,runsAndProbs):
    """Pass a list of tuple pairs, with the first item of the pair a run number
    and the second number of the pair a weight.  The function will normalize the
    weights so you do not have to worry about that.  The pairs will be used to randomly choose what Run
    should be assigned to the job.
    """
    from random import SystemRandom
    totalProb = 0.
    for r,p in runsAndProbs:
        totalProb+=p
    #this is the same random generator used to set the seeds for the RandomNumberGeneratorService
    random = SystemRandom()
    runProb = random.uniform(0,totalProb)
    sumProb = 0
    runNumber = 0
    for r,p in runsAndProbs:
        sumProb+=p
        if sumProb >= runProb:
            runNumber = r
            break
    print('setting runNumber to: ',runNumber)
    if source.type_() == "PoolSource":
        source.setRunNumber = cms.untracked.uint32(runNumber)
    else:
        #sources that inherit from ConfigurableInputSource use 'firstRun'
        source.firstRun = cms.untracked.uint32(runNumber)

    return
