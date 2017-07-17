#G.Benelli Feb 7 2008
#This fragment is used to have the random generator seeds saved to test
#simulation reproducibility. Anothe fragment then allows to run on the
#root output of cmsDriver.py to test reproducibility.

import FWCore.ParameterSet.Config as cms
def customise(process):
    #Renaming the process
    process.__dict__['_Process__name']='DIGISavingSeeds'
    #Storing the random seeds
    process.rndmStore=cms.EDProducer("RandomEngineStateProducer")
    #Adding the RandomEngine seeds to the content
    process.output.outputCommands.append("keep RandomEngineStates_*_*_*")
    process.rndmStore_step=cms.Path(process.rndmStore)
    #Modifying the schedule:
    #First delete the current one:
    del process.schedule[:]
    #Then add the wanted sequences
    process.schedule.append(process.digitisation_step)
    process.schedule.append(process.rndmStore_step)
    process.schedule.append(process.out_step)
    #Adding SimpleMemoryCheck service:
    process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                          ignoreTotal=cms.untracked.int32(1),
                                          oncePerEventMode=cms.untracked.bool(True))
    #Adding Timing service:
    process.Timing=cms.Service("Timing")
    
    #Add these 3 lines to put back the summary for timing information at the end of the logfile
    #(needed for TimeReport report)
    if hasattr(process,'options'):
        process.options.wantSummary = cms.untracked.bool(True)
    else:
        process.options = cms.untracked.PSet(
            wantSummary = cms.untracked.bool(True)
        )
    
    return(process)
